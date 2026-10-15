"""Cranial biological-sex baseline with explicit documented reference provenance."""
from pathlib import Path
import io

import cv2
import joblib
import numpy as np
import pandas as pd

from .core import atomic, digest, image, load, new_run, save, sha
from .references import paired

VIEWS = ['CF', 'CE', 'CD']


def import_curated_reference(source, author, manual_only=True):
    source = Path(source)
    if not author.strip():
        raise ValueError('Informe o responsavel pela importacao experimental')
    frame = pd.read_csv(source, dtype=str).fillna('')
    if not {'id_imagem', 'genero_final', 'origem_genero_final'}.issubset(frame.columns):
        raise ValueError('Curadoria requer id_imagem, genero_final e origem_genero_final')
    if manual_only:
        frame = frame[frame.origem_genero_final.isin(['CORRIGIDO_MANUALMENTE', 'FAIRFACE_CONFIRMADO_MANUALMENTE'])].copy()
    frame['id_pessoa'] = frame.id_imagem.str.removesuffix('_FFN').str.strip()
    frame['sexo'] = frame.genero_final.str.upper().str.strip()
    if frame.empty or frame.id_pessoa.eq('').any() or frame.id_pessoa.duplicated().any():
        raise ValueError('Curadoria vazia ou IDs ausentes/repetidos')
    if not frame.sexo.isin(['MASCULINO', 'FEMININO']).all() or frame.origem_genero_final.eq('').any():
        raise ValueError('Rotulos ou origens invalidos na curadoria')
    frame['fonte'] = str(source)
    frame['tipo_referencia'] = 'CURADORIA_EXPLORATORIA'
    run = new_run('sex_references')
    atomic(run/'reference.csv', frame[['id_pessoa','sexo','fonte','tipo_referencia','origem_genero_final']].to_csv(index=False).encode())
    save(run/'config.json', dict(author=author.strip(), confirmed=False,
         reference_kind='CURADORIA_EXPLORATORIA', documented=False, manual_only=manual_only,
         source=str(source), source_hash=sha(source), hash=sha(run/'reference.csv'),
         origins=frame.origem_genero_final.value_counts().to_dict()))
    return run


def import_reference(frame, author, confirmed):
    needed = {'id_pessoa', 'sexo', 'fonte', 'tipo_referencia'}
    if not needed.issubset(frame.columns) or not author.strip() or not confirmed:
        raise ValueError('CSV requer id_pessoa, sexo, fonte e tipo_referencia; confirme a origem e informe o autor')
    frame = frame.copy().fillna('')
    frame['id_pessoa'] = frame.id_pessoa.astype(str).str.strip()
    frame['sexo'] = frame.sexo.astype(str).str.upper().str.strip()
    frame['tipo_referencia'] = frame.tipo_referencia.astype(str).str.upper().str.strip()
    if frame.id_pessoa.eq('').any() or frame.id_pessoa.duplicated().any():
        raise ValueError('IDs ausentes ou repetidos')
    if not frame.sexo.isin(['MASCULINO', 'FEMININO']).all():
        raise ValueError('Sexo deve ser MASCULINO ou FEMININO')
    if not frame.tipo_referencia.isin(['DOCUMENTAL', 'ESPECIALISTA']).all() or frame.fonte.astype(str).str.strip().eq('').any():
        raise ValueError('Use origem DOCUMENTAL ou ESPECIALISTA com fonte identificavel; FairFace nao e referencia documental')
    run = new_run('sex_references')
    atomic(run/'reference.csv', frame.to_csv(index=False).encode())
    save(run/'config.json', dict(author=author.strip(), confirmed=True, hash=sha(run/'reference.csv')))
    return run


def references(path):
    path = Path(path)
    if sha(path/'reference.csv') != load(path/'config.json')['hash']:
        raise ValueError('Referencia documental alterada')
    return pd.read_csv(path/'reference.csv', dtype=str).set_index('id_pessoa')


def silhouette_features(path, view):
    im = image(path)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if not mask.any():
        raise ValueError('Mascara vazia')
    # Retain all foreground; the user can inspect support/background contamination.
    ys, xs = np.where(mask > 0)
    crop = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
    from .core import letterbox
    canvas = letterbox(crop, 64)[0]
    small = cv2.resize(canvas, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)/255
    return np.r_[small, [float(view == v) for v in VIEWS]].astype(np.float32)


def aggregate(values, strategy='mean'):
    return float(np.mean(values) if strategy == 'mean' else np.median(values))


def metrics(y, probabilities, threshold):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
    predicted = np.array(probabilities) >= threshold
    return dict(accuracy=float(accuracy_score(y, predicted)), balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                f1=float(f1_score(y, predicted, zero_division=0)), matrix=confusion_matrix(y, predicted, labels=[0, 1]).tolist(), n=len(y))


def train(rows, reference, C=1., threshold_low=.3, threshold_high=.7, seed=42, stop=None, log=print):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if C <= 0 or not (0 < threshold_low <= threshold_high < 1):
        raise ValueError('Parametros invalidos')
    labels = references(reference)
    reference_kind = load(Path(reference)/'config.json').get('reference_kind', 'DOCUMENTAL')
    candidates = [p for p in paired(rows) if p['person'] in labels.index and p['split'] in ('treino', 'validacao', 'teste')]
    run = new_run('cranial_sex')
    groups = {'treino': [], 'validacao': [], 'teste': []}
    failures = []
    seen = {}
    for item in candidates:
        for view, row in item['views'].items():
            if view not in VIEWS:
                continue
            token = sha(row['path'])
            row['hash'] = token
            if token in seen and seen[token] != item['split']:
                raise ValueError('Imagem repetida entre splits')
            seen[token] = item['split']
    save(run/'inputs.json', candidates)
    for item in candidates:
        if stop and Path(stop).exists():
            save(run/'status.json', dict(state='INTERROMPIDO'))
            return run
        try:
            if not all(v in item['views'] for v in VIEWS):
                raise ValueError('Tripleta CF/CE/CD incompleta')
            X = []
            for view in VIEWS:
                path = item['views'][view]['path']
                X.append(silhouette_features(path, view))
            groups[item['split']].append(dict(person=item['person'], X=np.array(X), y=int(labels.loc[item['person'], 'sexo'] == 'MASCULINO')))
            log('Silhuetas: ' + item['person'])
        except (ValueError, OSError) as exc:
            failures.append(dict(person=item['person'], split=item['split'], error=str(exc)))
    save(run/'failures.json', failures)
    for split in ['treino', 'validacao']:
        if len(groups[split]) < 2 or len({r['y'] for r in groups[split]}) < 2:
            raise ValueError('Referencias insuficientes: ambas as classes devem existir em treino e validacao')
    X = np.concatenate([r['X'] for r in groups['treino']])
    y = np.repeat([r['y'] for r in groups['treino']], 3)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight='balanced', random_state=seed))
    model.fit(X, y)
    val = groups['validacao']
    per_view = [model.predict_proba(r['X'])[:, 1] for r in val]
    grid = []
    for strategy in ['mean', 'median']:
        probabilities = [aggregate(v, strategy) for v in per_view]
        for threshold in np.linspace(threshold_low, threshold_high, 9):
            grid.append(dict(strategy=strategy, threshold=float(threshold), **metrics([r['y'] for r in val], probabilities, threshold)))
    best = max(grid, key=lambda r: (r['balanced_accuracy'], r['f1'], -abs(r['threshold']-.5)))
    config = dict(version='cranial_sex_v2', reference_kind=reference_kind, reference=str(reference), reference_hash=sha(Path(reference)/'reference.csv'),
                  C=C, seed=seed, strategy=best['strategy'], threshold=best['threshold'],
                  train_ids=[r['person'] for r in groups['treino']], validation_ids=[r['person'] for r in val],
                  model='silhuetas_16x16_logistic_multivista')
    buf = io.BytesIO()
    joblib.dump(model, buf)
    atomic(run/'model.joblib', buf.getvalue())
    config['model_hash'] = sha(run/'model.joblib')
    save(run/'config.json', config)
    save(run/'validation_grid.json', grid)
    view_metrics = {view: metrics([r['y'] for r in val], [p[i] for p in per_view], best['threshold']) for i, view in enumerate(VIEWS)}
    save(run/'evaluation.json', dict(reference_kind=reference_kind, validation=best, by_view_validation=view_metrics, failures=failures,
                                    matrix_labels=['FEMININO', 'MASCULINO'], test_pending=True))
    save(run/'status.json', dict(state='CONCLUIDO'))
    log(f'Validacao: balanced accuracy={best["balanced_accuracy"]:.3f}; threshold={best["threshold"]:.2f}')
    return run


def apply(run, items, evaluate_test=False, stop=None, log=print):
    run = Path(run)
    config = load(run/'config.json')
    if sha(run/'model.joblib') != config['model_hash']:
        raise ValueError('Modelo alterado')
    labels = references(config['reference'])
    model = joblib.load(run/'model.joblib')
    target = new_run('cranial_sex_predictions')
    results = []
    for item in items:
        if stop and Path(stop).exists():
            save(target/'status.json', dict(state='INTERROMPIDO'))
            break
        record = dict(person=item['person'], split=item.get('split'), model=str(run), reference_kind=config.get('reference_kind', 'DOCUMENTAL'))
        try:
            if not all(v in item['views'] for v in VIEWS):
                raise ValueError('Tripleta CF/CE/CD incompleta')
            for view in VIEWS:
                row = item['views'][view]
                if row.get('hash') and row['hash'] != sha(row['path']):
                    raise ValueError('Imagem alterada desde a selecao')
            probs = {v: float(model.predict_proba(silhouette_features(item['views'][v]['path'], v)[None])[0, 1]) for v in VIEWS}
            pm = aggregate(list(probs.values()), config['strategy'])
            record.update(prob_masculino=pm, prob_feminino=1-pm, per_view=probs,
                          predicted='MASCULINO' if pm >= config['threshold'] else 'FEMININO', status='OK')
            if item['person'] in labels.index:
                truth = labels.loc[item['person'], 'sexo']
                record.update(reference=truth, matches=truth == record['predicted'],
                              evaluation='TESTE_INDEPENDENTE' if evaluate_test and item.get('split') == 'teste' else 'CONCORDANCIA')
                if config.get('reference_kind') == 'CURADORIA_EXPLORATORIA':
                    record['evaluation'] = 'CONCORDANCIA_COM_CURADORIA_EXPLORATORIA'
        except (ValueError, OSError) as exc:
            record.update(status='ERRO', error=str(exc))
        results.append(record)
        save(target/'predictions.json', results)
        log(item['person'] + ': ' + record['status'])
    if evaluate_test:
        valid = [r for r in results if r['status'] == 'OK' and r.get('reference') and r['split'] == 'teste']
        if valid:
            save(target/'metrics.json', dict(reference_kind=config.get('reference_kind', 'DOCUMENTAL'), **metrics([r['reference'] == 'MASCULINO' for r in valid], [r['prob_masculino'] for r in valid], config['threshold']),
                                              eligible_test=sum(r.get('split') == 'teste' for r in results), failures=len(results)-len(valid),
                                              by_view={v: metrics([r['reference'] == 'MASCULINO' for r in valid], [r['per_view'][v] for r in valid], config['threshold']) for v in VIEWS}))
    save(target/'config.json', dict(run=str(run), model_hash=config['model_hash'], evaluate_test=evaluate_test))
    if not (target/'status.json').exists():
        save(target/'status.json', dict(state='CONCLUIDO'))
    return target
