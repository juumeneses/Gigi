from __future__ import annotations

import base64
import html
import io
import random
from .portable import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .backbone import Extractor
from .core import (ARTIFACTS, ROOT, EXPERIMENTS, VERSION, VIEWS, atomic, check_stop, checkpoint,
                   csv_write, digest, letterbox, load, new_run, np_save, read_image, save, sha)
from .data import cohort


def configured_extractor(config, device='auto'):
    name = config.get('backbone', 'dinov2_vits14')
    if config.get('preprocessing', 'original') != 'original':
        return Extractor(config['size'], device, backbone=name, preprocessing=config['preprocessing'])
    return Extractor(config['size'], device) if name == 'dinov2_vits14' else Extractor(config['size'], device, backbone=name)


class Towers(nn.Module):
    def __init__(self, dimension=384):
        super().__init__()
        self.face = nn.Sequential(nn.Linear(dimension, 256), nn.GELU(), nn.Linear(256, 128))
        self.skull = nn.Sequential(nn.Linear(dimension, 256), nn.GELU(), nn.Linear(256, 128))
        self.view = nn.Embedding(6, dimension)
        nn.init.zeros_(self.view.weight)

    def forward(self, features, mask, views, modality):
        x = features + self.view(torch.tensor([VIEWS.index(v) for v in views], device=features.device))[None]
        x = (self.face if modality == 'face' else self.skull)(x)
        x = (x * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[..., None]
        return F.normalize(x, dim=-1)


def triplet(face, skull, margin=.2):
    distances = 1-face@skull.T
    values = []
    for matrix in (distances, distances.T):
        for i in range(len(matrix)):
            positive = matrix[i, i]
            negative = torch.cat([matrix[i, :i], matrix[i, i+1:]])
            semi = negative[(negative > positive) & (negative < positive+margin)]
            # No semi-hard example: take the hardest negative, never the positive.
            values.append(F.relu(positive-(semi.min() if len(semi) else negative.min())+margin))
    return torch.stack(values).mean()


def features(people, experiment, extractor, stop=None, log=print):
    names = EXPERIMENTS[experiment]
    data, failures = {}, []
    for modality, views in zip(('face', 'skull'), names):
        values = np.zeros((len(people), len(views), getattr(extractor, 'dimension', 384)), dtype=np.float32)
        mask = np.zeros((len(people), len(views)), dtype=np.float32)
        for i, person in enumerate(people):
            for j, view in enumerate(views):
                check_stop(stop)
                row = person['views'].get(view)
                try:
                    if row is None:
                        raise ValueError('VISTA_AUSENTE')
                    values[i, j] = extractor.feature(row)
                    mask[i, j] = 1
                except Exception as exc:
                    failures.append(dict(person=person['person'], view=view, error=str(exc)))
            if i % 20 == 0:
                log(f'Caracteristicas {modality}: {i+1}/{len(people)}', flush=True)
        data[modality] = (torch.tensor(values), torch.tensor(mask))
    return data, failures


def encode(model, data, experiment, indices=None):
    values = []
    for modality, views in zip(('face', 'skull'), EXPERIMENTS[experiment]):
        x, mask = data[modality]
        if indices is not None:
            x, mask = x[indices], mask[indices]
        if model is None:
            y = F.normalize((x*mask[..., None]).sum(1)/mask.sum(1).clamp_min(1)[..., None], dim=-1)
        else:
            y = model(x, mask, views, modality)
        values.append((y, mask.sum(1) > 0))
    return values


def score_matrix(model, data, experiment, indices=None):
    with torch.no_grad():
        (face, fm), (skull, sm) = encode(model, data, experiment, indices)
        scores = (face@skull.T).numpy()
        scores[~fm.numpy(), :] = -np.inf
        scores[:, ~sm.numpy()] = -np.inf
    return scores


def metrics(rows, field='rank'):
    ranks = [r[field] for r in rows]
    rr = np.array([1/r if r else 0. for r in ranks])
    rng = np.random.default_rng(42)
    boot = [float(rr[rng.integers(len(rr), size=len(rr))].mean()) for _ in range(1000)] if len(rr) else [0.]
    valid = [r for r in ranks if r]
    return dict(n=len(rows), failures=len(rows)-len(valid), coverage=len(valid)/max(1, len(rows)),
                mrr=float(rr.mean()) if len(rr) else 0., mrr_ci95=np.quantile(boot, [.025, .975]).tolist(),
                median_rank=float(np.median(valid)) if valid else None,
                **{f'recall_{k}': sum(r is not None and r <= k for r in ranks)/max(1, len(rows)) for k in (1, 3, 5, 10)})


def evaluation(scores, people, gallery_seed=2026):
    if len(people) != scores.shape[0] or scores.shape != (len(people), len(people)):
        raise ValueError('Galeria e matriz incompativeis')
    ids = [p['person'] for p in people]
    rows, galleries = [], {}
    size = min(10, len(ids))
    for i, person in enumerate(people):
        candidates = list(range(len(ids)))
        order = sorted(candidates, key=lambda j: (-float(scores[i, j]), ids[j]))
        rng = random.Random(digest([gallery_seed, ids[i], ids]))
        small = sorted(rng.sample([j for j in candidates if j != i], size-1)+[i])
        galleries[ids[i]] = [ids[j] for j in small]
        small_order = sorted(small, key=lambda j: (-float(scores[i, j]), ids[j]))
        finite = np.isfinite(scores[i, i])
        rows.append(dict(person=ids[i], year=person['year'], rank=order.index(i)+1 if finite else None,
                         gallery_rank=small_order.index(i)+1 if finite else None,
                         top_candidate=ids[order[0]] if order and np.isfinite(scores[i, order[0]]) else None,
                         correct_score=float(scores[i, i]) if finite else None))
    return dict(full=metrics(rows), gallery=metrics(rows, 'gallery_rank'), gallery_size=size,
                gallery_seed=gallery_seed, galleries=galleries, rows=rows,
                by_year={year: metrics([r for r in rows if r['year'] == year]) for year in sorted({p['year'] for p in people})})


def export_report(run, result, people):
    csv_write(run/'rankings.csv', result['rows'])
    chunks = ['<!doctype html><html lang="pt"><meta charset="utf-8"><title>DINOv2: resultados</title>',
              '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto}img{width:220px;height:220px;object-fit:contain;background:#eee}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style>',
              '<h1>Matching visual DINOv2</h1>', '<p>Sem landmarks obrigatorios. Similaridade nao e probabilidade de identidade.</p>',
              '<pre>'+html.escape(str(result['full']))+'</pre>', '<h2>Exemplos</h2>']
    lookup = {p['person']: p for p in people}
    examples = sorted(result['rows'], key=lambda r: r['rank'] or 10**9)
    examples = (examples[:3]+examples[-3:]) if len(examples) > 6 else examples
    for row in examples:
        chunks.append('<h3>'+html.escape(row['person'])+' · posicao '+str(row['rank'])+'</h3>')
        for pid, view in [(row['person'], 'FFN'), (row['person'], 'CF'), (row['top_candidate'], 'CF')]:
            source = lookup.get(pid, {}).get('views', {}).get(view)
            if source:
                try:
                    im = letterbox(read_image(source['path']), 224)[0]
                    buf = io.BytesIO(); im.save(buf, format='JPEG')
                    chunks.append('<img alt="'+html.escape(str(pid)+' '+view)+'" src="data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()+'">')
                except (OSError, ValueError):
                    pass
    chunks.append('</html>')
    atomic(run/'report.html', ''.join(chunks).encode())


def selection_key(metrics, objective='recall_1'):
    if objective == 'gallery_recall_1':
        metrics = metrics['gallery']
        return metrics['recall_1'], metrics['mrr'], metrics['recall_3'], metrics['recall_5']
    if objective not in ('recall_1', 'recall_10', 'mrr'):
        raise ValueError('Criterio de selecao invalido')
    return (metrics[objective], metrics['mrr' if objective != 'mrr' else 'recall_1'])


def training_batches(model, data, experiment, usable, batch, seed, epoch, hard=False, hard_k=8):
    order = np.random.default_rng(seed+epoch).permutation(usable).tolist()
    if not hard:
        batches = [order[i:i+batch] for i in range(0, len(order), batch)]
    else:
        model.eval()
        scores = score_matrix(model, data, experiment, usable)
        similarities = np.maximum(scores, scores.T)
        np.fill_diagonal(similarities, -np.inf)
        neighbours = {anchor: [usable[j] for j in np.argsort(-similarities[i], kind='stable')
                               if np.isfinite(similarities[i,j])][:hard_k] for i,anchor in enumerate(usable)}
        batches, remaining = [], set(order)
        for anchor in order:
            if anchor not in remaining:
                continue
            selected = [anchor]
            selected += [j for j in neighbours[anchor] if j in remaining and j != anchor][:batch-1]
            selected += [j for j in order if j in remaining and j not in selected][:batch-len(selected)]
            remaining.difference_update(selected)
            batches.append(selected)
    if len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    model.train()
    return batches


def train(manifest, experiment='E1', cohort_experiment=None, mode='matched', ids=None, epochs=100,
          lr=.001, weight_decay=.0001, batch=32, patience=15, seed=42, loss='infonce',
          temperature=.07, margin=.2, size=224, device='auto', run_dir=None, resume=False,
          smoke=False, stop=None, log=print, backbone='dinov2_vits14', selection_metric='gallery_recall_1',
          preprocessing='original', negative_sampling='random', hard_k=8, hard_warmup=5):
    selection_key(dict(mrr=0., recall_1=0., recall_10=0., gallery=dict(mrr=0., recall_1=0., recall_3=0., recall_5=0.)), selection_metric)
    if experiment not in EXPERIMENTS or loss not in ('infonce', 'triplet'):
        raise ValueError('Experimento ou loss invalido')
    if any('CE' in EXPERIMENTS[e][1] for e in (experiment, cohort_experiment or experiment)):
        raise ValueError('Protocolo CE legado desativado. Inicie novo treino L2 (FL x CD) ou E6D (sem CE).')
    if batch < 2 or epochs < 1 or patience < 1 or lr <= 0 or temperature <= 0:
        raise ValueError('Parametros invalidos')
    if preprocessing not in ('original', 'uniform_background') or negative_sampling not in ('random', 'hard') or hard_k < 1 or hard_warmup < 0:
        raise ValueError('Enquadramento ou mineracao invalidos')
    c = cohort(manifest, cohort_experiment or experiment, mode, ids)
    people = [p for p in c['people'] if p['split'] in ('treino', 'validacao')]
    train_ids = [i for i, p in enumerate(people) if p['split'] == 'treino']
    val_ids = [i for i, p in enumerate(people) if p['split'] == 'validacao']
    if len(train_ids) < 2 or len(val_ids) < 2:
        raise ValueError('Necessarias pelo menos duas identidades de treino e duas de validacao.')
    run = Path(run_dir) if run_dir else new_run('runs')
    run.mkdir(parents=True, exist_ok=True)
    extractor = configured_extractor(dict(size=size, backbone=backbone, preprocessing=preprocessing), device)
    config = dict(version=VERSION, experiment=experiment, cohort_experiment=cohort_experiment or experiment,
                  mode=mode, ids=ids, epochs=epochs, lr=lr, weight_decay=weight_decay, batch=batch,
                  patience=patience, seed=seed, loss=loss, temperature=temperature, margin=margin,
                  size=size, manifest=str(Path(manifest).resolve()), cohort=c['fingerprint'],
                  extractor=extractor.fingerprint, smoke=smoke, backbone=backbone,
                  dimension=getattr(extractor, 'dimension', 384), selection_metric=selection_metric,
                  preprocessing=preprocessing, negative_sampling=negative_sampling, hard_k=hard_k, hard_warmup=hard_warmup,
                  code_hash=digest({name: sha(ROOT/name) for name in ('engine.py', 'data.py', 'backbone.py', 'core.py', 'preprocessing.py')}))
    invariant = {k: v for k, v in config.items() if k != 'epochs'}
    if (run/'config.json').exists():
        if not resume:
            raise ValueError('Execucao ja existe; use retomar ou uma nova pasta.')
        old = load(run/'config.json')
        if {k: v for k, v in old.items() if k != 'epochs'} != invariant or epochs < old['epochs']:
            raise ValueError('Retomada incompativel com dados, pesos ou parametros.')
    save(run/'config.json', config)
    save(run/'cohort.json', c)
    save(run/'status.json', dict(state='RODANDO', phase='extract', device=extractor.device))
    try:
        data, failures = features(people, experiment, extractor, stop, log)
        save(run/'failures.json', failures)
        usable = [i for i in train_ids if data['face'][1][i].sum() and data['skull'][1][i].sum()]
        if len(usable) < 2:
            raise ValueError('Menos de dois pares de treino legiveis.')
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = None if experiment == 'E0' else Towers(config['dimension'])
        history, start, best, bad = [], 0, (-1., -1.), 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) if model else None
        if resume and (run/'last.pt').exists() and model:
            state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            start, best, bad, history = state['epoch'], state['best'], state['bad'], state['history']
        if model:
            for epoch in range(start, epochs):
                check_stop(stop)
                hard = negative_sampling == 'hard' and epoch >= hard_warmup
                batches = training_batches(model, data, experiment, usable, batch, seed, epoch, hard, hard_k)
                if hard:
                    save(run/'mining_last.json', dict(epoch=epoch+1, split='treino',
                        batches=[[people[i]['person'] for i in selected] for selected in batches]))
                    log('Candidatos dificeis: mineracao somente nas identidades de treino', flush=True)
                losses = []
                for selected in batches:
                    check_stop(stop)
                    optimizer.zero_grad()
                    (a, _), (b, _) = encode(model, data, experiment, selected)
                    if loss == 'infonce':
                        logits = a@b.T/temperature
                        labels = torch.arange(len(a))
                        value = (F.cross_entropy(logits, labels)+F.cross_entropy(logits.T, labels))/2
                    else:
                        value = triplet(a, b, margin)
                    if not torch.isfinite(value):
                        raise ValueError('Loss nao finita')
                    value.backward(); optimizer.step()
                    losses.append(float(value.detach()))
                model.eval()
                result = evaluation(score_matrix(model, data, experiment, val_ids), [people[i] for i in val_ids])
                mrr = result['full']['mrr']
                history.append(dict(epoch=epoch+1, loss=float(np.mean(losses)), val_mrr=mrr,
                                    val_recall_1=result['full']['recall_1'],val_recall_3=result['full']['recall_3'],
                                    val_recall_5=result['full']['recall_5'],val_recall_10=result['full']['recall_10']))
                history[-1].update({f'gallery_{k}': v for k, v in result['gallery'].items() if k in ('recall_1','recall_3','recall_5','mrr')})
                current = selection_key(result if selection_metric == 'gallery_recall_1' else result['full'], selection_metric)
                if current > best:
                    best, bad = current, 0
                    checkpoint(run/'best.pt', dict(model=model.state_dict(), epoch=epoch+1, config=config))
                else:
                    bad += 1
                checkpoint(run/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                           epoch=epoch+1, best=best, bad=bad, history=history))
                save(run/'history.json', history)
                save(run/'status.json', dict(state='RODANDO', phase='train', epoch=epoch+1, epochs=epochs))
                log(f'Epoca {epoch+1}/{epochs}: loss {history[-1]["loss"]:.5f}; Gallery-{result["gallery_size"]}: Top-1 {result["gallery"]["recall_1"]:.1%}; Top-3 {result["gallery"]["recall_3"]:.1%}; Top-5 {result["gallery"]["recall_5"]:.1%}; selecao {selection_metric}', flush=True)
                if bad >= patience:
                    break
            state = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(state['model']); model.eval()
        else:
            checkpoint(run/'best.pt', dict(model=None, epoch=0, config=config))
        check_stop(stop)
        scores = score_matrix(model, data, experiment, val_ids)
        result = evaluation(scores, [people[i] for i in val_ids])
        result.update(split='validacao', cohort=c['fingerprint'], experiment=experiment,
                      failed_inputs=failures, smoke=smoke, checkpoint_hash=sha(run/'best.pt'))
        save(run/'evaluation.json', result)
        np_save(run/'scores.npz', scores=scores, ids=np.array([people[i]['person'] for i in val_ids]))
        export_report(run, result, [people[i] for i in val_ids])
        save(run/'status.json', dict(state='CONCLUIDO', phase='evaluation', device=extractor.device))
    except InterruptedError:
        save(run/'status.json', dict(state='INTERROMPIDO'))
    except Exception as exc:
        save(run/'status.json', dict(state='ERRO', error=str(exc)))
        raise
    return run


def load_run(run):
    run = Path(run)
    config = load(run/'config.json')
    state = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
    if {k: v for k, v in state['config'].items() if k != 'epochs'} != {k: v for k, v in config.items() if k != 'epochs'}:
        raise ValueError('Configuracao diverge do checkpoint')
    model = None if config['experiment'] == 'E0' else Towers(config.get('dimension', 384))
    if model:
        model.load_state_dict(state['model']); model.eval()
    return config, model


def freeze(run):
    run = Path(run)
    if load(run/'status.json')['state'] != 'CONCLUIDO' or load(run/'config.json')['smoke']:
        raise ValueError('Apenas execucoes completas, fora do smoke test, podem ser congeladas.')
    save(run/'frozen.json', dict(checkpoint_hash=sha(run/'best.pt'), config_hash=sha(run/'config.json')))
    return run


def evaluate(run, split='validacao', stop=None, log=print):
    run = Path(run)
    config, model = load_run(run)
    if split not in ('validacao', 'teste'):
        raise ValueError('Split invalido')
    if split == 'teste':
        frozen = load(run/'frozen.json') if (run/'frozen.json').exists() else {}
        if frozen != dict(checkpoint_hash=sha(run/'best.pt'), config_hash=sha(run/'config.json')):
            raise ValueError('Congele a configuracao antes de avaliar o teste.')
    c = load(run/'cohort.json')
    people = [p for p in c['people'] if p['split'] == split]
    if not people:
        raise ValueError('Split vazio')
    extractor = configured_extractor(config)
    if extractor.fingerprint != config['extractor']:
        raise ValueError('Extrator diferente do treino')
    data, failures = features(people, config['experiment'], extractor, stop, log)
    result = evaluation(score_matrix(model, data, config['experiment']), people)
    result.update(split=split, cohort=c['fingerprint'], failed_inputs=failures,
                  checkpoint_hash=sha(run/'best.pt'), smoke=config['smoke'])
    target = new_run('evaluations')
    save(target/'config.json', dict(run=str(run), split=split))
    save(target/'evaluation.json', result)
    export_report(target, result, people)
    return target


def paired_comparison(run_a, run_b):
    a, b = load(Path(run_a)/'evaluation.json'), load(Path(run_b)/'evaluation.json')
    ca, cb = load(Path(run_a)/'config.json'), load(Path(run_b)/'config.json')
    if a['cohort'] != b['cohort'] or a['galleries'] != b['galleries'] or a['split'] != b['split']:
        raise ValueError('Execucoes usam coortes ou galerias diferentes. Reexecute o frontal na mesma coorte.')
    if ca['seed'] != cb['seed']:
        raise ValueError('Compare a mesma seed primeiro.')
    rows = []
    for x, y in zip(a['rows'], b['rows']):
        if x['person'] != y['person']:
            raise ValueError('Identidades desalinhadas')
        rows.append(dict(person=x['person'], rank_a=x['rank'], rank_b=y['rank'],
                         delta_rr=(1/y['rank'] if y['rank'] else 0)-(1/x['rank'] if x['rank'] else 0)))
    delta = np.array([r['delta_rr'] for r in rows])
    rng = np.random.default_rng(42)
    ci = np.quantile([delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(2000)], [.025, .975]).tolist()
    return dict(delta_mrr=float(delta.mean()), ci95=ci, rows=rows,
                conclusion='ganho inconclusivo' if ci[0] <= 0 <= ci[1] else 'melhora' if ci[0] > 0 else 'piora')


def suite(manifest, experiment='E1', seeds=None, mode='matched', resume_dir=None, stop=None, log=print, **kwargs):
    seeds = seeds or [42]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Seeds repetidas na suite')
    request = dict(manifest=manifest, experiment=experiment, seeds=seeds, mode=mode, **kwargs)
    run = Path(resume_dir) if resume_dir else new_run('suites')
    if resume_dir and load(run/'request.json') != request:
        raise ValueError('Parametros da suite alterados')
    save(run/'request.json', request)
    save(run/'status.json', dict(state='RODANDO'))
    child_runs = []
    try:
        for seed in seeds:
            for exp in (['E0'] if experiment == 'E0' else dict.fromkeys(['E0', 'E1', experiment])):
                check_stop(stop)
                target = run/f'{exp}_seed{seed}'
                child_runs.append(str(target))
                save(run/'children.json', child_runs)
                if (target/'status.json').exists() and load(target/'status.json')['state'] == 'CONCLUIDO':
                    continue
                train(manifest, experiment=exp, cohort_experiment=experiment, mode=mode, seed=seed,
                      run_dir=target, resume=(target/'config.json').exists(), stop=stop, log=log, **kwargs)
                if load(target/'status.json')['state'] != 'CONCLUIDO':
                    raise InterruptedError()
        comparisons = []
        if experiment not in ('E0', 'E1'):
            for seed in seeds:
                comparisons.append(dict(seed=seed, **paired_comparison(run/f'E1_seed{seed}', run/f'{experiment}_seed{seed}')))
        save(run/'comparisons.json', comparisons)
        if comparisons:
            deltas = np.array([[r['delta_rr'] for r in c['rows']] for c in comparisons])
            # Seeds repeat the same people: bootstrap identities after averaging seeds.
            mean_person = deltas.mean(0)
            rng = np.random.default_rng(42)
            ci = np.quantile([mean_person[rng.integers(len(mean_person), size=len(mean_person))].mean()
                              for _ in range(2000)], [.025, .975]).tolist()
            save(run/'summary.json', dict(seeds=seeds, delta_mrr=float(mean_person.mean()),
                seed_std=float(deltas.mean(1).std()), ci95=ci, n_identities=len(mean_person),
                conclusion='ganho inconclusivo' if ci[0] <= 0 <= ci[1] else 'melhora' if ci[0] > 0 else 'piora',
                confirmed_three_seeds=set(seeds) >= {42, 43, 44}))
        save(run/'status.json', dict(state='CONCLUIDO'))
    except InterruptedError:
        save(run/'status.json', dict(state='INTERROMPIDO'))
    except Exception as exc:
        save(run/'status.json', dict(state='ERRO', error=str(exc)))
        raise
    return run


def sample_gallery(people, count=10, seed=42):
    people = sorted(people, key=lambda p: p['person'])
    if len({p['person'] for p in people}) != len(people) or not 2 <= count <= len(people):
        raise ValueError('Quantidade invalida ou identidades duplicadas')
    rng = random.Random(seed)
    query = rng.choice(people)
    candidates = rng.sample([p for p in people if p['person'] != query['person']], count-1)+[query]
    rng.shuffle(candidates)
    return dict(query=query, candidates=candidates)


def compare(run, query, candidates, stop=None, log=print):
    from .core import uses_ce
    if uses_ce(run):
        raise ValueError('Modelo legado com CE: treine novamente sem CE antes de comparar.')
    config, model = load_run(run)
    if len({p['person'] for p in candidates}) != len(candidates) or not candidates:
        raise ValueError('Candidatos precisam ter identificadores distintos')
    extractor = configured_extractor(config)
    if extractor.fingerprint != config['extractor']:
        raise ValueError('Extrator divergente')
    data, failures = features([query]+candidates, config['experiment'], extractor, stop, log)
    scores = score_matrix(model, data, config['experiment'])[0, 1:]
    relevant = [f for f in failures if (f['person'] == query['person'] and f['view'] in EXPERIMENTS[config['experiment']][0])
                or (f['person'] in {c['person'] for c in candidates} and f['view'] in EXPERIMENTS[config['experiment']][1])]
    rows = [dict(person=c['person'], score=float(s) if np.isfinite(s) else None,
                 available_views=[v for j, v in enumerate(EXPERIMENTS[config['experiment']][1])
                                  if data['skull'][1][i+1, j]])
            for i, (c, s) in enumerate(zip(candidates, scores))]
    rows.sort(key=lambda r: (-r['score'] if r['score'] is not None else float('inf'), r['person']))
    for i, row in enumerate(rows):
        row['rank'] = i+1 if row['score'] is not None else None
    return dict(ranking=rows, failures=relevant, query=query, candidates=candidates,
                model=str(run), checkpoint_hash=sha(Path(run)/'best.pt'))


def explain(run, query, candidate, grid=4, stop=None, log=print):
    config, model = load_run(run)
    extractor = configured_extractor(config)
    if extractor.fingerprint != config['extractor']:
        raise ValueError('Extrator divergente')
    data, failures = features([query, candidate], config['experiment'], extractor, stop, log)
    base = float(score_matrix(model, data, config['experiment'])[0, 1])
    if not np.isfinite(base):
        raise ValueError('Par sem score valido')
    result = dict(score=base, removed_views=[], maps=[], failures=failures)
    for modality, views, person, index in [('face', EXPERIMENTS[config['experiment']][0], query, 0),
                                          ('skull', EXPERIMENTS[config['experiment']][1], candidate, 1)]:
        for j, view in enumerate(views):
            check_stop(stop)
            x, mask = data[modality]
            if not mask[index, j]:
                continue
            mask[index, j] = 0
            removed = float(score_matrix(model, data, config['experiment'])[0, 1])
            result['removed_views'].append(dict(modality=modality, view=view,
                   delta=base-removed if np.isfinite(removed) else None,
                   reason='' if np.isfinite(removed) else 'Ultima vista da modalidade'))
            mask[index, j] = 1
            im = read_image(person['views'][view]['path'])
            original = x[index, j].clone()
            heat = np.zeros((grid, grid))
            for y in range(grid):
                for col in range(grid):
                    check_stop(stop)
                    changed = im.copy()
                    changed.paste((124, 116, 104), (col*im.width//grid, y*im.height//grid,
                                  (col+1)*im.width//grid, (y+1)*im.height//grid))
                    feature, _ = extractor.encode_image(changed)
                    x[index, j] = torch.tensor(feature)
                    heat[y, col] = base-float(score_matrix(model, data, config['experiment'])[0, 1])
            x[index, j] = original
            result['maps'].append(dict(modality=modality, view=view, path=person['views'][view]['path'], values=heat.tolist()))
            log('Mapa de oclusao: '+modality+' '+view, flush=True)
    return result
