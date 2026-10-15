"""Join stored labels by explicit record IDs; never infer from images."""
import pandas as pd
from pathlib import Path

from .core import REPO, ARTIFACTS, load, atomic, save, sha, new_run


def snapshot_curated():
    """Archive labels as supplied, without converting provenance to documentation."""
    source = REPO/'Treinamento_faces/resultados/genero_faces/genero_faces_curado.csv'
    token = sha(source)
    for manifest in (ARTIFACTS/'curated_references').glob('*/source.json'):
        if load(manifest)['sha256'] == token:
            if sha(manifest.parent/'genero_faces_curado.csv') != token:
                raise ValueError('Copia da curadoria alterada')
            return manifest.parent
    frame = pd.read_csv(source)
    run = new_run('curated_references')
    atomic(run/'genero_faces_curado.csv', source.read_bytes())
    save(run/'source.json', dict(path=str(source), sha256=token, rows=len(frame),
                                origins=frame.origem_genero_final.value_counts().to_dict(),
                                usage='COMPARACAO_COM_CURADORIA', documented=False))
    return run


def snapshot_facial_results():
    """Copy stored outputs and metrics only; keep exact source provenance."""
    base = REPO/'Treinamento_faces/resultados/genero_faces'
    names = ['genero_faces_curado.csv', 'revisao_manual_genero_faces.csv',
             'predicoes_genero_fairface.csv', 'predicoes_genero_metricas_faces.csv',
             'metricas_modelo_genero_faces.csv', 'matriz_confusao_genero_faces.csv',
             'comparacao_genero_fairface_modelo.csv']
    sources = {name: {'path': str(base/name), 'sha256': sha(base/name)}
               for name in names if (base/name).exists()}
    for manifest in (ARTIFACTS/'facial_recorded_results').glob('*/sources.json'):
        if load(manifest)['files'] == sources:
            for name, meta in sources.items():
                if sha(manifest.parent/name) != meta['sha256']:
                    raise ValueError('Copia facial alterada: '+name)
            return manifest.parent
    run = new_run('facial_recorded_results')
    for name, meta in sources.items():
        atomic(run/name, Path(meta['path']).read_bytes())
    save(run/'sources.json', dict(files=sources, usage='RESULTADOS_HISTORICOS',
                                 reference='curadoria com procedencia preservada'))
    return run


def table():
    base = REPO / 'Treinamento_faces/resultados/genero_faces'
    paths = [base/'genero_faces_curado.csv', base/'predicoes_genero_metricas_faces.csv',
             REPO/'Treinamento_cranios/resultados/sexo_cranio/predicoes_sexo_cranio.csv']
    snapshots = sorted((ARTIFACTS/'curated_references').glob('*/source.json'), reverse=True)
    if snapshots:
        copied = snapshots[0].parent/'genero_faces_curado.csv'
        if sha(copied) != load(snapshots[0])['sha256']:
            raise ValueError('Copia da curadoria alterada')
        paths[0] = copied
    facial_copies = sorted((ARTIFACTS/'facial_recorded_results').glob('*/sources.json'), reverse=True)
    if facial_copies:
        manifest = load(facial_copies[0])
        name = 'predicoes_genero_metricas_faces.csv'
        copied = facial_copies[0].parent/name
        if sha(copied) != manifest['files'][name]['sha256']:
            raise ValueError('Copia das previsoes faciais alterada')
        paths[1] = copied
    frames = [pd.read_csv(p) if p.exists() else pd.DataFrame() for p in paths]
    curated, face, skull = frames
    if curated.empty:
        return pd.DataFrame()
    out = curated[['ano', 'id_imagem', 'genero_final', 'origem_genero_final', 'classe_fairface']].copy()
    for col in ['prob_masculino_fairface', 'prob_feminino_fairface', 'confianca_fairface']:
        if col in curated:
            out[col] = curated[col]
    out['id_pessoa'] = out.id_imagem.astype(str).str.removesuffix('_FFN')
    out = out.rename(columns={'genero_final': 'valor_curado', 'origem_genero_final': 'origem_valor_curado',
                              'classe_fairface': 'estimativa_fairface'})
    # None of these source columns is a documented external ground truth.
    out['referencia_documentada'] = 'NÃO DISPONÍVEL NESTAS FONTES'
    if not face.empty:
        face = face.rename(columns={'classe_modelo': 'estimativa_modelo_face',
                                    'confianca_modelo': 'confianca_face'})
        out = out.merge(face.drop(columns=['ano'], errors='ignore'), on='id_imagem', how='left', validate='one_to_one')
    if not skull.empty:
        skull = skull.rename(columns={'classe_cranio': 'estimativa_modelo_cranio', 'ano': 'ano_cranio'})
        out = out.merge(skull.drop(columns=['id_imagem_cranio'], errors='ignore'),
                        on='id_pessoa', how='outer', validate='one_to_one')
        out['ano'] = out['ano'].combine_first(out.pop('ano_cranio'))
    out['referencia_documentada'] = out['referencia_documentada'].fillna('NÃO DISPONÍVEL NESTAS FONTES')
    documented = sorted((ARTIFACTS/'sex_references').glob('*/reference.csv'), reverse=True)
    documented = [p for p in documented if load(p.parent/'config.json').get('reference_kind', 'DOCUMENTAL') == 'DOCUMENTAL']
    if documented:
        from .references_io import references
        labels = references(documented[0].parent).reset_index().rename(columns={'sexo': 'sexo_documentado', 'fonte': 'fonte_documentada'})
        out = out.merge(labels[['id_pessoa', 'sexo_documentado', 'fonte_documentada']], how='outer', on='id_pessoa', validate='one_to_one')
        valid = out.sexo_documentado.notna()
        out.loc[valid, 'referencia_documentada'] = out.loc[valid, 'sexo_documentado']
    fresh = sorted((ARTIFACTS/'cranial_sex_predictions').glob('*/predictions.json'), reverse=True)
    if fresh:
        predictions = pd.DataFrame(load(fresh[0]))
        if not predictions.empty:
            predictions = predictions.rename(columns={'person': 'id_pessoa', 'predicted': 'nova_estimativa_craniana', 'reference': 'referencia_da_execucao'})
            out = out.merge(predictions, how='outer', on='id_pessoa', validate='one_to_one')
    out['referencia_documentada'] = out['referencia_documentada'].fillna('NÃO DISPONÍVEL NESTAS FONTES')
    for modality in ['face', 'cranio']:
        column = 'estimativa_modelo_' + modality
        if column not in out:
            out[column] = None
        values = out[column]
        if modality == 'cranio' and 'nova_estimativa_craniana' in out:
            attempted = out['model'].notna() if 'model' in out else out.nova_estimativa_craniana.notna()
            values = values.where(~attempted, out.nova_estimativa_craniana)
            out['origem_previsao_cranio'] = attempted.map({True: 'NOVA EXECUÇÃO', False: 'MODELO LEGADO'})
        predicted = values.astype('string').str.upper().str.replace('_PROVAVEL', '', regex=False)
        out['previsao_' + modality] = predicted
        valid = predicted.isin(['MASCULINO', 'FEMININO']) & out.valor_curado.isin(['MASCULINO', 'FEMININO'])
        out['concordancia_curadoria_' + modality] = 'SEM COMPARAÇÃO'
        out.loc[valid, 'concordancia_curadoria_' + modality] = 'DIVERGE'
        out.loc[valid & predicted.eq(out.valor_curado), 'concordancia_curadoria_' + modality] = 'CONCORDA'
    primary = ['id_pessoa', 'valor_curado', 'previsao_face', 'previsao_cranio',
               'concordancia_curadoria_face', 'concordancia_curadoria_cranio',
               'estimativa_fairface', 'ano', 'origem_valor_curado', 'referencia_documentada']
    return out[primary + [c for c in out if c not in primary]]


def result_summary(frame, models):
    """Keep missing predictions/references out of rates, visible in counts."""
    rows = []
    reference = frame.valor_curado.astype('string').str.strip().str.upper()
    for label, column in models.items():
        values = frame[column].astype('string').str.strip().str.upper().str.replace('_PROVAVEL', '', regex=False)
        present = values.isin(['MASCULINO', 'FEMININO'])
        comparable = present & reference.isin(['MASCULINO', 'FEMININO'])
        n = int(comparable.sum())
        hits = int((comparable & values.eq(reference)).sum())
        rows.append({'Modelo': label, 'Pessoas': len(frame), 'Com previsão': int(present.sum()),
                     'Masculino previsto': int(values.eq('MASCULINO').sum()),
                     'Feminino previsto': int(values.eq('FEMININO').sum()),
                     'Comparáveis': n, 'Concordantes': hits, 'Divergentes': n-hits,
                     'Concordância (%)': 100*hits/n if n else None,
                     'Divergência (%)': 100*(n-hits)/n if n else None,
                     'Sem comparação': len(frame)-n})
    return pd.DataFrame(rows)
