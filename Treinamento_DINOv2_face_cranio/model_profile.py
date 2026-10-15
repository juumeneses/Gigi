"""Metrics for exactly the selected views/weights, plus leaf training provenance."""
from .portable import Path

import numpy as np

from . import engine, fusion
from .core import load


def validation_scores(run, weights=None):
    run=Path(run)
    config=load(run/'config.json')
    people=[p for p in load(run/'cohort.json')['people'] if p['split']=='validacao']
    if config.get('rank_fusion_members'):
        if weights is not None and not np.allclose(weights,config['weights']):
            raise ValueError('A mistura usa as vistas fixas de suas fontes; avalie outra mistura para mudar essas vistas.')
        with np.load(run/'scores.npz',allow_pickle=False) as arrays:
            matrix=arrays['combinado'].copy()
    elif (run/'results.json').exists():
        with np.load(run/'scores.npz',allow_pickle=False) as arrays:
            matrix=fusion.combine([arrays[n].copy() for n in fusion.COMPONENTS],config['normalization'],config['weights'] if weights is None else weights)
    else:
        with np.load(run/'scores.npz',allow_pickle=False) as arrays:
            matrix=arrays['scores'].copy()
            if list(arrays['ids'])!=[p['person'] for p in people]:
                raise ValueError('Ordem das identidades divergente')
    if matrix.shape!=(len(people),len(people)):
        raise ValueError('Scores e galeria de validacao divergentes')
    return matrix,people


def evaluation(run,weights=None):
    scores,people=validation_scores(run,weights)
    if load(Path(run)/'config.json').get('rank_fusion_members'):
        return load(Path(run)/'results.json')['combinado']
    return engine.evaluation(scores,people)


def leaves(run,weights=None,source='Modelo'):
    run=Path(run)
    config=load(run/'config.json')
    if config.get('rank_fusion_members'):
        return [leaf for i,member in enumerate(config['rank_fusion_members']) if member['weight']>0
                for leaf in leaves(member['run'],member['view_weights'],f'Fonte {i+1}')]
    if config.get('ensemble_members'):
        return [leaf for member in config['ensemble_members'] for leaf in leaves(member['run'],weights,source)]
    if config.get('children'):
        selected=config['weights'] if weights is None else weights
        return [leaf for name,w in zip(fusion.COMPONENTS,selected) if w>0 for leaf in leaves(config['children'][name]['run'],source=source+' / '+name)]
    history=load(run/'history.json') if (run/'history.json').exists() else []
    criterion=config.get('selection_metric','mrr')
    best=max(history,key=lambda r:(r.get('val_'+criterion,-1),r['val_mrr']) if criterion.startswith('recall_') else (r['val_mrr'],r.get('val_recall_1',-1))) if history else {}
    return [dict(Fonte=source,Execucao=str(run),Extrator=config.get('backbone','dinov2_vits14'),
        Loss=config.get('loss'),Seed=config.get('seed'),Epocas=len(history),Melhor_epoca=best.get('epoch'),
        Criterio=criterion,Learning_rate=config.get('lr'),Batch=config.get('batch'),
        Resolucao=config.get('size'),Paciencia=config.get('patience'),Weight_decay=config.get('weight_decay'),
        Temperatura=config.get('temperature'),Margem_triplet=config.get('margin'),
        Enquadramento=config.get('preprocessing','original'),Mineracao=config.get('negative_sampling','random'))]
