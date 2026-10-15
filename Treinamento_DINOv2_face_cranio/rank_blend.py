"""Weighted reciprocal rank fusion of two fixed model/view configurations."""
from .portable import Path
import numpy as np

from . import engine,fusion
from .core import check_stop,csv_write,load,new_run,np_save,save,sha
from .model_profile import validation_scores
from .weight_search import rank_metrics


def ordered_ranks(scores,ids):
    scores=np.asarray(scores,float)
    if scores.ndim != 2 or scores.shape[1]!=len(ids) or len(set(ids))!=len(ids):
        raise ValueError('Candidatos invalidos')
    finite = np.isfinite(scores)
    scores = np.where(finite,scores,-np.inf)
    sorted_ids=np.argsort(np.asarray(ids),kind='stable')
    order=sorted_ids[np.argsort(-scores[:,sorted_ids],axis=1,kind='stable')]
    ranks=np.empty_like(order)
    np.put_along_axis(ranks,order,np.broadcast_to(np.arange(1,len(ids)+1),order.shape),axis=1)
    return np.where(finite,ranks,np.inf)


def combine(matrices,ids,weights,k=1):
    if k<=0 or not np.isfinite(k) or len(matrices)!=len(weights) or not np.isfinite(weights).all() or min(weights)<0 or sum(weights)<=0:
        raise ValueError('Pesos ou constante RRF invalidos')
    shape=matrices[0].shape
    output=np.zeros(shape,float);valid=np.ones(shape,bool);parts=[]
    for matrix,weight in zip(matrices,weights):
        if matrix.shape!=shape:raise ValueError('Matrizes desalinhadas')
        ranks=ordered_ranks(matrix,ids)
        part=(weight/sum(weights))/(k+ranks)
        if weight>0:valid &= np.isfinite(matrix)
        output += part
        parts.append((ranks,part))
    output[~valid]=-np.inf
    return output,parts


def evaluate(matrices, people, weights, k):
    ids = [p['person'] for p in people]
    scores, _ = combine(matrices, ids, weights, k)
    result = engine.evaluation(scores, people)
    # RRF ranks depend on the candidate gallery: recompute within Gallery-10.
    for i, row in enumerate(result['rows']):
        small_ids = result['galleries'][row['person']]
        indices = [ids.index(pid) for pid in small_ids]
        local, _ = combine([m[i:i+1, indices] for m in matrices], small_ids, weights, k)
        ranks = ordered_ranks(local, small_ids)
        rank = ranks[0, small_ids.index(row['person'])]
        row['gallery_rank'] = int(rank) if np.isfinite(rank) else None
    result['gallery'] = engine.metrics(result['rows'], 'gallery_rank')
    return result


def search(source_a,source_b,view_weights_a=(.5,0,.2),view_weights_b=(.5,.3,.2),stop=None,log=print):
    sources=[Path(source_a),Path(source_b)]
    configs=[load(p/'config.json') for p in sources]
    if any(c.get('rank_fusion_members') or c.get('smoke') or c.get('split')!='validacao' for c in configs):
        raise ValueError('Use duas fontes individuais ou ensembles completos de validacao')
    from .core import uses_ce
    if any(c.get('children') and uses_ce(p) for p,c in zip(sources,configs)):
        raise ValueError('Modelos legados com CE ficam fora do protocolo de matching atual')
    from .improvement_pipeline import file_signatures,check_signatures
    signatures=[file_signatures(p) for p in sources]
    matrices=[];people=None
    for p,w in zip(sources,(view_weights_a,view_weights_b)):
        matrix,group=validation_scores(p,w)
        if people is not None and [x['person'] for x in people]!=[x['person'] for x in group]:
            raise ValueError('As fontes usam identidades ou galerias diferentes')
        people=group;matrices.append(matrix)
    if configs[0]['cohort']!=configs[1]['cohort']:raise ValueError('Coortes diferentes')
    ids=[p['person'] for p in people]
    output=new_run('blend_searches');rows=[]
    save(output/'config.json',dict(sources=list(map(str,sources)),view_weights=[list(view_weights_a),list(view_weights_b)],
         objective='recall_1_then_recall_10_then_mrr',split='validacao',input_hashes=signatures,code_hash=sha(__file__)))
    for k in (1,5,10,30,60):
        for i in range(21):
            check_stop(stop)
            alpha=i/20
            scores,_=combine(matrices,ids,[alpha,1-alpha],k)
            rows.append(dict(weight_a=alpha,weight_b=1-alpha,k=k,**rank_metrics(scores,ids)))
        log(f'Mistura: constante {k}; {len(rows)}/105 combinacoes',flush=True)
    best=max(rows,key=lambda r:(r['recall_1'],r['recall_10'],r['mrr']))
    for signature in signatures:check_signatures(signature)
    scores,_=combine(matrices,ids,[best['weight_a'],best['weight_b']],best['k'])
    target=new_run('fusions')
    members=[dict(run=str(p.resolve()),config_hash=sha(p/'config.json'),view_weights=list(w),weight=weight)
        for p,w,weight in zip(sources,(view_weights_a,view_weights_b),(best['weight_a'],best['weight_b']))]
    active=[float(any(m['weight']>0 and m['view_weights'][j]>0 for m in members)) for j in range(3)]
    cfg=dict(configs[1],rank_fusion_members=members,rank_constant=best['k'],weights=active,
             blend_search=str(output.resolve()),policy='weighted_rrf_two_models',weight_objective='recall_1_then_recall_10')
    for key in ('ensemble_members','weight_search'):cfg.pop(key,None)
    save(target/'config.json',cfg);save(target/'cohort.json',load(sources[0]/'cohort.json'))
    results={name:engine.evaluation(matrix,people) for name,matrix in zip(('source_a','source_b'),matrices)}
    results['combinado']=evaluate(matrices,people,[best['weight_a'],best['weight_b']],best['k'])
    save(target/'results.json',results)
    np_save(target/'scores.npz',source_a=matrices[0],source_b=matrices[1],combinado=scores)
    engine.export_report(target,results['combinado'],people)
    save(target/'status.json',dict(state='CONCLUIDO'))
    save(output/'grid.json',rows);csv_write(output/'grid.csv',rows)
    save(output/'summary.json',dict(best=best,output=str(target.resolve()),
         metrics={name:r['full'] for name,r in results.items()},
         note='Mistura escolhida nesta validacao; resultados exploratorios. Recall@1 e prioridade, Recall@10 desempata antes de MRR.'))
    save(output/'status.json',dict(state='CONCLUIDO'))
    return output


def compare(run,query,candidates,stop=None,log=print):
    run=Path(run).resolve()
    cfg=load(run/'config.json');ids=[c['person'] for c in candidates]
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError('Candidatos precisam ter identificadores distintos')
    matrices=[];lookups=[];failures=[]
    for member in cfg['rank_fusion_members']:
        check_stop(stop)
        if sha(Path(member['run'])/'config.json')!=member['config_hash']:raise ValueError('Fonte da mistura alterada')
        if member['weight']==0:
            matrices.append(np.full((1,len(ids)),-np.inf));lookups.append({});continue
        out=fusion.compare_fusion(member['run'],query,candidates,stop,log,weights=member['view_weights'])
        result=load(out/'result.json');lookup={r['person']:r for r in result['ranking']}
        matrices.append(np.array([[lookup[pid]['score'] if lookup[pid]['score'] is not None else -np.inf for pid in ids]]))
        lookups.append(lookup);failures.extend(result['failures'])
    scores,parts=combine(matrices,ids,[m['weight'] for m in cfg['rank_fusion_members']],cfg['rank_constant'])
    rows=[]
    for j,candidate in enumerate(candidates):
        row=dict(person=candidate['person'],score=float(scores[0,j]) if np.isfinite(scores[0,j]) else None)
        for i,(ranks,contribution) in enumerate(parts):
            row[f'source_{i+1}_rank']=int(ranks[0,j]) if np.isfinite(ranks[0,j]) else None
            row[f'source_{i+1}_contribuicao']=float(contribution[0,j])
        views=sorted({v for lookup,member in zip(lookups,cfg['rank_fusion_members']) if member['weight']>0
                      for name in fusion.COMPONENTS for v in lookup.get(candidate['person'],{}).get(name+'_vistas',[])})
        row['available_views']=views
        for name in fusion.COMPONENTS:
            row[name]=None;row[name+'_contribuicao']=None;row[name+'_vistas']=views
        rows.append(row)
    rows.sort(key=lambda r:(-r['score'] if r['score'] is not None else float('inf'),r['person']))
    for i,row in enumerate(rows,1):row['rank']=i if row['score'] is not None else None
    out=new_run('fusion_comparisons')
    save(out/'result.json',dict(model=str(run),query=query,candidates=candidates,ranking=rows,weights=cfg['weights'],
        failures=failures,all_subsets=False,combinations={},score_space='weighted_rrf'))
    return out
