"""Staged, resumable Top-10 retrieval: backbones, small grid, three-seed confirmation."""
from .portable import Path
import html
import numpy as np

from . import backbone, engine, ensemble, fusion
from .core import ROOT, atomic, check_stop, csv_write, digest, load, new_run, save, sha
from .data import cohort
from .improvement_pipeline import check_signatures, file_signatures
from .weight_search import rank_metrics


def key(metrics):
    return metrics['recall_10'],metrics['mrr'],metrics['recall_1']


def trial_id(model,lr,margin):
    return model+'_'+digest([float(lr),float(margin)])[:12]


def summarize(run,records,phase,baseline_model='dinov2_vitb14'):
    grouped={}
    for r in records:grouped.setdefault((r['trial'],r['kind']),[]).append(r)
    rows=[];subsets=[];examples=[]
    for (trial,kind),members in grouped.items():
        evaluations=[load(Path(m['output'])/'results.json')['combinado'] for m in members]
        first=members[0]
        row={k:first[k] for k in ('trial','backbone','lr','margin','kind')}
        row.update(n=evaluations[0]['full']['n'],seeds=sorted(m['seed'] for m in members if m['seed']>=0),
            n_seeds=len(members) if kind=='individual' else first['n_seeds'],
            confirmed=kind=='ensemble' or len(members)>=3,outputs=[m['output'] for m in members])
        for name in ('recall_1','recall_3','recall_5','recall_10','mrr'):
            values=[e['full'][name] for e in evaluations]
            row[name]=float(np.mean(values));row[name+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        row['failures']=sum(e['full']['failures'] for e in evaluations)
        rows.append(row)
        for m,e in zip(members,evaluations):
            model=Path(m['output']);cfg=load(model/'config.json')
            people=[p for p in load(model/'cohort.json')['people'] if p['split']=='validacao']
            with np.load(model/'scores.npz',allow_pickle=False) as arrays:
                matrices=[arrays[name].copy() for name in fusion.COMPONENTS]
            for bits in ('100','010','001','110','101','011','111'):
                weights=[int(b)*w for b,w in zip(bits,cfg['weights'])]
                subsets.append(dict(m,subset=bits,**rank_metrics(fusion.combine(matrices,cfg['normalization'],weights),[p['person'] for p in people])))
            hit=next((r for r in e['rows'] if r['rank'] is not None and r['rank']<=10),None)
            miss=next((r for r in e['rows'] if r['rank'] is None or r['rank']>10),None)
            examples.extend(dict(m,**r) for r in (hit,miss) if r)
    baseline=next((r for r in rows if r['backbone']==baseline_model and r['lr']==.001 and r['margin']==.2 and r['kind']=='individual' and r['confirmed']),None)
    baseline_ensemble=next((r for r in rows if r['backbone']==baseline_model and r['lr']==.001 and r['margin']==.2 and r['kind']=='ensemble'),None)
    for row in rows:
        base=baseline_ensemble if row['kind']=='ensemble' else baseline
        row['delta_recall_10']=row['recall_10']-base['recall_10'] if base else None
        row['ci95']=None;row['conclusion']='triagem de uma seed' if not row['confirmed'] else 'baseline pendente'
        if base and row['confirmed']:
            evaluations=[load(Path(p)/'results.json')['combinado'] for p in row['outputs']]
            baselines=[load(Path(p)/'results.json')['combinado'] for p in base['outputs']]
            reference=evaluations[0]
            if any(e['galleries']!=reference['galleries'] or [r['person'] for r in e['rows']]!=[r['person'] for r in reference['rows']] for e in evaluations+baselines):
                raise ValueError('Comparacao usa galerias ou identidades diferentes')
            hits=lambda es:np.mean([[float(r['rank'] is not None and r['rank']<=10) for r in e['rows']] for e in es],axis=0)
            delta=hits(evaluations)-hits(baselines);rng=np.random.default_rng(42)
            interval=np.round(np.quantile(delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(1),[.025,.975]),12).tolist()
            row.update(ci95=interval,conclusion='ganho inconclusivo' if interval[0]<=0<=interval[1] else 'melhora' if interval[0]>0 else 'piora')
    rows.sort(key=lambda r:(r['confirmed'],*key(r)),reverse=True)
    summary=dict(phase=phase,primary_metric='recall_10',rows=rows,subsets=subsets,examples=examples,
        note='Validacao exploratoria: triagem e escolha de parametros usam as mesmas pessoas. Seeds nao sao novas identidades. Teste final reservado; nenhuma acuracia prometida.')
    save(run/'summary.json',summary);csv_write(run/'results.csv',rows);csv_write(run/'subsets.csv',subsets)
    fields=('backbone','lr','margin','kind','n_seeds','confirmed','recall_1','recall_3','recall_5','recall_10','mrr','delta_recall_10','conclusion')
    document='<meta charset="utf-8"><h1>Pipeline Top-10</h1><p>'+html.escape(summary['note'])+'</p><table border="1"><tr>'+''.join('<th>'+f+'</th>' for f in fields)+'</tr>'
    for row in rows:document+='<tr>'+''.join('<td>'+html.escape(str(row.get(f,'')))+'</td>' for f in fields)+'</tr>'
    atomic(run/'report.html',(document+'</table>').encode())
    return rows


def run_pipeline(manifest,run_dir=None,backbones=('dinov2_vitb14','convnext_tiny'),
                 seeds=(42,43,44),learning_rates=(.001,.0003),margins=(.2,.1,.3),
                 tune=True,weights=(.5,.3,.2),epochs=100,patience=15,batch=32,size=224,
                 device='auto',smoke=False,stop=None,log=print):
    models=list(dict.fromkeys(backbones));seeds=list(seeds)
    rates=list(dict.fromkeys([.001,*map(float,learning_rates)]));margins=list(dict.fromkeys([.2,*map(float,margins)]))
    if not models or any(m not in backbone.MODELS for m in models):raise ValueError('Extratores invalidos')
    if len(seeds)!=3 or len(set(seeds))!=3:raise ValueError('Use tres seeds distintas')
    if any(not np.isfinite(x) or x<=0 for x in rates+margins):raise ValueError('Learning rates e margens precisam ser positivos e finitos')
    if len(rates)*len(margins)>12:raise ValueError('Limite a busca a doze configuracoes')
    if epochs<1 or patience<1 or batch<2 or size<56 or size%14:raise ValueError('Parametros de treino invalidos')
    if len(weights)!=3 or any(w<=0 for w in weights):raise ValueError('Use tres pesos positivos para comparar as sete combinacoes')
    fusion.combine([np.zeros((1,1))]*3,[dict(mean=0,std=1)]*3,weights)
    c=cohort(manifest,'E6D','matched')
    inputs={r['path']:sha(r['path']) if Path(r['path']).exists() else 'MISSING' for p in c['people'] if p['split'] in ('treino','validacao') for v,r in p['views'].items() if v in ('FFN','FFS','FL','CF','CD')}
    run=Path(run_dir) if run_dir else new_run('retrieval_pipelines');run.mkdir(parents=True,exist_ok=True)
    request=dict(manifest=str(Path(manifest).resolve()),manifest_hash=sha(manifest),backbones=models,seeds=seeds,
        learning_rates=rates,margins=margins,tune=tune,weights=list(weights),epochs=epochs,patience=patience,batch=batch,size=size,device=device,
        smoke=smoke,input_hashes=inputs,cohort=c['fingerprint'],selection_metric='recall_10',loss='triplet',
        code_hash=digest({name:sha(ROOT/name) for name in ('engine.py','backbone.py','data.py','fusion.py','core.py','preprocessing.py','ensemble.py','weight_search.py','retrieval_pipeline.py')}))
    if (run/'request.json').exists() and load(run/'request.json')!=request:raise ValueError('Retomada incompativel com dados, parametros ou codigo')
    save(run/'request.json',request);save(run/'cohort.json',c)
    ledger=load(run/'ledger.json') if (run/'ledger.json').exists() else {}
    records=[];phase='Comparacao de extratores';planned=[]
    def plan(keys):
        for k in keys:
            if k not in planned:planned.append(k)
    def status(state,stage='',**extra):
        save(run/'status.json',dict(state=state,phase=phase,stage=stage,done=sum(k in ledger for k in planned),total=len(planned),**extra))
    def stage(k,function,verify=True):
        check_stop(stop);plan([k]);status('RODANDO',k)
        number=min(sum(p in ledger for p in planned)+(0 if k in ledger else 1),len(planned))
        log(f'Pipeline {number}/{len(planned)}: {k}',flush=True)
        if k in ledger:
            check_signatures(ledger[k]['hashes']);output=Path(ledger[k]['output'])
            log('Reutilizando etapa concluida: '+k,flush=True)
        else:
            output=Path(function())
            if verify and load(output/'status.json')['state']!='CONCLUIDO':raise InterruptedError('Etapa interrompida')
            hashes=file_signatures(output) if verify else {str(p.resolve()):sha(p) for p in output.glob('*') if p.is_file()}
            ledger[k]=dict(output=str(output.resolve()),hashes=hashes);save(run/'ledger.json',ledger)
        return output
    def seed_keys(model,lr,margin,seed):
        tid=trial_id(model,lr,margin);return [f'{tid}/seed{seed}/train',f'{tid}/seed{seed}/scores']
    def trial(model,lr,margin,selected_seeds):
        tid=trial_id(model,lr,margin);outputs=[]
        for seed in selected_seeds:plan(seed_keys(model,lr,margin,seed))
        for seed in selected_seeds:
            bundle=run/'trials'/tid/f'seed{seed}';keys=seed_keys(model,lr,margin,seed)
            stage(keys[0],lambda:fusion.train_components(manifest,run_dir=bundle,loss='triplet',seed=seed,backbone=model,
                epochs=epochs,patience=patience,batch=batch,lr=lr,margin=margin,size=size,device=device,
                selection_metric='recall_10',preprocessing='original',negative_sampling='random',smoke=smoke,stop=stop,log=log))
            out=stage(keys[1],lambda:fusion.fuse(bundle,weights,stop,log));outputs.append(str(out.resolve()))
            if not any(r['trial']==tid and r['seed']==seed and r['kind']=='individual' for r in records):
                records.append(dict(trial=tid,backbone=model,lr=lr,margin=margin,kind='individual',seed=seed,n_seeds=1,output=str(out.resolve())))
                save(run/'records.json',records);summarize(run,records,phase,models[0])
        return outputs
    try:
        plan([f'prepare/{m}' for m in models]+[k for m in models for s in seeds for k in seed_keys(m,.001,.2,s)])
        for model in models:stage(f'prepare/{model}',lambda:backbone.prepare_variant(model,log),verify=False)
        benchmark=[]
        for model in models:
            outputs=trial(model,.001,.2,seeds)
            metrics=[load(Path(p)/'results.json')['combinado']['full'] for p in outputs]
            benchmark.append(dict(backbone=model,**{k:float(np.mean([m[k] for m in metrics])) for k in ('recall_1','recall_10','mrr')}))
        chosen=max(benchmark,key=key)['backbone'];save(run/'backbone_comparison.json',benchmark)
        save(run/'selection.json',dict(backbone=chosen,criterion='mean_seed_recall_10_then_mrr',screening_seed=seeds[0]))
        chosen_lr,chosen_margin=.001,.2
        if tune:
            phase='Triagem de parametros (uma seed)';candidates=[]
            plan([k for lr in rates for margin in margins for k in seed_keys(chosen,lr,margin,seeds[0])])
            for lr in rates:
                for margin in margins:
                    out=trial(chosen,lr,margin,[seeds[0]])[0]
                    candidates.append(dict(lr=lr,margin=margin,**load(Path(out)/'results.json')['combinado']['full']))
            best=max(candidates,key=key);chosen_lr,chosen_margin=best['lr'],best['margin']
            save(run/'parameter_screening.json',dict(seed=seeds[0],rows=candidates,chosen=best,note='Triagem provisoria, ainda nao confirmada nas outras seeds'))
            phase='Confirmacao com tres seeds';trial(chosen,chosen_lr,chosen_margin,seeds)
        phase='Ensembles e relatorio'
        confirmed={r['trial'] for r in records if sum(x['trial']==r['trial'] and x['kind']=='individual' for x in records)==3}
        plan([tid+'/ensemble' for tid in sorted(confirmed)])
        for tid in sorted(confirmed):
            members=sorted([r for r in records if r['trial']==tid and r['kind']=='individual'],key=lambda r:r['seed'])
            output=stage(tid+'/ensemble',lambda:ensemble.build([m['output'] for m in members],weights,stop,log,allow_smoke=smoke))
            records.append(dict(members[0],kind='ensemble',seed=-1,n_seeds=3,output=str(output.resolve())))
        save(run/'records.json',records);rows=summarize(run,records,phase,models[0])
        final=max([r for r in rows if r['kind']=='ensemble'],key=key)
        save(run/'recommended.json',dict(output=final['outputs'][0],metrics=final,criterion='validation_recall_10_then_mrr',
            note='Configuracao escolhida nesta validacao. Nao e teste independente.'))
        status('CONCLUIDO','Concluido')
    except InterruptedError:status('INTERROMPIDO')
    except Exception as exc:
        status('ERRO',error=str(exc));raise
    return run
