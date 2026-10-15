"""Resumable controlled ablations; validation only, never a final-test sweep."""
from .portable import Path
import html

import numpy as np

from . import engine, ensemble, fusion
from .core import ROOT, atomic, check_stop, csv_write, digest, load, new_run, read_image, save, sha
from .data import cohort
from .preprocessing import prepare_image
from .weight_search import rank_metrics

CONDITIONS = {
    'original': ('Original', 'original', 'random'),
    'hard': ('Candidatos dificeis', 'original', 'hard'),
    'crop': ('Enquadramento', 'uniform_background', 'random'),
    'hard_crop': ('Enquadramento + candidatos dificeis', 'uniform_background', 'hard'),
}


def file_signatures(path):
    path = Path(path)
    files = [p for name in ('config.json','request.json','cohort.json','results.json','scores.npz') if (p := path/name).exists()]
    files += [p for name in ('config.json','best.pt','evaluation.json') for p in path.glob('*/'+name)]
    signatures = {str(p.resolve()):sha(p) for p in files}
    if (path/'config.json').exists():
        config = load(path/'config.json')
        for child in config.get('children',{}).values():
            for name in ('best.pt','config.json'):
                file = Path(child['run'])/name
                signatures[str(file.resolve())] = sha(file)
        for member in config.get('ensemble_members',[]) + config.get('rank_fusion_members',[]):
            signatures.update(file_signatures(member['run']))
    return signatures


def check_signatures(signatures):
    if any(not Path(p).exists() or sha(p) != token for p,token in signatures.items()):
        raise ValueError('Uma entrada ou saida concluida foi alterada; inicie outro pipeline.')


def paired_gain(evaluations, baselines):
    if not evaluations or not baselines:
        raise ValueError('Comparacao sem resultados')
    reference = evaluations[0]
    for result in [*evaluations,*baselines]:
        if result['galleries'] != reference['galleries']:
            raise ValueError('Galerias diferentes')
        if [r['person'] for r in result['rows']] != [r['person'] for r in reference['rows']]:
            raise ValueError('Identidades diferentes')
    def hits(results):
        return np.mean([[float(r['rank']==1) for r in result['rows']] for result in results],axis=0)
    values = hits(evaluations)-hits(baselines)
    rng = np.random.default_rng(42)
    interval = np.round(np.quantile(values[rng.integers(0,len(values),(2000,len(values)))].mean(1),[.025,.975]),12).tolist()
    delta = float(values.mean())
    return dict(delta_recall_1=0. if abs(delta)<1e-12 else delta, ci95=interval,
        conclusion='ganho inconclusivo' if interval[0] <= 0 <= interval[1] else 'melhora' if interval[0] > 0 else 'piora')


def summarize(run, records):
    groups = {}
    for record in records:
        groups.setdefault((record['loss'],record['condition'],record['kind']),[]).append(record)
    rows, examples, subsets = [], [], []
    for record in records:
        source = Path(record['output'])
        cache = run/'subsets'/(digest(record['output'])+'.json')
        if cache.exists():
            subsets.extend(load(cache))
            continue
        cfg = load(source/'config.json')
        people = [p for p in load(source/'cohort.json')['people'] if p['split']=='validacao']
        with np.load(source/'scores.npz',allow_pickle=False) as arrays:
            matrices = [arrays[n].copy() for n in fusion.COMPONENTS]
        entries = []
        for bits in ('100','010','001','110','101','011','111'):
            weights = [int(b)*w for b,w in zip(bits,cfg['weights'])]
            if not any(weights):
                continue
            entries.append(dict(record,subset=bits,weights=weights,**rank_metrics(fusion.combine(matrices,cfg['normalization'],weights),[p['person'] for p in people])))
        save(cache,entries)
        subsets.extend(entries)
    for (loss,condition,kind), members in groups.items():
        members.sort(key=lambda r:r['seed'])
        evaluations = [load(Path(r['output'])/'results.json')['combinado'] for r in members]
        row = dict(loss=loss, condition=condition, label=CONDITIONS.get(condition,('Historico',))[0], kind=kind,
                   seeds=len(members) if kind=='individual' else members[0]['n_seeds'],
                   n=evaluations[0]['full']['n'], outputs=[r['output'] for r in members])
        for metric in ('recall_1','recall_3','recall_5','recall_10','mrr'):
            row[metric] = float(np.mean([e['full'][metric] for e in evaluations]))
            row[metric+'_sd'] = float(np.std([e['full'][metric] for e in evaluations],ddof=1)) if len(evaluations)>1 else None
        row['failures'] = sum(e['full']['failures'] for e in evaluations)
        row['gallery_recall_1'] = float(np.mean([e['gallery']['recall_1'] for e in evaluations]))
        baseline = groups.get((loss,'original',kind))
        if baseline and condition != 'historical':
            baseline = sorted(baseline,key=lambda r:r['seed'])
            if [r['seed'] for r in baseline] == [r['seed'] for r in members]:
                row.update(paired_gain(evaluations,[load(Path(r['output'])/'results.json')['combinado'] for r in baseline]))
        row.setdefault('delta_recall_1',None)
        row.setdefault('ci95',None)
        row.setdefault('conclusion','referencia contextual' if condition=='historical' else 'baseline pendente')
        if kind == 'ensemble' and (individuals := groups.get((loss,condition,'individual'))):
            comparison = paired_gain(evaluations,[load(Path(r['output'])/'results.json')['combinado'] for r in individuals])
            row['delta_vs_individual_mean'] = comparison['delta_recall_1']
            row['ensemble_ci95'] = comparison['ci95']
            row['ensemble_conclusion'] = comparison['conclusion']
        rows.append(row)
        # One hit and one miss from the first seed, clearly labeled.
        evaluation, member = evaluations[0], members[0]
        for hit in (True,False):
            case = next((r for r in evaluation['rows'] if (r['rank']==1)==hit),None)
            if case:
                examples.append(dict(loss=loss,condition=condition,kind=kind,seed=member['seed'],output=member['output'],**case))
    rows.sort(key=lambda r:(r['recall_1'],r['mrr']),reverse=True)
    summary = dict(rows=rows, examples=examples, primary_metric='recall_1', split='validacao', subsets=subsets, analysis_hash=sha(__file__),
                   note='Exploracao na mesma validacao; ganhos pareados por identidade. Seeds nao sao novas pessoas. O teste final permanece reservado.')
    save(run/'summary.json',summary)
    csv_write(run/'results.csv',rows)
    csv_write(run/'examples.csv',examples)
    csv_write(run/'subsets.csv',subsets)
    headers = ('loss','label','kind','recall_1','recall_3','recall_5','recall_10','mrr','delta_recall_1','conclusion')
    report = '<!doctype html><meta charset="utf-8"><title>Pipeline Face-Cranio</title><h1>Comparacao por Recall@1</h1><p>'+html.escape(summary['note'])+'</p><table border="1"><tr>'
    report += ''.join('<th>'+k+'</th>' for k in headers)+'</tr>'
    for row in rows:
        report += '<tr>'+''.join('<td>'+html.escape(str(row.get(k,'')))+'</td>' for k in headers)+'</tr>'
    report += '</table><h2>Exemplos visuais e rankings</h2><ul>'
    for record in records:
        report += '<li><a href="'+html.escape(str(Path(record['output'])/'report.html'),quote=True)+'">'+html.escape(f'{record["loss"]} / {record["condition"]} / {record["kind"]} / {record["seed"]}')+'</a></li>'
    atomic(run/'report.html',(report+'</ul>').encode())


def previews(run, people, stop=None):
    records = []
    for person in [p for p in people if p['split']=='treino'][:3]:
        for view,row in person['views'].items():
            if view not in ('FFN','FFS','FL','CF','CD'):
                continue
            check_stop(stop)
            try:
                image = read_image(row['path'])
                crop,info = prepare_image(image,'uniform_background')
                path = run/'previews'/f'{len(records)}.jpg'
                path.parent.mkdir(exist_ok=True)
                crop.thumbnail((480,480))
                crop.save(path)
                records.append(dict(person=person['person'],view=view,source=row['path'],preview=str(path.resolve()),**info))
            except (ValueError,OSError) as exc:
                records.append(dict(person=person['person'],view=view,error=str(exc)))
    save(run/'previews.json',records)


def run_pipeline(manifest, run_dir=None, conditions=None, losses=None, seeds=None,
                 weights=(.5,.3,.2), reference_sources=None, existing_only=False, stop=None, log=print, **params):
    conditions = list(dict.fromkeys(['original',*(list(CONDITIONS) if conditions is None else conditions)]))
    losses, seeds = losses or ['infonce','triplet'], seeds or [42,43,44]
    if any(c not in CONDITIONS for c in conditions) or any(l not in ('infonce','triplet') for l in losses):
        raise ValueError('Condicao ou loss invalida')
    if len(seeds)<3 or len(set(seeds))!=len(seeds) or len(set(losses))!=len(losses):
        raise ValueError('Use ao menos tres seeds distintas e losses sem repeticoes')
    if not existing_only and set(conditions)&{'hard','hard_crop'} and params.get('hard_warmup',5) >= params.get('epochs',100):
        raise ValueError('O limite de epocas precisa superar as epocas iniciais sem mineracao.')
    fusion.combine([np.zeros((1,1))]*3,[dict(mean=0.,std=1.)]*3,weights)
    run = Path(run_dir) if run_dir else new_run('improvement_pipelines')
    run.mkdir(parents=True,exist_ok=True)
    c = cohort(manifest,'E6D','matched',params.get('ids'))
    inputs = {row['path']:sha(row['path']) if Path(row['path']).exists() else 'MISSING'
              for person in c['people'] if person['split'] in ('treino','validacao')
              for view,row in person['views'].items() if view in ('FFN','FFS','FL','CF','CD')}
    reference_sources = reference_sources or {}
    params = dict(params, selection_metric='gallery_recall_1')
    request = dict(manifest=str(Path(manifest).resolve()),manifest_hash=sha(manifest),conditions=conditions,
        losses=losses,seeds=seeds,weights=list(weights),reference_sources=reference_sources,params=params,existing_only=existing_only,
        reference_hashes={str(p):file_signatures(p) for sources in reference_sources.values() for p in sources},
        input_hashes=inputs,
        code_hash=digest({name:sha(ROOT/name) for name in ('engine.py','backbone.py','data.py','fusion.py','core.py','preprocessing.py','ensemble.py','weight_search.py','improvement_pipeline.py')}))
    if (run/'request.json').exists() and load(run/'request.json')!=request:
        raise ValueError('Retomada incompativel com dados, parametros ou codigo.')
    save(run/'request.json',request)
    save(run/'cohort.json',c)
    ledger = load(run/'ledger.json') if (run/'ledger.json').exists() else {}
    records = []
    if existing_only and any(not reference_sources.get(loss) for loss in losses):
        raise ValueError('Faltam tres seeds existentes para uma das losses selecionadas')
    total = sum(1+len(reference_sources[l]) for l in losses if reference_sources.get(l)) + (0 if existing_only else len(losses)*len(conditions)*(2*len(seeds)+1))
    done = 0
    def stage(key, function):
        nonlocal done
        check_stop(stop)
        save(run/'status.json',dict(state='RODANDO',stage=key,done=done,total=total))
        log(f'Pipeline {done+1}/{total}: {key}',flush=True)
        if key in ledger:
            check_signatures(ledger[key]['hashes'])
            output = Path(ledger[key]['output'])
            log('Reutilizando etapa concluida: '+key,flush=True)
        else:
            output = Path(function())
            status = load(output/'status.json') if (output/'status.json').exists() else {}
            if status.get('state')!='CONCLUIDO':
                raise InterruptedError('Etapa interrompida')
            ledger[key] = dict(output=str(output.resolve()),hashes=file_signatures(output))
            save(run/'ledger.json',ledger)
        done += 1
        return output
    def record(output, loss, condition, kind, seed):
        records.append(dict(output=str(Path(output).resolve()),loss=loss,condition=condition,kind=kind,seed=seed,n_seeds=len(seeds)))
        save(run/'records.json',records)
        summarize(run,records)
    try:
        save(run/'status.json',dict(state='RODANDO',done=0,total=total))
        if not (run/'previews.json').exists():
            previews(run,c['people'],stop)
        for loss in losses:
            sources = reference_sources.get(loss,[])
            if sources:
                original_sources = sources
                sources = []
                # Context only: old checkpoints may have been selected by MRR.
                for source in original_sources:
                    cfg = load(Path(source)/'config.json')
                    old = load(Path(cfg['bundle'])/'frontal/config.json')
                    people = [p['person'] for p in load(Path(source)/'cohort.json')['people'] if p['split']=='validacao']
                    if people != [p['person'] for p in c['people'] if p['split']=='validacao']:
                        raise ValueError('Historico usa outras identidades; desative a referencia historica.')
                    output = stage(f'{loss}/historical/{old["seed"]}',lambda: fusion.reweight_saved(source,weights,stop,log))
                    sources.append(str(output.resolve()))
                    record(output,loss,'historical','individual',old['seed'])
                output = stage(loss+'/historical/ensemble',lambda:ensemble.build(sources,weights,stop,log,allow_smoke=params.get('smoke',False)))
                record(output,loss,'historical','ensemble',-1)
            if existing_only:
                continue
            for condition in conditions:
                _, preprocessing, negative_sampling = CONDITIONS[condition]
                sources = []
                for seed in seeds:
                    bundle = run/loss/condition/f'seed{seed}'
                    output = stage(f'{loss}/{condition}/{seed}/train',lambda: fusion.train_components(
                        manifest,run_dir=bundle,stop=stop,log=log,loss=loss,seed=seed,
                        preprocessing=preprocessing,negative_sampling=negative_sampling,**params))
                    output = stage(f'{loss}/{condition}/{seed}/scores',lambda: fusion.fuse(bundle,weights,stop,log))
                    sources.append(str(output.resolve()))
                    record(output,loss,condition,'individual',seed)
                output = stage(f'{loss}/{condition}/ensemble',lambda:ensemble.build(sources,weights,stop,log,allow_smoke=params.get('smoke',False)))
                record(output,loss,condition,'ensemble',-1)
        save(run/'status.json',dict(state='CONCLUIDO',done=done,total=total))
    except InterruptedError:
        save(run/'status.json',dict(state='INTERROMPIDO',done=done,total=total))
    except Exception as exc:
        save(run/'status.json',dict(state='ERRO',done=done,total=total,error=str(exc)))
        raise
    return run
