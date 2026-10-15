"""Reproducible batches of known queries, using the regular matching inference."""
import random

from .portable import Path
from .core import check_stop, load, new_run, save, sha
from .matching_summary import query_summary


def samples(people, queries=10, candidates=10, seed=42):
    people=sorted(people,key=lambda p:p['person'])
    if len({p['person'] for p in people})!=len(people):
        raise ValueError('Identidades duplicadas na coorte')
    if not 1 <= queries <= len(people) or not 2 <= candidates <= len(people):
        raise ValueError('Quantidade de consultas ou candidatos invalida')
    rng=random.Random(seed)
    result=[]
    for query in rng.sample(people,queries):
        gallery=rng.sample([p for p in people if p['person']!=query['person']],candidates-1)+[query]
        rng.shuffle(gallery)
        result.append(dict(query=query,candidates=gallery))
    return result


def summarize(rows,total):
    return dict(planned=total,processed=len(rows),failures=sum(bool(r.get('error')) or r.get('Posicao do correto') is None for r in rows),
                **{f'top_{k}':dict(hits=sum(r.get(f'Top-{k}')=='Sim' for r in rows),total=total,
                                  recall=sum(r.get(f'Top-{k}')=='Sim' for r in rows)/total) for k in (1,3,5)})


def run_batch(run,queries=10,candidates=10,seed=42,weights=None,all_subsets=False,stop=None,log=print):
    from . import engine, fusion
    model=Path(run);cfg=load(model/'config.json')
    combined='children' in cfg or bool(cfg.get('rank_fusion_members'))
    people=[p for p in load(model/'cohort.json')['people'] if p['split']=='validacao']
    selections=samples(people,queries,candidates,seed)
    output=new_run('matching_batches')
    save(output/'request.json',dict(model=str(model),queries=queries,candidates=candidates,seed=seed,
                                   weights=weights,all_subsets=all_subsets,samples=selections,
                                   model_config=cfg,model_config_hash=sha(model/'config.json')))
    rows=[];combination_rows={}
    def snapshot(state):
        save(output/'summary.json',dict(**summarize(rows,queries),rows=rows,
             combinations={k:summarize(v,queries) for k,v in combination_rows.items()}))
        save(output/'status.json',dict(state=state,done=len(rows),total=queries))
    snapshot('RODANDO')
    try:
        for i,selection in enumerate(selections):
            check_stop(stop)
            log(f'Matching em lote: {i+1}/{queries} | {selection["query"]["person"]}',flush=True)
            try:
                if combined:
                    single=fusion.compare_fusion(str(model),**selection,weights=weights,all_subsets=all_subsets,stop=stop,log=log)
                    result=load(single/'result.json')
                else:
                    result=engine.compare(str(model),**selection,stop=stop,log=log)
                    single=new_run('comparisons');save(single/'result.json',result)
                rows.append(dict(query_summary(result),output=str(single),error=None))
                for bits,entry in result.get('combinations',{}).items():
                    combination_rows.setdefault(bits,[]).append(query_summary(result,entry['ranking']))
            except InterruptedError:
                raise
            except (OSError,ValueError,RuntimeError) as exc:
                rows.append(dict(query_summary(dict(selection,ranking=[])),error=str(exc),output=None))
                log('Falha registrada: '+str(exc),flush=True)
            snapshot('RODANDO')
        snapshot('CONCLUIDO')
    except InterruptedError:
        snapshot('INTERROMPIDO')
    return output


def controls(run,weights,all_subsets,launch,busy):
    import secrets
    import streamlit as st
    people=[p for p in load(Path(run)/'cohort.json')['people'] if p['split']=='validacao']
    with st.expander('Rodar varios sorteios automaticamente',expanded=False):
        a,b,c=st.columns(3)
        queries=a.number_input('Consultas no lote',1,len(people),min(10,len(people)),key='batch_queries')
        count=b.number_input('Cranios por consulta',2,len(people),min(10,len(people)),key='batch_candidates')
        fixed=c.checkbox('Repetir lote pela seed',value=False,key='batch_fixed')
        seed=c.number_input('Seed do lote',0,999999,42,key='batch_seed')
        st.caption('Faces distintas, candidatos distintos em cada consulta e correto incluido uma vez. Usa o modelo e os pesos selecionados acima. Nao treina; os mapas de sensibilidade nao sao gerados automaticamente neste lote.')
        if st.button('Rodar lote de Matching',disabled=busy(),icon=':material/play_arrow:'):
            launch('matching_batch',dict(run=str(run),queries=queries,candidates=count,
                                        seed=seed if fixed else secrets.randbelow(1000000),weights=weights,all_subsets=all_subsets))


def results():
    import streamlit as st
    from .core import ARTIFACTS
    from .reporting import newest_first, started
    from zoneinfo import ZoneInfo
    paths=newest_first(list((ARTIFACTS/'matching_batches').glob('*/summary.json')))
    if not paths:return
    def label(p):
        date=started(p)
        return (date.astimezone(ZoneInfo('America/Sao_Paulo')).strftime('%d/%m/%Y %H:%M:%S') if date else p.parent.name)+' | '+p.parent.name
    with st.expander('Resultados dos lotes de Matching',expanded=True):
        path=st.selectbox('Lote salvo',paths,format_func=label)
        report=load(path);status=load(path.parent/'status.json')
        st.caption(f'{status["state"]} | {report["processed"]}/{report["planned"]} consultas processadas | {report["failures"]} falhas/sem posicao. Percentuais usam o total planejado; lote interrompido nao e resultado final.')
        for col,k in zip(st.columns(3),(1,3,5)):
            m=report[f'top_{k}'];col.metric(f'Lote: Top-{k}',f'{m["hits"]}/{m["total"]}',f'{m["recall"]:.1%}',delta_color='off')
        st.dataframe(report['rows'],hide_index=True)
        if report['combinations']:
            st.dataframe([dict(Combinacao=' + '.join(n for n,b in zip(('Neutra','Sorrindo','Lateral'),bits) if b=='1'),
                               **{f'Top-{k}':f'{m[f"top_{k}"]["hits"]}/{m[f"top_{k}"]["total"]}' for k in (1,3,5)}) for bits,m in report['combinations'].items()],hide_index=True)
        successful=[r for r in report['rows'] if r.get('output')]
        if successful:
            index=st.selectbox('Consulta do lote',range(len(successful)),format_func=lambda i:successful[i]['Pessoa de referencia'])
            result=load(Path(successful[index]['output'])/'result.json')
            st.dataframe(result['ranking'],hide_index=True)
        with st.expander('Parametros e sorteios do lote'):st.json(load(path.parent/'request.json'),expanded=False)
        st.download_button('Baixar lote (JSON)',path.read_bytes(),'matching_lote.json')
