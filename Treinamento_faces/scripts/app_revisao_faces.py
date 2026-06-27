# -*- coding: utf-8 -*- 
from __future__ import annotations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['GLOG_minloglevel']='3'
os.environ.setdefault('MPLCONFIGDIR','/tmp/treinamento_faces_mpl')

import html
import subprocess
import sys
import time
from pathlib import Path 

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

try:
    import mediapipe as mp
except Exception:
    mp=None

from common_faces import *


st.set_page_config(page_title='Treinamento Faces', page_icon='🧠', layout='wide')

st.markdown(
    """
    <style>
    .main .block-container{padding-top:1rem;max-width:1680px}
    div[data-testid="stMetric"]{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:12px}
    .small-muted{color:#6b7280;font-size:.88rem}
    .gallery-caption{font-size:.82rem;color:#374151;line-height:1.35;margin:.3rem 0 1rem;padding:.45rem .55rem;border-left:3px solid #d1d5db;background:#f9fafb}
    .gallery-caption strong{color:#111827}
    .gallery-caption .gender-auto{color:#166534;font-weight:700}
    .gallery-caption .gender-manual{color:#1d4ed8;font-weight:700}
    .gallery-caption .gender-review{color:#a16207;font-weight:700}
    </style>
    """,
    unsafe_allow_html=True,
)

SCRIPTS={
    'pipeline':'run_pipeline_faces.py',
    'check':'00_checar_dataset_faces.py',
    'landmarks':'01_extrair_landmarks_faces.py',
    'medidas':'02_gerar_medidas_faces.py',
    'analise':'05_analisar_padroes_faces.py',
    'html':'03_relatorio_landmarks_html.py',
    'curadoria':'curadoria_matching_faces.py',
    'fairface':'06_predizer_genero_fairface_faces.py',
    'curadoria_genero':'07_curadoria_genero_faces.py',
    'features_genero':'08_preparar_features_genero_faces.py',
    'treinar_genero':'09_treinar_modelo_genero_faces.py',
    'comparar_genero':'10_comparar_genero_fairface_modelo.py',
    'exportar_genero':'11_exportar_matching_faces_com_genero.py',
}


def guess_root():
    here=Path(__file__).resolve()
    return here.parent.parent if here.parent.name=='scripts' else Path.cwd()


def script_path(root,key):
    return root/'scripts'/SCRIPTS[key]


def project_python(root):
    for p in [root/'.venv'/'bin'/'python', root/'.venv'/'Scripts'/'python.exe']:
        if p.exists():
            return p
    return Path(sys.executable)


def build_cmd(root,key,years=None,force=False,extra=None):
    sp=script_path(root,key)
    if not sp.exists():
        st.error(f'Script não encontrado: {sp}')
        return None
    cmd=[str(project_python(root)),str(sp),'--root',str(root)]
    if years:
        cmd+=['--years']+[str(y) for y in years]
    if force:
        cmd.append('--force')
    if key in ['pipeline','html']:
        cmd.append('--no-open')
    if extra:
        cmd+=extra
    return cmd


def bgr_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


@st.cache_resource(show_spinner=False)
def get_face_mesh():
    if mp is None or not hasattr(mp,'solutions'):
        return None
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=.5)


def detect_local(img):
    fm=get_face_mesh()
    if fm is None:
        return None
    res=fm.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    return None if not res.multi_face_landmarks else res.multi_face_landmarks[0].landmark


def run_command_live(cmd,cwd,title):
    if not cmd:
        return 1
    st.code(' '.join(map(str,cmd)), language='bash')
    follow=st.toggle('Auto-scroll: mostrar sempre o final do log', value=True, key=f'follow_{title}_{time.time()}')
    box=st.empty()
    prog=st.progress(0,text=title)
    logs=[]
    try:
        env=os.environ.copy()
        env.setdefault('MPLCONFIGDIR',str(resultados_root(cwd)/'cache'/'matplotlib'))
        Path(env['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
        proc=subprocess.Popen(cmd,cwd=str(cwd),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
    except Exception as e:
        st.error(f'Erro ao iniciar: {e}')
        return 1
    start=time.time()
    def render():
        text='\n'.join(logs[-260:] if follow else logs[-900:])
        if follow:
            safe=html.escape(text)
            box.empty()
            with box.container():
                components.html(
                    f"""
                    <div id="logbox" style="height:360px;overflow:auto;background:#0f172a;color:#e5e7eb;
                    border-radius:8px;padding:12px;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
                    white-space:pre-wrap;">{safe}</div>
                    <script>
                    const el = document.getElementById("logbox");
                    if (el) el.scrollTop = el.scrollHeight;
                    </script>
                    """,
                    height=390,
                )
        else:
            box.code(text, language='text', height=360)
    while True:
        line=proc.stdout.readline()
        if line:
            logs.append(line.rstrip())
            prog.progress(min(95,int((time.time()-start)%95)), text=f'{title}: {len(logs)} linhas')
            render()
        if proc.poll() is not None:
            rest=proc.stdout.read()
            if rest:
                logs.extend(rest.splitlines())
            break
    prog.progress(100,text='Finalizado')
    render()
    logdir=resultados_root(cwd)/'logs'
    logdir.mkdir(parents=True,exist_ok=True)
    lp=logdir/f"streamlit_{title.lower().replace(' ','_')}_{int(time.time())}.log"
    lp.write_text('\n'.join(logs),encoding='utf-8',errors='replace')
    if proc.returncode==0:
        st.success(f'{title} finalizado. Log: {lp}')
    else:
        st.error(f'{title} terminou com erro {proc.returncode}. Log: {lp}')
    return proc.returncode


def st_df(df,height=360,key=None):
    if df is None or df.empty:
        st.info('Sem dados para mostrar.')
        return
    try:
        st.dataframe(style_dataframe(df), width='stretch', hide_index=True, height=height, key=key)
    except Exception:
        st.dataframe(df, width='stretch', hide_index=True, height=height, key=key)


def status_chip(v):
    s=str(v).upper()
    if any(x in s for x in ['OK','TRUE','SIM','ALTA','MASCULINO','FEMININO','CONCORDANTE']):
        return '🟢'
    if any(x in s for x in ['REVISAR','PENDENTE','MEDIA','BAIXA','INDEFINIDO','DIVERGENTE']):
        return '🟡'
    if any(x in s for x in ['FALHA','EXCLUIR','ERRO','NAO_USAR','FALSE']):
        return '🔴'
    if any(x in s for x in ['CORRIGIR','APLICAR']):
        return '🔵'
    return ''


def selectable_df(df,key,height=320):
    if df is None or df.empty:
        st.info('Sem dados para selecionar.')
        return None
    view=df.copy().reset_index(drop=True)
    status_cols=[c for c in ['qualidade_landmarks','status_final','status_manual','classe_fairface','confianca_fairface','genero_final','concordancia_fairface_modelo'] if c in view.columns]
    if status_cols:
        view.insert(0,'cor',view[status_cols].astype(str).agg(' '.join,axis=1).map(status_chip))
    try:
        ev=st.dataframe(view, width='stretch', hide_index=True, height=height, key=key, on_select='rerun', selection_mode='single-row')
        rows=getattr(getattr(ev,'selection',None),'rows',[])
        if rows:
            return df.iloc[int(rows[0])]
    except Exception:
        st_df(view,height=height,key=f'{key}_styled')
    opts=[f"{r.get('ano','')} | {r.get('id_imagem','')} | {r.get(status_cols[0], '') if status_cols else ''}" for _,r in df.iterrows()]
    choice=st.selectbox('Selecionar item',opts,key=f'{key}_fallback')
    return df.iloc[opts.index(choice)] if opts else None


def review_navigator(df,key,label_col='status_final',height=320):
    if df is None or df.empty:
        st.info('Nenhum item nesta lista.')
        return None
    df=df.reset_index(drop=True)
    idx_key=f'{key}_idx'
    if idx_key not in st.session_state:
        st.session_state[idx_key]=0
    st.session_state[idx_key]=max(0,min(int(st.session_state[idx_key]),len(df)-1))
    opts=[]
    for _,r in df.iterrows():
        label=str(r.get(label_col,'')) if label_col in df.columns else ''
        opts.append(f"{r.get('ano','')} | {r.get('id_imagem','')} | {label}")
    choice=st.selectbox('Item',opts,index=st.session_state[idx_key],key=f'{key}_select')
    st.session_state[idx_key]=opts.index(choice)
    st.caption(f'Item {st.session_state[idx_key]+1} de {len(df)}')
    st_df(df, height)
    return df.iloc[st.session_state[idx_key]]


def _move_review(key,opts,delta):
    idx_key=f'{key}_idx'
    idx=max(0,min(len(opts)-1,int(st.session_state.get(idx_key,0))+delta))
    st.session_state[idx_key]=idx
    st.session_state[f'{key}_select']=opts[idx]


def review_footer(df,key,label_col='status_final'):
    if df is None or df.empty:
        return
    opts=[]
    for _,r in df.reset_index(drop=True).iterrows():
        label=str(r.get(label_col,'')) if label_col in df.columns else ''
        opts.append(f"{r.get('ano','')} | {r.get('id_imagem','')} | {label}")
    idx=max(0,min(len(opts)-1,int(st.session_state.get(f'{key}_idx',0))))
    left,center,right=st.columns([1,2,1])
    left.button('Anterior',key=f'{key}_prev',width='stretch',disabled=idx<=0,on_click=_move_review,args=(key,opts,-1))
    center.markdown(f"<p style='text-align:center;margin:.55rem 0 0'>Item <b>{idx+1}</b> de <b>{len(opts)}</b></p>",unsafe_allow_html=True)
    right.button('Próximo',key=f'{key}_next',width='stretch',disabled=idx>=len(opts)-1,on_click=_move_review,args=(key,opts,1))


def show_image(root,year,stem,mode='marcacao'):
    ip=find_image_path(root,year,stem)
    pp=preview_path(root,year,stem)
    if mode=='original':
        if ip and ip.exists():
            st.image(str(ip), width='stretch', caption='Original sem marcação')
        else:
            st.warning('Imagem original não encontrada.')
        return
    if mode=='marcacao':
        if pp.exists():
            st.image(str(pp), width='stretch', caption='Com marcação')
            return
        if ip and ip.exists():
            img=read_image(ip)
            lms=detect_local(img) if img is not None else None
            if img is not None and lms is not None:
                st.image(bgr_to_pil(draw_landmarks_preview(img,lms)), width='stretch', caption='Marcação recalculada para visualização')
                return
        st.warning('Marcação não encontrada.')
        return
    a,b=st.columns(2)
    with a:
        show_image(root,year,stem,'original')
    with b:
        show_image(root,year,stem,'marcacao')


def image_file_for_mode(root,year,stem,mode):
    ip=find_image_path(root,year,stem)
    pp=preview_path(root,year,stem)
    if mode=='original':
        return ip if ip and ip.exists() else None
    if mode=='marcacao':
        if pp.exists():
            return pp
        return ip if ip and ip.exists() else None
    return ip if ip and ip.exists() else None


def gallery_card(root,row,mode):
    y=str(row.get('ano',''))
    stem=str(row.get('id_imagem',''))
    qualidade=str(row.get('qualidade_landmarks',''))
    motivo=str(row.get('motivo_qualidade',''))
    cap=f'{y} | {stem}'
    if qualidade:
        cap+=f' | {qualidade}'
    if mode=='lado a lado':
        c1,c2=st.columns(2,gap='small')
        with c1:
            ip=image_file_for_mode(root,y,stem,'original')
            if ip:
                st.image(str(ip), width='stretch')
            else:
                st.warning('Original ausente')
            st.caption('Original')
        with c2:
            pp=image_file_for_mode(root,y,stem,'marcacao')
            if pp:
                st.image(str(pp), width='stretch')
            else:
                st.warning('Marcação ausente')
            st.caption('Marcação')
    else:
        img=image_file_for_mode(root,y,stem,mode)
        if img:
            st.image(str(img), width='stretch')
        else:
            st.warning('Imagem ausente')
    genero=str(row.get('genero_final','')).upper() or 'SEM DEFINIÇÃO'
    origem=str(row.get('origem_genero_final','')).upper()
    if 'ALTA_CONFIANCA' in origem:
        metodo='Automático · alta confiança'; metodo_class='gender-auto'
    elif 'MANUAL' in origem:
        metodo='Manual'; metodo_class='gender-manual'
    else:
        metodo='Aguardando revisão'; metodo_class='gender-review'
    def prob_text(value):
        try:
            value=float(value)
            return '—' if np.isnan(value) else f'{value*100:.1f}%'
        except Exception:
            return '—'
    pm=prob_text(row.get('prob_masculino_fairface',''))
    pf=prob_text(row.get('prob_feminino_fairface',''))
    confianca=str(row.get('confianca_fairface','')).upper() or '—'
    detalhe=(
        f'<div class="gallery-caption"><strong>{html.escape(cap)}</strong>'
        f'<br>Gênero: <strong>{html.escape(genero)}</strong> · <span class="{metodo_class}">{html.escape(metodo)}</span>'
        f'<br>FairFace: M {pm} · F {pf} · confiança <strong>{html.escape(confianca)}</strong>'
        f'<br>{html.escape(motivo[:120])}</div>'
    )
    st.markdown(detalhe,unsafe_allow_html=True)


def comparison_card(root,row,mode='lado a lado'):
    y=str(row.get('ano','')); stem=str(row.get('id_imagem',''))
    if mode=='lado a lado':
        a,b=st.columns(2,gap='small')
        with a:
            ip=image_file_for_mode(root,y,stem,'original')
            st.image(str(ip),width='stretch') if ip else st.warning('Original ausente')
            st.caption('Original')
        with b:
            pp=image_file_for_mode(root,y,stem,'marcacao')
            st.image(str(pp),width='stretch') if pp else st.warning('Marcação ausente')
            st.caption('Marcação')
    else:
        img=image_file_for_mode(root,y,stem,mode)
        st.image(str(img),width='stretch') if img else st.warning('Imagem ausente')
    def pct(v):
        try:
            v=float(v); return '—' if np.isnan(v) else f'{v*100:.1f}%'
        except Exception: return '—'
    result=str(row.get('resultado_modelo_curadoria','SEM_DEFINICAO')).upper()
    result_label='ACERTO' if result=='ACERTO' else ('DIVERGÊNCIA' if result=='DIVERGENCIA' else 'SEM DEFINIÇÃO')
    result_class='gender-auto' if result=='ACERTO' else 'gender-review'
    origem=str(row.get('origem_genero_final','')).upper()
    metodo='Automático' if 'ALTA_CONFIANCA' in origem else ('Manual' if 'MANUAL' in origem else 'Não definido')
    detail=(
        f'<div class="gallery-caption"><strong>{html.escape(y)} | {html.escape(stem)}</strong> · <span class="{result_class}">{result_label}</span>'
        f'<br>FairFace: <strong>{html.escape(str(row.get("genero_fairface","")))}</strong> · M {pct(row.get("prob_masculino_fairface"))} · F {pct(row.get("prob_feminino_fairface"))}'
        f'<br>Curadoria: <strong>{html.escape(str(row.get("genero_curadoria","")))}</strong> · {html.escape(metodo)}'
        f'<br>Modelo: <strong>{html.escape(str(row.get("genero_modelo","")))}</strong> · M {pct(row.get("prob_masculino_modelo"))} · F {pct(row.get("prob_feminino_modelo"))} · confiança {html.escape(str(row.get("confianca_modelo","—")))}</div>'
    )
    st.markdown(detail,unsafe_allow_html=True)


def _move_page(key,delta,last_page):
    st.session_state[key]=max(1,min(last_page,int(st.session_state.get(key,1))+delta))


def render_gender_comparison(root,years):
    comp=normalize_key_columns(load_genero(root)[4])
    if comp.empty:
        st.info('Execute “3. Comparar resultados” para gerar esta comparação.')
        return
    if 'genero_fairface' not in comp.columns:
        st.warning('A comparação está no formato antigo. Execute novamente “3. Comparar resultados”.')
        return
    fair=comp['genero_fairface'].astype(str).str.upper().value_counts()
    model=comp['genero_modelo'].astype(str).str.upper().value_counts()
    curated=comp['genero_curadoria'].astype(str).str.upper().value_counts()
    cards=st.columns(6)
    cards[0].metric('FairFace · mulheres',int(fair.get('FEMININO',0)))
    cards[1].metric('FairFace · homens',int(fair.get('MASCULINO',0)))
    cards[2].metric('Curadoria · mulheres',int(curated.get('FEMININO',0)))
    cards[3].metric('Curadoria · homens',int(curated.get('MASCULINO',0)))
    cards[4].metric('Modelo · mulheres',int(model.get('FEMININO',0)))
    cards[5].metric('Modelo · homens',int(model.get('MASCULINO',0)))
    results=comp['resultado_modelo_curadoria'].astype(str).str.upper().value_counts()
    st.caption(f"FairFace indefinidos: {int(fair.get('INDEFINIDO',0))} · Modelo indefinidos: {int(model.get('INDEFINIDO',0))} · Acertos do modelo contra a curadoria: {int(results.get('ACERTO',0))} · Divergências: {int(results.get('DIVERGENCIA',0))}")
    f1,f2,f3,f4,f5,f6=st.columns(6)
    ano=f1.selectbox('Ano',['TODOS']+years,key='cmp_ano')
    resultado=f2.selectbox('Resultado',['TODOS','ACERTOS','DIVERGÊNCIAS','SEM DEFINIÇÃO'],key='cmp_resultado')
    genero=f3.selectbox('Gênero curado',['TODOS','MASCULINO','FEMININO'],key='cmp_genero')
    mode=f4.selectbox('Visualização',['lado a lado','marcacao','original'],format_func=lambda x:{'marcacao':'com marcação','original':'sem marcação','lado a lado':'lado a lado'}[x],key='cmp_mode')
    per_row=f5.selectbox('Por fileira',[3,4,5],index=1,key='cmp_cols')
    search=f6.text_input('Buscar imagem',key='cmp_search')
    view=filter_by_year(comp,ano)
    if resultado!='TODOS':
        wanted={'ACERTOS':'ACERTO','DIVERGÊNCIAS':'DIVERGENCIA','SEM DEFINIÇÃO':'SEM_DEFINICAO'}[resultado]
        view=view[view['resultado_modelo_curadoria'].astype(str).str.upper().eq(wanted)].copy()
    if genero!='TODOS':
        view=view[view['genero_curadoria'].astype(str).str.upper().eq(genero)].copy()
    if search:
        view=view[view['id_imagem'].astype(str).str.lower().str.contains(search.strip().lower(),regex=False,na=False)].copy()
    cols=['ano','id_imagem','genero_fairface','genero_curadoria','origem_genero_final','genero_modelo','resultado_modelo_curadoria','prob_masculino_fairface','prob_feminino_fairface','prob_masculino_modelo','prob_feminino_modelo','confianca_modelo']
    st.caption(f'{len(view)} imagem(ns) neste filtro.')
    st_df(view[[c for c in cols if c in view.columns]],360)
    view=view.sort_values(['ano','id_imagem']).reset_index(drop=True)
    nav1,nav2,nav3,nav4=st.columns([1,1,1,1])
    page_size=nav1.selectbox('Imagens por página',[12,24,48,96],index=1,key='cmp_page_size')
    last_page=max(1,int(np.ceil(len(view)/page_size)))
    page_key='cmp_page'
    st.session_state[page_key]=max(1,min(last_page,int(st.session_state.get(page_key,1))))
    page=st.session_state[page_key]
    nav2.button('Anterior',key='cmp_prev',width='stretch',disabled=page<=1,on_click=_move_page,args=(page_key,-1,last_page))
    nav3.markdown(f"<p style='text-align:center;margin:.55rem 0 0'>Página <b>{page}</b> de <b>{last_page}</b></p>",unsafe_allow_html=True)
    nav4.button('Próxima',key='cmp_next',width='stretch',disabled=page>=last_page,on_click=_move_page,args=(page_key,1,last_page))
    page_view=view.iloc[(page-1)*page_size:page*page_size].copy()
    for start in range(0,len(page_view),int(per_row)):
        columns=st.columns(int(per_row),gap='medium')
        for column,(_,row) in zip(columns,page_view.iloc[start:start+int(per_row)].iterrows()):
            with column: comparison_card(root,row,mode)


def filter_by_year(df,year):
    if df.empty or year=='TODOS' or 'ano' not in df.columns:
        return df
    return df[df['ano'].astype(str)==str(year)].copy()


def metric_cards(ana,status,dataset,pred,cur,years):
    cols=st.columns(6)
    cols[0].metric('Anos',len(years))
    cols[1].metric('Imagens',sum(len(list_images_for_year(st.session_state.root,y)) for y in years))
    cols[2].metric('Analisadas',len(ana))
    cols[3].metric('Matching OK',len(dataset))
    cols[4].metric('FairFace',len(pred))
    cols[5].metric('Gênero curado',len(cur))


def page_painel(root,years):
    st.title('Painel')
    ana=load_all_analysis(root,years)
    status,dataset,resumo=load_curadoria(root)
    pred,cur,feat,predm,comp=load_genero(root)
    metric_cards(ana,status,dataset,pred,cur,years)

    c1,c2=st.columns([1.2,1])
    with c1:
        st.subheader('Cobertura por ano')
        st_df(overview_counts(root,years),260)
    with c2:
        st.subheader('Legenda')
        st.markdown('🟢 esperado · 🟡 revisar/atenção · 🔴 erro/excluir · 🔵 ação automática pendente')
        with st.expander('Faixas das métricas'):
            st_df(metric_legend_dataframe(),420)

    tabs=st.tabs(['Revisar imagem original','Revisar matching','Revisar gênero','Modelo'])
    with tabs[0]:
        if not ana.empty and 'qualidade_landmarks' in ana.columns:
            q=ana.groupby(['ano','qualidade_landmarks'],dropna=False).size().reset_index(name='qtd')
            st_df(q,260)
            st.bar_chart(q.pivot(index='ano',columns='qualidade_landmarks',values='qtd').fillna(0))
        else:
            st.info('Rode a análise para ver qualidade.')
    with tabs[1]:
        st_df(resumo,260)
        if not status.empty and 'status_final' in status.columns:
            st.bar_chart(status['status_final'].astype(str).value_counts())
    with tabs[2]:
        resumo_g=read_csv_safe(genero_dir(root)/'resumo_genero_faces_curado.csv')
        st_df(resumo_g,260)
        if not cur.empty and 'genero_final' in cur.columns:
            st.bar_chart(cur['genero_final'].astype(str).value_counts())
    with tabs[3]:
        st_df(read_csv_safe(genero_dir(root)/'metricas_modelo_genero_faces.csv'),240)
        st_df(read_csv_safe(genero_dir(root)/'matriz_confusao_genero_faces.csv'),220)


def page_pipeline(root,years):
    st.title('Pipeline')
    st.caption('Retomar roda sem `--force`: cada script pula o que já existe quando consegue. Refazer usa `--force` onde a etapa suporta e sobrescreve landmarks/previews.')
    st_df(overview_counts(root,years),240)
    sel=st.multiselect('Anos',years,default=years)
    yrs=sel if sel and len(sel)!=len(years) else None
    a,b,c=st.columns([1,1,1])
    with a:
        if st.button('Retomar pipeline faltantes', type='primary', help='Não usa force. Continua do que já existe e processa o que falta.'):
            rc=run_command_live(build_cmd(root,'pipeline',yrs,False),root,'Pipeline retomar')
            if rc==0:
                st.subheader('Resumo atualizado')
                st_df(overview_counts(root,discover_years(root)),240)
                st_df(load_all_analysis(root,discover_years(root)),420)
    with b:
        if st.button('Refazer pipeline com force', help='Usa --force na extração de landmarks/previews e depois recalcula as etapas seguintes.'):
            rc=run_command_live(build_cmd(root,'pipeline',yrs,True),root,'Pipeline force')
            if rc==0:
                st.subheader('Resumo atualizado')
                st_df(overview_counts(root,discover_years(root)),240)
                st_df(load_all_analysis(root,discover_years(root)),420)
    with c:
        if st.button('Limpar gerados sem rodar', help='Apaga resultados, marcações, previews e arquivos gerados; preserva imagens originais.'):
            removed=cleanup_generated_outputs(root,sel if sel else years,keep_logs=False)
            st.success(f'Limpeza concluída. {len(removed)} item(ns) gerados removidos. Originais preservados.')
            st.rerun()
        st.caption('Sem filtro, roda todos os anos detectados.')
    etapa=st.selectbox('Etapa individual',[('check','Checar dataset'),('landmarks','Extrair landmarks'),('medidas','Gerar medidas'),('analise','Analisar padrões'),('html','Gerar relatório HTML')],format_func=lambda x:x[1])
    e1,e2=st.columns(2)
    if e1.button('Rodar etapa selecionada: faltantes'):
        rc=run_command_live(build_cmd(root,etapa[0],yrs,False),root,etapa[1])
        if rc==0:
            st.subheader('Resumo atualizado')
            st_df(overview_counts(root,discover_years(root)),240)
    if e2.button('Rodar etapa selecionada: force'):
        rc=run_command_live(build_cmd(root,etapa[0],yrs,True if etapa[0] in ['landmarks','fairface'] else False),root,f'{etapa[1]} force')
        if rc==0:
            st.subheader('Resumo atualizado')
            st_df(overview_counts(root,discover_years(root)),240)
    st.subheader('Scripts disponíveis')
    st_df(pd.DataFrame([{'etapa':k,'script':v,'existe':script_path(root,k).exists(),'caminho':str(script_path(root,k))} for k,v in SCRIPTS.items()]),420)


def page_matching(root,years):
    st.title('Matching')
    tabs=st.tabs(['Rodar curadoria','Revisar','Tabelas'])
    with tabs[0]:
        sel=st.multiselect('Anos',years,default=years,key='matching_years')
        yrs=sel if sel and len(sel)!=len(years) else None
        c1,c2,c3,c4=st.columns(4)
        min_score=c1.slider('Score mínimo',40.,95.,70.,1.)
        min_gain=c2.slider('Ganho mínimo',0.,20.,3.,.5)
        only=c3.checkbox('Só REVISAR/FALHA',True)
        apply=c4.checkbox('Aplicar nas originais (com backup)',False,help='Só altera a original quando marcado. Antes, salva uma cópia em backup_originais_nitidez; a versão melhorada fica em images_melhoradas_nitidez.')
        extra=['--min-score',str(min_score),'--min-gain',str(min_gain)] + (['--apply'] if apply else []) + ([] if only else ['--all'])
        if st.button('Rodar curadoria matching',type='primary'):
            run_command_live(build_cmd(root,'curadoria',yrs,False,extra),root,'Curadoria matching')
    with tabs[1]:
        status,dataset,resumo=load_curadoria(root)
        status=normalize_key_columns(status)
        if status.empty:
            st.info('Rode a curadoria matching primeiro.')
            return
        f1,f2,f3=st.columns(3)
        fano=f1.selectbox('Ano',['TODOS']+years,key='matching_fano')
        flista=f2.selectbox('Lista',['REVISAR IMAGEM ORIGINAL','REVISAR MATCHING','TODAS','APROVADAS','EXCLUIDAS'],key='matching_lista')
        busca=f3.text_input('Buscar id/iniciais',key='matching_busca')
        df=filter_by_year(status,fano)
        if flista=='REVISAR IMAGEM ORIGINAL':
            qualidade=df.get('qualidade_landmarks',pd.Series('',index=df.index)).astype(str).str.upper()
            df=df[qualidade.eq('REVISAR')].copy()
        elif flista=='REVISAR MATCHING':
            df=df[matching_pending_mask(df)].copy()
        elif flista=='APROVADAS':
            df=df[df.get('usar_matching',pd.Series(False,index=df.index)).astype(str).str.upper().isin(['TRUE','1','SIM','YES'])].copy()
        elif flista=='EXCLUIDAS':
            df=df[df.get('status_final',pd.Series('',index=df.index)).astype(str).str.upper().str.contains('EXCLUIR|NAO_USAR|FALHA|ERRO',regex=True,na=False)].copy()
        if busca:
            needle=busca.strip().lower()
            df=df[df.astype(str).agg(' '.join,axis=1).str.lower().str.contains(needle,regex=False,na=False)].copy()
        st.caption(f'{len(df)} item(ns). Ao salvar OK, qualidade_landmarks passa para OK e o item sai de Revisar imagem original.')
        row=review_navigator(df,'matching_review','status_final',360)
        if row is not None:
            y=str(row.get('ano','')); stem=str(row.get('id_imagem',''))
            a,b=st.columns(2)
            with a:
                show_image(root,y,stem,'marcacao')
            with b:
                show_image(root,y,stem,'original')
            st_df(pd.DataFrame([row.to_dict()]),220)
            d1,d2,d3=st.columns([1,1,2])
            decisao=d1.selectbox('Decisão',['OK_MANUAL','EXCLUIR_MANUAL','REVISAR_MANUAL','MANTER_ORIGINAL'],key=f'match_dec_{y}_{stem}')
            obs=d2.text_input('Observação','Curado no app',key=f'match_obs_{y}_{stem}')
            if d3.button('Salvar decisão de matching',type='primary'):
                ok,msg=apply_matching_decision(root,y,stem,decisao,obs)
                st.success(msg) if ok else st.warning(msg)
                time.sleep(.15)
                st.rerun()
            review_footer(df,'matching_review','status_final')
    with tabs[2]:
        status,dataset,resumo=load_curadoria(root)
        st.subheader('Resumo')
        st_df(resumo,260)
        if not status.empty and {'ano','status_final','usar_matching'}.issubset(status.columns):
            st.subheader('OK por ano')
            ok=status[status['usar_matching'].astype(str).str.upper().isin(['TRUE','1','SIM','YES'])].copy()
            if not ok.empty:
                ok['tipo_ok']=ok['status_final'].astype(str).str.upper().map(lambda s:'OK_MANUAL' if 'MANUAL' in s else 'OK_AUTO')
                st_df(ok.groupby(['ano','tipo_ok'],dropna=False).size().reset_index(name='qtd').pivot(index='ano',columns='tipo_ok',values='qtd').fillna(0).reset_index(),260)
        st.subheader('Status')
        st_df(status,520)
        st.subheader('Dataset final para matching')
        st_df(dataset,420)


def page_genero(root,years):
    st.title('Gênero / padrões')
    tabs=st.tabs(['FairFace','Curadoria','Modelo e exportação','Tabelas'])
    with tabs[0]:
        st.caption('O FairFace usa todas as faces aprovadas no matching (`usar_matching=True`), tanto OK automático quanto OK manual. Use “Continuar pendentes” se parou no meio; “Refazer tudo” sobrescreve as predições existentes.')
        st_df(overview_counts(root,years),240)
        sel=st.multiselect('Anos',years,default=years,key='ff_years')
        yrs=sel if sel and len(sel)!=len(years) else None
        limit=st.number_input('Limite de teste (0 = todos pendentes)',0,100000,0)
        extra=(['--limit',str(int(limit))] if limit else [])
        a,b,c=st.columns(3)
        with a:
            if st.button('Continuar FairFace pendentes',type='primary'):
                run_command_live(build_cmd(root,'fairface',yrs,False,extra),root,'FairFace pendentes')
        with b:
            if st.button('Completar todos os anos pendentes'):
                run_command_live(build_cmd(root,'fairface',None,False,[]),root,'FairFace todos')
        with c:
            if st.button('Refazer FairFace tudo (--force)'):
                run_command_live(build_cmd(root,'fairface',yrs,True,extra),root,'FairFace force')
        st_df(load_genero(root)[0],420)
    with tabs[1]:
        a,b=st.columns(2)
        with a:
            if st.button('Gerar/atualizar gênero curado'):
                run_command_live(build_cmd(root,'curadoria_genero'),root,'Curadoria gênero')
        with b:
            if st.button('Aplicar FairFace de alta confiança'):
                run_command_live(build_cmd(root,'curadoria_genero',extra=['--apply-high']),root,'Alta confiança')
        pred,cur,*_=load_genero(root)
        src=normalize_key_columns(cur if not cur.empty else pred)
        if src.empty:
            st.info('Rode FairFace primeiro.')
        f1,f2,f3,f4=st.columns(4)
        anos_src=sorted(src['ano'].dropna().astype(str).unique()) if 'ano' in src.columns else years
        fano=f1.selectbox('Ano',['TODOS']+anos_src,key='gen_fano')
        flista=f2.selectbox('Lista',['REVISAR','TODAS','DEFINIDAS','INDEFINIDOS/REVISAR','DIVERGENTES'],key='gen_lista')
        fconf=f3.selectbox('Confiança',['TODAS','ALTA','MEDIA','BAIXA','INDEFINIDA','ERRO'],key='gen_conf')
        busca=f4.text_input('Buscar id/iniciais',key='gen_busca')
        df=filter_by_year(src,fano)
        if fconf!='TODAS' and 'confianca_fairface' in df.columns:
            df=df[df['confianca_fairface'].astype(str).str.upper()==fconf].copy()
        if flista=='REVISAR':
            df=df[genero_pending_mask(df)].copy()
        elif flista=='DEFINIDAS' and 'genero_final' in df.columns:
            df=df[df['genero_final'].astype(str).str.upper().isin(['MASCULINO','FEMININO','NAO_USAR'])].copy()
        elif flista=='INDEFINIDOS/REVISAR':
            df=df[genero_pending_mask(df)].copy()
        elif flista=='DIVERGENTES':
            comp=normalize_key_columns(load_genero(root)[4])
            if not comp.empty and 'concordancia_fairface_modelo' in comp.columns:
                keys=set(map(tuple,comp[comp['concordancia_fairface_modelo'].astype(str).str.contains('DIVERGENTE|REVISAR',case=False,regex=True,na=False)][['ano','id_imagem']].astype(str).values.tolist()))
                df=df[df.apply(lambda r:(str(r.get('ano','')),str(r.get('id_imagem',''))) in keys,axis=1)].copy()
        if busca:
            needle=busca.strip().lower()
            df=df[df.astype(str).agg(' '.join,axis=1).str.lower().str.contains(needle,regex=False,na=False)].copy()
        st.caption(f'{len(df)} item(ns). Ao confirmar, o item sai de Revisar.')
        row=review_navigator(df,'genero_review','genero_final',360)
        if row is not None:
            y=str(row.get('ano','')); stem=str(row.get('id_imagem',''))
            a,b=st.columns(2)
            with a:
                show_image(root,y,stem,'lado a lado')
            with b:
                crop=str(row.get('crop_fairface',''))
                if crop and Path(crop).exists():
                    st.image(crop,width='stretch',caption='Crop FairFace')
                st.write(f"FairFace: **{row.get('classe_fairface','')}**")
                st.write(f"M={row.get('prob_masculino_fairface','')} · F={row.get('prob_feminino_fairface','')} · confiança={row.get('confianca_fairface','')}")
            opcoes=['CONFIRMAR_IA','MASCULINO','FEMININO','INDEFINIDO','NAO_USAR']
            classe=str(row.get('classe_fairface','')).upper()
            atual=str(row.get('status_genero_manual','')).upper()
            default='CONFIRMAR_IA' if classe in ['MASCULINO_PROVAVEL','FEMININO_PROVAVEL'] else 'INDEFINIDO'
            if atual in opcoes:
                default=atual
            d1,d2,d3=st.columns([1,1,2])
            decisao=d1.selectbox('Decisão gênero',opcoes,index=opcoes.index(default),key=f'gen_dec_{y}_{stem}')
            obs=d2.text_input('Observação','Curado no app',key=f'gen_obs_{y}_{stem}')
            if d3.button('Salvar gênero desta imagem',type='primary'):
                ok,msg=apply_genero_decision(root,y,stem,decisao,obs)
                st.success(msg) if ok else st.warning(msg)
                time.sleep(.15)
                st.rerun()
            review_footer(df,'genero_review','genero_final')
    with tabs[2]:
        st.subheader('Fluxo do modelo')
        outg=genero_dir(root)
        etapas_modelo=[
            ('1. Features',outg/'features_genero_faces.csv'),
            ('2. Modelo treinado',outg/'modelo_genero_faces.pkl'),
            ('3. Métricas',outg/'metricas_modelo_genero_faces.csv'),
            ('4. Comparação',outg/'comparacao_genero_fairface_modelo.csv'),
            ('5. Exportação',matching_features_dir(root)/'features_matching_faces_com_genero.csv'),
        ]
        st_df(pd.DataFrame([{'etapa':nome,'status':'PRONTO' if path.exists() else 'PENDENTE','arquivo':str(path),'linhas':len(read_csv_safe(path)) if path.suffix=='.csv' and path.exists() else ''} for nome,path in etapas_modelo]),240)
        st.caption('Ordem: gerar features → treinar os modelos selecionados → comparar com FairFace → exportar o dataset final. Durante cada execução, o log abaixo mostra o script e o progresso.')
        with st.expander('Parâmetros do treinamento',expanded=True):
            p1,p2,p3=st.columns(3)
            modelos=p1.multiselect('Modelos para comparar',['logistic_regression','svm_rbf','random_forest'],default=['logistic_regression','svm_rbf','random_forest'])
            metrica=p2.selectbox('Escolher melhor modelo por',['f1_masculino','accuracy','precision_masculino','recall_masculino'])
            test_size=p3.slider('Proporção para teste',.10,.50,.25,.05)
            p4,p5,p6=st.columns(3)
            seed=p4.number_input('Seed aleatória',0,100000,42,1)
            max_iter=p5.number_input('Máx. iterações logística',100,10000,2000,100)
            rf_trees=p6.number_input('Árvores Random Forest',50,2000,300,50)
            p7,p8=st.columns(2)
            svm_c=p7.number_input('SVM C',.01,100.,1.,.1)
            svm_gamma=p8.selectbox('SVM gamma',['scale','auto'])
        train_extra=['--models']+(modelos or ['logistic_regression'])+['--selection-metric',metrica,'--test-size',str(test_size),'--random-state',str(int(seed)),'--logistic-max-iter',str(int(max_iter)),'--rf-trees',str(int(rf_trees)),'--svm-c',str(float(svm_c)),'--svm-gamma',svm_gamma]
        c1,c2,c3,c4=st.columns(4)
        if c1.button('1. Gerar features'):
            run_command_live(build_cmd(root,'features_genero'),root,'1 · Features gênero')
        if c2.button('2. Treinar modelos',type='primary'):
            run_command_live(build_cmd(root,'treinar_genero',extra=train_extra),root,'2 · Treinamento dos modelos')
        if c3.button('3. Comparar resultados'):
            run_command_live(build_cmd(root,'comparar_genero'),root,'3 · Comparação FairFace x modelo')
        if c4.button('4. Exportar final'):
            run_command_live(build_cmd(root,'exportar_genero'),root,'4 · Exportação final')
        if st.button('Rodar fluxo completo',help='Executa as quatro etapas em ordem e para se alguma falhar.'):
            fluxo=[('features_genero',[],'1 · Features gênero'),('treinar_genero',train_extra,'2 · Treinamento dos modelos'),('comparar_genero',[],'3 · Comparação FairFace x modelo'),('exportar_genero',[],'4 · Exportação final')]
            for key,extra,title in fluxo:
                if run_command_live(build_cmd(root,key,extra=extra),root,title)!=0:
                    st.error(f'Fluxo interrompido em: {title}')
                    break
            else:
                st.success('Fluxo completo finalizado.')
        st.subheader('Métricas do modelo')
        st_df(read_csv_safe(genero_dir(root)/'metricas_modelo_genero_faces.csv'),220)
        st.caption('Matriz de confusão: as linhas são os gêneros definidos na curadoria; as colunas mostram o que o modelo previu. A diagonal representa acertos e os outros dois números representam divergências.')
        st_df(read_csv_safe(genero_dir(root)/'matriz_confusao_genero_faces.csv'),220)
        st.subheader('Comparação por imagem')
        render_gender_comparison(root,years)
        st.subheader('Arquivo final')
        final=matching_features_dir(root)/'features_matching_faces_com_genero.csv'
        st.code(str(final))
        st_df(read_csv_safe(final),320)
    with tabs[3]:
        st.subheader('Resumo das definições de gênero')
        resumo_anos=overview_counts(root,years)
        cols=[c for c in ['ano','revisar_genero','genero_alta_confianca','genero_definido_manual'] if c in resumo_anos.columns]
        st_df(resumo_anos[cols] if cols else pd.DataFrame(),260)
        for nome,df in zip(['FairFace','Gênero curado','Features','Predições modelo','Comparação'],load_genero(root)):
            st.subheader(nome)
            st_df(df,300)


def page_imagens(root,years):
    st.title('Imagens')
    ana=load_all_analysis(root,years)
    rows=[]
    for year in years:
        for p in list_images_for_year(root,year):
            rows.append({'ano':str(year),'id_imagem':p.stem})
    base=normalize_key_columns(pd.DataFrame(rows))
    if not ana.empty:
        ana=normalize_key_columns(ana).drop_duplicates(['ano','id_imagem'],keep='last')
        ana=base.merge(ana,on=['ano','id_imagem'],how='left')
    else:
        ana=base
    pred,cur,*_=load_genero(root)
    genero=normalize_key_columns(cur if not cur.empty else pred)
    if not genero.empty:
        campos=['ano','id_imagem','genero_final','origem_genero_final','prob_masculino_fairface','prob_feminino_fairface','confianca_fairface']
        genero=genero[[c for c in campos if c in genero.columns]].drop_duplicates(['ano','id_imagem'],keep='last')
        ana=ana.merge(genero,on=['ano','id_imagem'],how='left')
    if ana.empty:
        st.info('Nenhuma imagem encontrada.')
        return
    ana=normalize_key_columns(ana)
    f1,f2,f3,f4,f5,f6=st.columns(6)
    y=f1.selectbox('Ano',['TODOS']+years,key='img_ano')
    q=f2.selectbox('Qualidade/status',['TODAS','OK','REVISAR','FALHA','EXCLUIR'],key='img_q')
    fgenero=f3.selectbox('Gênero',['TODOS','MASCULINO','FEMININO','INDEFINIDO','SEM DEFINIÇÃO'],key='img_genero')
    mode=f4.selectbox('Visualização',['lado a lado','marcacao','original'],format_func=lambda x:{'marcacao':'com marcação','original':'sem marcação','lado a lado':'lado a lado'}[x],key='img_mode')
    cols_por_linha=f5.selectbox('Por fileira',[3,4,5],index=1,key='img_cols')
    busca=f6.text_input('Buscar id/iniciais',key='img_busca')
    df=filter_by_year(ana,y)
    if q!='TODAS' and 'qualidade_landmarks' in df.columns:
        df=df[df['qualidade_landmarks'].astype(str).str.upper()==q].copy()
    if fgenero!='TODOS':
        genero_final=df.get('genero_final',pd.Series('',index=df.index)).fillna('').astype(str).str.upper()
        df=df[genero_final.eq('') if fgenero=='SEM DEFINIÇÃO' else genero_final.eq(fgenero)].copy()
    if busca:
        needle=busca.strip().lower()
        df=df[df.astype(str).agg(' '.join,axis=1).str.lower().str.contains(needle,regex=False,na=False)].copy()
    if df.empty:
        st.info('Nenhuma imagem para esses filtros.')
        return
    df=df.sort_values(['ano','id_imagem']).reset_index(drop=True)
    st.caption(f'Mostrando {len(df)} imagem(ns) filtrada(s).')
    st_df(df[[c for c in ['ano','id_imagem','qualidade_landmarks','genero_final','origem_genero_final','prob_masculino_fairface','prob_feminino_fairface','confianca_fairface','motivo_qualidade','status_manual','observacao_manual'] if c in df.columns]].head(300),260)
    for start in range(0,len(df),int(cols_por_linha)):
        cols=st.columns(int(cols_por_linha),gap='medium')
        for col,(_,row) in zip(cols,df.iloc[start:start+int(cols_por_linha)].iterrows()):
            with col:
                gallery_card(root,row,mode)


def page_metricas(root,years):
    st.title('Métricas e tabelas')
    ana=load_all_analysis(root,years)
    status,dataset,resumo=load_curadoria(root)
    pred,cur,feat,predm,comp=load_genero(root)
    tabs=st.tabs(['Análise','Matching','Gênero','Legenda'])
    with tabs[0]:
        st_df(ana,560)
    with tabs[1]:
        st_df(status,360)
        st_df(dataset,360)
    with tabs[2]:
        st_df(pred,300)
        st_df(cur,300)
        st_df(feat,300)
        st_df(predm,300)
        st_df(comp,300)
    with tabs[3]:
        st_df(metric_legend_dataframe(),560)


def page_logs(root,years):
    st.title('Logs e arquivos')
    tabs=st.tabs(['Logs','Arquivos'])
    with tabs[0]:
        logdir=resultados_root(root)/'logs'
        logs=sorted(logdir.glob('*.log'),reverse=True) if logdir.exists() else []
        if not logs:
            st.info('Sem logs.')
            return
        p=st.selectbox('Log',logs,format_func=lambda x:x.name)
        st.code(p.read_text(encoding='utf-8',errors='replace')[-70000:],language='text',height=560)
    with tabs[1]:
        st_df(pd.DataFrame([{'ano':y,'images':str(images_dir(root,y)),'previews':str(previews_dir(root,y)),'marcacoes':str(marks_dir(root,y))} for y in years]),260)
        st_df(pd.DataFrame([{'arquivo':'dataset_matching','caminho':str(resultados_root(root)/'curadoria_matching'/'dataset_matching_faces.csv'),'existe':(resultados_root(root)/'curadoria_matching'/'dataset_matching_faces.csv').exists()},{'arquivo':'genero_curado','caminho':str(genero_dir(root)/'genero_faces_curado.csv'),'existe':(genero_dir(root)/'genero_faces_curado.csv').exists()},{'arquivo':'export_final','caminho':str(matching_features_dir(root)/'features_matching_faces_com_genero.csv'),'existe':(matching_features_dir(root)/'features_matching_faces_com_genero.csv').exists()}]),260)


def main():
    root0=guess_root()
    with st.sidebar:
        st.title('Treinamento Faces')
        root=Path(st.text_input('Pasta raiz',value=str(root0))).expanduser().resolve()
        if not root.exists():
            st.error('Pasta raiz não existe.')
            st.stop()
        ensure_base_dirs(root)
        years=discover_years(root)
        st.session_state.root=root
        if years:
            st.success('Anos: '+', '.join(years))
        else:
            st.warning('Nenhum ano encontrado.')
        page=st.radio('Menu',['Painel','Pipeline','Matching','Gênero','Imagens','Métricas','Logs'],index=0)
        py=project_python(root)
        st.caption(f'Python dos scripts: {py}')
        st.code('.venv/bin/python -m streamlit run scripts/app_revisao_faces.py',language='bash')
    if page=='Painel':
        page_painel(root,years)
    elif page=='Pipeline':
        page_pipeline(root,years)
    elif page=='Matching':
        page_matching(root,years)
    elif page=='Gênero':
        page_genero(root,years)
    elif page=='Imagens':
        page_imagens(root,years)
    elif page=='Métricas':
        page_metricas(root,years)
    else:
        page_logs(root,years)


if __name__=='__main__':
    main()
