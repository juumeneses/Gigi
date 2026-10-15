"""Regional audit UI; never writes legacy labels or starts training."""
import html
import json
import queue
import subprocess
import sys
import threading
import time

import cv2
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from .region_pilot import (RUNS, LAB_ROOT, REPO_ROOT, STAGES, FACE_GROUPS, VISIBILITY,
                           key, image_input, render_overlay, save_review, digest, verified, letterbox)


def region_outline(image, bounds):
    result=image.copy()
    if bounds is not None:
        h,w=image.shape[:2]
        left,top,right,bottom=bounds
        cv2.rectangle(result,(max(0,round(left)),max(0,round(top))),
                      (min(w-1,round(right)),min(h-1,round(bottom))),
                      (0,220,255),max(1,round(min(h,w)/600)))
    return result


def free_zoom_bounds(width, height, center_x, center_y, zoom):
    """Viewport in original pixels, independent of any detection or proposal."""
    crop_w,crop_h=width/zoom,height/zoom
    left=max(0,min(width-crop_w,width*center_x/100-crop_w/2))
    top=max(0,min(height-crop_h,height*center_y/100-crop_h/2))
    return [left,top,left+crop_w,top+crop_h]


def regional_zoom_defaults(width, height, region, extraction, regions, modality):
    if region=='Imagem inteira': return 1.,50,50
    box=regions.get('regions',{}).get(region,{})
    if 'crop_bounds' in box:
        left,top,right,bottom=viewing_bounds(box['crop_bounds'],region,extraction,modality)
        zoom=max(1.,min(8.,width/max((right-left)*1.2,1),height/max((bottom-top)*1.2,1)))
        import math
        return math.floor(zoom*10)/10,int(max(0,min(100,(left+right)/2/width*100))),int(max(0,min(100,(top+bottom)/2/height*100)))
    # A navigational starting view, never a detected region or inferred landmark.
    return {'olho_direito':(3.,35,45),'olho_esquerdo':(3.,65,45),
            'nariz':(3.,50,55),'central':(2.,50,50),
            'mandibula_queixo':(2.,50,75)}[region]


def viewing_bounds(bounds, region, extraction, modality):
    """Display-only context, not an anatomical boundary or a new landmark."""
    left,top,right,bottom=map(float,bounds)
    if region.startswith('olho'):
        cy=(top+bottom)/2
        side=max(right-left,bottom-top)
        top,bottom=cy-side/2,cy+side/2
    if modality=='cranio':
        points={p['numero']:(p['x'],p['y']) for p in extraction['proposals']
                if p['x'] is not None and p['y'] is not None}
        if region=='nariz':
            orbit=[points[i][1] for i in [5,6,10,11] if i in points]
            if orbit: top=min(top,min(orbit))
            margin=max(4,(right-left)*.15)
            left-=margin; right+=margin; top-=margin; bottom+=margin
        elif region=='mandibula_queixo' and 2 in points and 7 in points:
            nasal,chin=points[2][1],points[7][1]
            if chin>nasal:
                top=nasal+.2*(chin-nasal)
                bottom=chin+.12*(chin-nasal)
    elif region=='mandibula_queixo':
        points={p['idx']:(p['x'],p['y']) for p in extraction['points']}
        if all(i in points for i in [172,397,152]):
            ys=[points[i][1] for i in [172,397,152]]
            margin=max(4,(max(ys)-min(ys))*.15)
            top=min(ys)-margin; bottom=max(ys)+margin
    elif region=='nariz':
        margin=max(4,(bottom-top)*.15)
        top-=margin; bottom+=margin
    return [left,top,right,bottom]


def display_frame(image, bounds=None, size=600, square_context=True):
    """Square viewing context only; never changes analytical boxes or saved crops."""
    if bounds is not None:
        import math
        left,top,right,bottom=bounds
        cx,cy=(left+right)/2,(top+bottom)/2
        side=max(right-left,bottom-top,8)
        h,w=image.shape[:2]
        if square_context:
            left,right=cx-side/2,cx+side/2
            top,bottom=cy-side/2,cy+side/2
        left=max(0,math.floor(left)); right=min(w,math.ceil(right))
        top=max(0,math.floor(top)); bottom=min(h,math.ceil(bottom))
        if right>left and bottom>top:
            image=image[top:bottom,left:right]
    return letterbox(image,size=size)[0]


def execute(command):
    st.code(' '.join(map(str,command)),language='bash')
    messages=queue.Queue(); lines=[]; started=time.monotonic()
    process=subprocess.Popen(list(map(str,command)),cwd=REPO_ROOT,stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,text=True,bufsize=1)
    def reader():
        for line in process.stdout: messages.put(line)
        process.stdout.close()
    worker=threading.Thread(target=reader,daemon=True); worker.start()
    status=st.empty(); output=st.empty()
    while process.poll() is None or worker.is_alive() or not messages.empty():
        while not messages.empty(): lines.append(messages.get_nowait())
        status.info(f'Em execução · {time.monotonic()-started:.0f}s · PID {process.pid}')
        with output.container():
            components.html('<pre id="log" style="height:260px;overflow:auto;white-space:pre-wrap">'+
                html.escape(''.join(lines[-100:]))+'</pre><script>let e=document.getElementById("log");e.scrollTop=e.scrollHeight;</script>',height=280)
        time.sleep(.3)
    worker.join(); code=process.wait()
    (status.success if code==0 else status.error)(f'Encerrado · código {code} · {time.monotonic()-started:.0f}s')
    st.code(''.join(lines[-35:]),language='text')
    return code


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def render_pending_originals(run, checks):
    pending = checks[checks.status.ne('OK_TECNICO')]
    options = list(pending[['id', 'modalidade']].drop_duplicates().itertuples(index=False, name=None))
    if not options:
        st.info('Nenhuma pendência técnica nesta verificação.')
        return
    selected = st.selectbox('Original com pendência', options,
                            format_func=lambda v: f'{v[0]} · {v[1]}', key='pending_original_' + str(run))
    item = next((i for i in load(run/'entradas.json')
                 if (i['id'], i['modalidade']) == selected), None)
    if item is None:
        st.error('Arquivo não encontrado no manifesto desta execução.')
        return
    records = pending[pending.id.eq(selected[0]) & pending.modalidade.eq(selected[1])]
    st.dataframe(records, hide_index=True)
    original, status = image_input(REPO_ROOT/item['path'])
    if original is None:
        st.error(f'Original indisponível: {status}')
        return
    st.caption('Sem marcações ou inferência de visibilidade. Ponto ausente não significa estrutura oculta. A inspeção não altera arquivos ou resultados.')
    token = str(run) + '_' + '_'.join(selected)
    zoom = st.slider('Ampliação do original pendente', 1., 8., 1., .1, key='pending_zoom_' + token)
    x = st.slider('Centro horizontal do original (%)', 0, 100, 50, key='pending_x_' + token)
    y = st.slider('Centro vertical do original (%)', 0, 100, 50, key='pending_y_' + token)
    h, w = original.shape[:2]
    bounds = free_zoom_bounds(w, h, x, y, zoom)
    st.image(display_frame(original, bounds, square_context=False), channels='BGR',
             width='stretch', caption=f'{selected[0]} · {selected[1]} · original sem alterações')
    st.caption('Para registrar visibilidade, use Revisão cega na mesma execução, modalidade e região. Não há aprovação automática neste diagnóstico.')


def render_pilot():
    st.subheader('Piloto de regiões')
    runs=sorted([p for p in RUNS.glob('*') if (p/'entradas.json').exists()],reverse=True)
    run=st.selectbox('Execução',runs,index=0,format_func=lambda p:p.name) if runs else None
    with st.expander('Executar auditoria'):
        stage=st.selectbox('Etapa',['all']+STAGES)
        cols=st.columns(3)
        action=None
        if cols[0].button('Nova execução'): action='new'
        if cols[1].button('Retomar',disabled=run is None): action='resume'
        if cols[2].button('Refazer em nova execução',disabled=run is None): action='force'
        if action:
            command=[sys.executable,'-u',LAB_ROOT/'scripts/piloto_regioes.py','--stage',stage]
            if action!='new': command+=['--run-dir',run,'--'+action]
            execute(command)
            st.button('Atualizar execuções')
    if run is None:
        st.info('Nenhuma execução do piloto disponível.'); return
    with st.expander('Diagnóstico técnico de integridade'):
        audit_key = 'technical_audit_' + str(run)
        if st.button('Verificar pontos e recortes', key='technical_integrity_audit'):
            from .technical_audit import audit_run
            with st.spinner('Verificando originais, coordenadas e arquivos de saída...'):
                st.session_state[audit_key] = pd.DataFrame(audit_run(run))
        if audit_key in st.session_state:
            checks = st.session_state[audit_key]
            st.caption('Verificação técnica apenas. Não avalia correspondência entre pessoas nem precisão anatômica. Visibilidade continua dependendo de revisão.')
            st.caption('Resultado da última verificação. Execute novamente após alterar arquivos.')
            only_pending = st.checkbox('Somente pendências', value=True, key='technical_pending_only')
            shown = checks[checks.status.ne('OK_TECNICO')] if only_pending else checks
            st.dataframe(shown, hide_index=True)
            st.write(f'{len(checks)} verificações; {checks.status.eq("ERRO").sum()} erros; {checks.status.eq("ATENCAO").sum()} alertas.')
            st.download_button('Baixar diagnóstico técnico', checks.to_csv(index=False).encode(), file_name='diagnostico_tecnico.csv')
            render_pending_originals(run, checks)
    progress=load(run/'progresso.json')
    if progress: st.json(progress,expanded=False)
    if (run/'resumo.csv').exists():
        with st.expander('Resumo dos 11 pares'):
            st.dataframe(pd.read_csv(run/'resumo.csv'),hide_index=True)
    with st.expander('Log desta execução'):
        if (run/'execucao.log').exists(): st.code('\n'.join((run/'execucao.log').read_text().splitlines()[-100:]),language='text')
    mode=st.radio('Visualização',['Auditoria','Revisão cega','Comparar sessões'],horizontal=True)
    items=load(run/'entradas.json')
    if mode=='Comparar sessões':
        records=[load(p) for p in (run/'revisoes').glob('*.json')]
        if not records: st.info('Sem revisões.'); return
        frame=pd.DataFrame(records).sort_values('data').drop_duplicates(
            ['id','modalidade','regiao','autor','sessao','output_hash'],keep='last')
        table=frame.pivot(index=['id','modalidade','regiao','autor','output_hash'],columns='sessao',values='estado')
        if 'A' not in table or 'B' not in table: st.info('Complete as duas sessões antes da comparação.'); return
        paired=table.dropna(subset=['A','B'])
        st.metric('Regiões revistas nas duas sessões',len(paired))
        if len(paired): st.metric('Concordância descritiva',f'{paired.A.eq(paired.B).mean():.1%}')
        st.caption('Concordância entre sessões não é validação anatômica especializada.')
        st.dataframe(paired.reset_index(),hide_index=True); return
    blind=mode=='Revisão cega'
    session=st.selectbox('Sessão',['A','B']) if blind else 'A'
    modality=st.selectbox('Modalidade',['face','cranio'])
    items=[i for i in items if i['modalidade']==modality]
    if not blind:
        year=st.selectbox('Ano',['Todos']+sorted({i['ano'] for i in items}))
        items=[i for i in items if year=='Todos' or i['ano']==year]
    else:
        items.sort(key=lambda i:digest([session,key(i)]))
    context=f'pilot_{run.name}_{mode}_{modality}_{session}'
    selected=st.selectbox('Caso' if blind else 'Par',list(range(len(items))),
        format_func=lambda n:f'Caso {n+1}' if blind else items[n]['id'],key=context)
    item=items[selected]; folder=run/key(item)
    ex=load(folder/'extracao/data.json'); reg=load(folder/'regioes/data.json')
    if not ex or not reg: st.warning('Extração ou regiões ainda não disponíveis.'); return
    if not all(verified(folder/stage/'data.json') for stage in ['extracao','regioes']):
        st.error('Saída inconsistente: retome ou refaça esta execução antes de revisar.'); return
    region=st.selectbox('Região',['Imagem inteira']+list(FACE_GROUPS))
    layers=[] if blind else st.multiselect('Camadas',['Malha completa','Seleção regional','Propostas anteriores','Contornos','Boxes'],default=['Seleção regional' if modality=='face' else 'Propostas anteriores','Boxes'])
    original,status=image_input(REPO_ROOT/item['path'])
    if original is None: st.error(status); return
    active_layers=[layer for layer in layers if layer!='Boxes' or region=='Imagem inteira']
    overlay=render_overlay(original,ex,reg,active_layers)
    bounds=None
    free_zoom=st.checkbox('Zoom independente dos pontos',value=True)
    if free_zoom:
        h,w=original.shape[:2]
        zoom_key=f'{context}_{selected}_{region}_regional_v3'
        z0,x0,y0=regional_zoom_defaults(w,h,region,ex,reg,modality)
        zoom=st.slider('Ampliação',1.,8.,z0,.1,key=zoom_key+'_zoom')
        x=st.slider('Centro horizontal (%)',0,100,x0,key=zoom_key+'_x')
        y=st.slider('Centro vertical (%)',0,100,y0,key=zoom_key+'_y')
        bounds=free_zoom_bounds(w,h,x,y,zoom)
        if region!='Imagem inteira' and 'crop_bounds' not in reg.get('regions',{}).get(region,{}):
            st.warning('Região sem box disponível: enquadramento inicial aproximado. Ajuste a posição para inspecionar; nenhum ponto foi inferido.')
        st.caption('Janela de visualização apenas: não cria pontos nem torna uma região disponível para análise.')
    elif region!='Imagem inteira':
        box=reg.get('regions',{}).get(region,{})
        if 'crop_bounds' in box:
            bounds=viewing_bounds(box['crop_bounds'],region,ex,modality)
        else: st.warning('REGIAO_INDISPONIVEL: sem pontos suficientes; exibindo original completo.')
    square_context=not free_zoom and region not in ['mandibula_queixo','nariz']
    if not blind and region!='Imagem inteira':
        show_outline=st.checkbox('Destacar retângulo da região',value=True)
        reference=reg.get('regions',{}).get(region,{}).get('crop_bounds')
        if show_outline and reference is not None:
            reference=viewing_bounds(reference,region,ex,modality)
            overlay=region_outline(overlay,reference)
            st.caption('Amarelo: enquadramento visual da região, ajustado a partir das referências disponíveis. Não é segmentação anatômica nem altera a box salva.')
        elif show_outline:
            st.info('Sem box de referência nesta região. O zoom continua disponível, mas nenhum limite anatômico foi inventado.')
    elif blind:
        st.caption('Para visualizar pontos e retângulos, selecione Auditoria. A revisão cega mantém essas referências ocultas.')
    original=display_frame(original,bounds,square_context=square_context)
    overlay=display_frame(overlay,bounds,square_context=square_context)
    if blind:
        st.image(original,channels='BGR',caption='Imagem sem pontos ou revisões anteriores',width=600)
        if bounds: st.caption('Contexto ampliado para visualização; a box analítica salva não foi alterada.')
    else:
        a,b=st.columns(2)
        a.image(original,channels='BGR',caption='Original',width='stretch')
        b.image(overlay,channels='BGR',caption='Saída técnica, não aprovação anatômica',width='stretch')
        description=f"{len(ex['points'])} pontos detectados" if modality=='face' else 'Regiões assistidas por propostas anteriores; sem novo detector craniano'
        st.write(f"{ex['status']} · {ex.get('reason','')} · {description}")
        with st.expander('Máscara e recortes salvos'):
            if (folder/'extracao/mask.png').exists(): st.image(str(folder/'extracao/mask.png'),caption='Primeiro plano candidato, não segmentação anatômica validada',width=400)
            crops=[(name,b) for name,b in reg.get('regions',{}).items() if b.get('arquivo')]
            columns=st.columns(2)
            for n,(name,b) in enumerate(crops): columns[n%2].image(str(folder/'regioes'/b['arquivo']),caption=name,width='stretch')
        st.dataframe(pd.DataFrame(reg.get('regions',{})).T)
        if ex.get('discordancia'):
            st.write('Discordância entre métodos, não erro anatômico')
            st.dataframe(ex['discordancia'],hide_index=True)
        stability=load(folder/'estabilidade/data.json')
        if stability:
            st.dataframe([{k:v for k,v in row.items() if k not in ['points_restored','matrix','inverse']} for row in stability['rows']],hide_index=True)
        with st.expander('Proveniência e parâmetros'): st.json(load(run/'config.json')); st.json(item)
    if blind and region!='Imagem inteira':
        with st.form('regional_review'):
            author=st.text_input('Autor')
            state=st.selectbox('Visibilidade',VISIBILITY)
            reason=st.text_area('Motivo')
            flags=st.multiselect('Alertas',['corte','suporte incluído','artefato','oclusão','barba'])
            if st.form_submit_button('Salvar revisão'):
                try:
                    save_review(run,item,region,session,author,state,reason,flags)
                    st.success('Revisão salva, sem aprovação anatômica.')
                except ValueError as e: st.error(str(e))
    left,right=st.columns(2)
    def navigate(delta): st.session_state[context]=selected+delta
    left.button('Anterior',disabled=selected==0,on_click=navigate,args=(-1,))
    right.button('Próximo',disabled=selected==len(items)-1,on_click=navigate,args=(1,))
    if not blind:
        for filename in ['relatorio.html','resumo.csv','pontos.csv','boxes.csv','estabilidade.csv','discordancia.csv']:
            if (run/filename).exists(): st.download_button('Baixar '+filename,(run/filename).read_bytes(),file_name=filename)
