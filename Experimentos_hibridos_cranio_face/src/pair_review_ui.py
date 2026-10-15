"""Read-only paired AI proposals; independent from accepted training labels."""
import json

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from .common import LAB_ROOT, REPO_ROOT


def display_image(view, points, marked):
    with Image.open(REPO_ROOT/view['path']) as original:
        image = original.convert('RGB').crop(tuple(view['box']))
    w,h=image.size
    image.thumbnail((800,900))
    if marked:
        draw=ImageDraw.Draw(image)
        font=ImageFont.load_default(size=20)
        for row in points.itertuples():
            if pd.isna(row.x_px) or pd.isna(row.y_px):
                continue
            x=(row.x_px-view['box'][0])*image.width/w
            y=(row.y_px-view['box'][1])*image.height/h
            draw.ellipse((x-5,y-5,x+5,y+5),outline='cyan',width=2)
            draw.text((x+7,y-20),str(row.numero),fill='cyan',font=font,stroke_width=2,stroke_fill='black')
    return image


def render_pairs():
    with st.expander('Piloto de treinamento com pseudorrótulos de IA'):
        runs=sorted((LAB_ROOT/'resultados/piloto_localizador_ia').glob('*/concluido.json'),reverse=True)
        if not runs:
            st.info('Nenhum piloto concluído.')
        else:
            run=st.selectbox('Execução do piloto',runs,format_func=lambda p:p.parent.name).parent
            st.warning('Teste técnico craniano, não validação anatômica. O modelo não estima visibilidade; pode propor pontos ocultos. Matching permanece bloqueado.')
            st.dataframe(pd.read_csv(run/'metricas.csv'),hide_index=True)
            st.caption('NME: erro dividido pela diagonal do recorte, comparado aos pseudorrótulos IA. Menor é melhor. Baseline: coordenadas médias apenas do treino.')
            config=json.loads((run/'config.json').read_text())
            st.json(config,expanded=False)
            curve=pd.read_csv(run/'curva_treino.csv')
            st.line_chart(curve.set_index('epoch'))
            predictions=pd.read_csv(run/'predicoes.csv')
            ids=config['validation_ids']+config['train_ids']
            selected=st.selectbox('Imagem do piloto',ids)
            st.image(str(run/(selected+'_comparacao.jpg')),width='stretch')
            st.caption('Ciano: proposta IA disponível. Magenta: previsão do modelo, inclusive onde falta referência. Branco: diferença. Não indica acerto anatômico.')
            st.dataframe(predictions[predictions.id_pessoa.eq(selected)],hide_index=True)
            st.code((run/'execucao.log').read_text())
    with st.expander('Protocolo consolidado e uso acadêmico de IA'):
        protocol = (LAB_ROOT/'docs/protocolo_pares_anatomicos_v1.md').read_text()
        st.markdown(protocol)
        st.download_button('Baixar protocolo', protocol, file_name='protocolo_pares_anatomicos_v1.md')
    paths = sorted((LAB_ROOT/'dataset/pares_visuais').glob('*/runs/*/lote.json'), reverse=True)
    if not paths:
        st.info('Nenhum lote de pares disponivel.')
        return
    folder = st.selectbox('Execução dos pares',paths,format_func=lambda p:p.parent.name).parent
    info = json.loads((folder/'lote.json').read_text())
    summary = pd.read_csv(folder/'resumo.csv')
    st.warning('Propostas visuais de IA, não validadas. Correspondências regionais não exigem coincidência osso-pele. Nenhum ponto liberado para treino.')
    st.caption('Pares vinculados pelos IDs do dataset. Escalas de exibição independentes; não é sobreposição. Lados pressupostos não espelhados.')
    year = st.selectbox('Ano dos pares',['Todos']+sorted(summary.ano.unique().tolist()))
    filtered = summary if year=='Todos' else summary[summary.ano.eq(year)]
    ids=filtered.id_pessoa.tolist()
    key='visual_pair_identity'
    if st.session_state.get(key) not in ids:
        st.session_state[key]=ids[0]
    identity=st.selectbox('Par face e crânio',ids,key=key)
    index=ids.index(identity)
    entry=next(e for e in info['entradas'] if e['id_pessoa']==identity)
    points=pd.read_csv(folder/identity/'pontos.csv')
    cols=st.columns(3)
    cols[0].metric('Pares no lote',len(summary))
    cols[1].metric('Propostas neste par',int(points.estado.eq('PROPOSTA_IA').sum()))
    cols[2].metric('Pontos em dúvida',int(points.estado.eq('DUVIDA').sum()))
    mode=st.radio('Versão',['Com marcações','Original'],horizontal=True)
    for col,kind in zip(st.columns(2),['cranio','face']):
        with col:
            shown=display_image(entry['views'][kind],points[points.modalidade.eq(kind)],mode=='Com marcações')
            st.image(shown,caption=('Crânio' if kind=='cranio' else 'Face')+' · '+identity,width='stretch')
    with st.expander('Ampliar correspondência'):
        number=st.selectbox('Número do ponto',list(range(1,12)))
        for col,kind in zip(st.columns(2),['cranio','face']):
            row=points[(points.modalidade==kind)&(points.numero==number)].iloc[0]
            with col:
                st.write(row.ponto)
                st.caption(row.observacao)
                if pd.isna(row.x_px) or pd.isna(row.y_px):
                    st.warning('Dúvida: sem coordenada proposta.')
                else:
                    im=Image.open(REPO_ROOT/entry['views'][kind]['path']).convert('RGB')
                    x,y=round(row.x_px),round(row.y_px)
                    radius=max(30,round(min(im.size)*.10))
                    draw=ImageDraw.Draw(im)
                    draw.ellipse((x-5,y-5,x+5,y+5),outline='cyan',width=2)
                    st.image(im.crop((max(0,x-radius),max(0,y-radius),min(im.width,x+radius),min(im.height,y+radius))),width='stretch')
    st.dataframe(points[['numero','modalidade','ponto','estado','observacao']],hide_index=True)
    def move(offset):
        st.session_state[key]=ids[index+offset]
    left,right=st.columns(2)
    left.button('Anterior',icon=':material/arrow_back:',disabled=index==0,on_click=move,args=(-1,))
    right.button('Próximo',icon=':material/arrow_forward:',disabled=index==len(ids)-1,on_click=move,args=(1,))
    st.download_button('Baixar coordenadas do par',points.to_csv(index=False).encode('utf-8-sig'),file_name=identity+'_propostas.csv')
    with st.expander('Resumo dos 10 pares'):
        st.dataframe(summary,hide_index=True)
