"""Image clicks edit a draft in original pixels, never an approved reference."""
import base64
import copy
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from .core import ARTIFACTS, LEGACY, POINT_NAMES, digest, letterbox, load, overlay, regions_for, save, sha

click_image = components.declare_component('lab_point_click', path=str(LEGACY/'src/click_image'))


def canvas(im, points, region, selected, show_regions=False, focus=None):
    h,w=im.shape[:2]
    left=top=0
    right,bottom=w,h
    if region and 'box' in region:
        x1,y1,x2,y2=region['box']
        side=max(x2-x1,y2-y1,16)*1.6
        cx,cy=(x1+x2)/2,(y1+y2)/2
        left,top=max(0,int(cx-side/2)),max(0,int(cy-side/2))
        right,bottom=min(w,int(cx+side/2)+1),min(h,int(cy+side/2)+1)
    visible=[p for p in points if p.get('visibility') not in ('OCULTO','CORTADO','INDETERMINADO')]
    boxes=regions_for(visible,w,h) if show_regions else {}
    marked=overlay(im,[p for p in visible if p['numero']!=selected],boxes)
    if focus in boxes and 'box' in boxes[focus]:
        x1,y1,x2,y2=map(round,boxes[focus]['box'])
        cv2.rectangle(marked,(x1,y1),(x2,y2),(0,210,255),max(1,round(min(w,h)/500)))
    marked=overlay(marked,[p for p in points if p['numero']==selected and p.get('visibility') not in ('OCULTO','CORTADO','INDETERMINADO')],point_color=(0,255,80))
    shown,matrix=letterbox(marked[top:bottom,left:right])
    matrix[:,2]-=matrix[:,:2] @ [left,top]
    return shown,matrix,(left,top,right,bottom)


def map_click(u,v,matrix,bounds):
    if not all(math.isfinite(x) for x in (u,v)) or not (0<=u<=1 and 0<=v<=1):
        return None
    xy=cv2.invertAffineTransform(matrix) @ np.array([u*600,v*600,1])
    left,top,right,bottom=bounds
    if not (left<=xy[0]<=right-1 and top<=xy[1]<=bottom-1):
        return None
    return float(xy[0]),float(xy[1])


def editor(selected, row, original, initial):
    token=digest([str(selected),sha(selected)])[:20]
    draft_key='point_draft_'+token
    revision=draft_key+'_revision'
    draft_path=ARTIFACTS/'point_drafts'/token/'draft.json'
    review_hash=initial.attrs.get('review_hash')
    if st.session_state.get(token+'_base_review_hash')!=review_hash:
        st.session_state.pop(draft_key,None)
    st.session_state[token+'_base_review_hash']=review_hash
    if draft_key not in st.session_state:
        saved=load(draft_path) if draft_path.exists() else None
        # A completed review always takes priority over an older draft.
        points=saved['points'] if saved and saved['image_hash']==row['hash'] and saved.get('base_review_hash')==initial.attrs.get('review_hash') else initial.to_dict('records')
        st.session_state[draft_key]=points
        st.session_state[revision]=0
    points=copy.deepcopy(st.session_state[draft_key])
    active=st.toggle('Modo revisão por clique', value=True, key=token+'_mode')
    number=st.selectbox('Ponto para posicionar',list(range(1,12)),format_func=lambda n:f'{n} · {POINT_NAMES[n-1]}',key=token+'_point')
    with st.expander('Coordenadas e visibilidade'):
        frame=pd.DataFrame(points)
        frame['ponto']=[POINT_NAMES[int(n)-1] for n in frame.numero]
        changed=st.data_editor(frame[['numero','ponto','x','y','visibility']],hide_index=True,
            disabled=['numero','ponto'],key=token+'_grid_'+str(st.session_state[revision]),
            column_config={'visibility':st.column_config.SelectboxColumn('Visibilidade',options=['NAO_AVALIADO','VISIVEL','OCULTO','CORTADO','INDETERMINADO']),
                'x':st.column_config.NumberColumn('X (px)',min_value=0,max_value=row['width']-1),
                'y':st.column_config.NumberColumn('Y (px)',min_value=0,max_value=row['height']-1)})
        points=[dict(numero=int(p.numero),x=float(p.x) if pd.notna(p.x) else None,y=float(p.y) if pd.notna(p.y) else None,visibility=p.visibility) for p in changed.itertuples()]
    st.session_state[draft_key]=copy.deepcopy(points)
    region=st.selectbox('Área da revisão',['Imagem inteira','nariz','olho_direito','olho_esquerdo','central','mandibula_queixo'],key=token+'_region')
    show_regions=st.checkbox('Mostrar quadrados das regiões',value=True,key=token+'_boxes')
    # Freeze framing per region: repositioning a point must not move the target.
    viewport_key=token+'_viewport_'+region
    if viewport_key not in st.session_state:
        box=regions_for(row['points'],row['width'],row['height']).get(region)
        if box and 'box' not in box and row.get('parent_proposal'):
            parent=load(row['parent_proposal'])
            box=regions_for(parent['points'],row['width'],row['height']).get(region)
        st.session_state[viewport_key]=box
    shown,matrix,bounds=canvas(original,points,st.session_state[viewport_key],number,show_regions,region)
    chosen=points[number-1]
    if chosen.get('x') is not None and chosen.get('y') is not None:
        left,top,right,bottom=bounds
        if not (left<=chosen['x']<right and top<=chosen['y']<bottom):
            st.info(f'O ponto {number} tem coordenadas, mas está fora deste zoom.')
            def full_view():
                st.session_state[token+'_region']='Imagem inteira'
            st.button('Ver ponto na imagem inteira',key=token+'_full',on_click=full_view)
        elif chosen['visibility'] in ('OCULTO','CORTADO','INDETERMINADO'):
            st.info(f'O ponto {number} está oculto pelo estado de visibilidade selecionado.')
    if show_regions and region!='Imagem inteira':
        current_boxes=regions_for([p for p in points if p['visibility'] not in ('OCULTO','CORTADO','INDETERMINADO')],row['width'],row['height'])
        if 'box' not in current_boxes[region]:
            st.caption('Quadrado desta região indisponível: faltam pontos. O zoom continua disponível para posicioná-los.')
    if active:
        context=digest([token,number,region,st.session_state[revision],matrix.tolist()])
        encoded=cv2.imencode('.png',shown)[1]
        event=click_image(image='data:image/png;base64,'+base64.b64encode(encoded).decode(),
                          context=context,point=f'{number} · {POINT_NAMES[number-1]}',key=token+'_canvas',default=None)
        if event and event.get('event_id')!=st.session_state.get(token+'_last'):
            st.session_state[token+'_last']=event.get('event_id')
            if event.get('context')==context:
                xy=map_click(event['u'],event['v'],matrix,bounds)
                if xy:
                    points[number-1].update(x=xy[0],y=xy[1],visibility='VISIVEL')
                    st.session_state[draft_key]=points
                    st.session_state[revision]+=1
                    st.rerun()
    else:
        st.image(shown,channels='BGR',width='stretch')
    p=points[number-1]
    x_label=f'{p["x"]:.1f}' if p['x'] is not None else 'ausente'
    y_label=f'{p["y"]:.1f}' if p['y'] is not None else 'ausente'
    st.caption(f'Ponto {number} · {p["visibility"]} · X: {x_label} · Y: {y_label}')
    state=st.selectbox('Visibilidade do ponto selecionado',['NAO_AVALIADO','VISIVEL','OCULTO','CORTADO','INDETERMINADO'],
                       index=['NAO_AVALIADO','VISIVEL','OCULTO','CORTADO','INDETERMINADO'].index(p['visibility']),key=token+'_state_'+str(number)+'_'+str(st.session_state[revision]))
    if state!=p['visibility']:
        p['visibility']=state
        if state in ('OCULTO','CORTADO','INDETERMINADO'):
            p.update(x=None,y=None)
        st.session_state[draft_key]=points
        st.session_state[revision]+=1
        st.rerun()
    if st.button('Limpar ponto selecionado',key=token+'_clear'):
        p.update(x=None,y=None,visibility='NAO_AVALIADO')
        st.session_state[draft_key]=points
        st.session_state[revision]+=1
        st.rerun()
    if st.button('Salvar rascunho',key=token+'_save'):
        if sha(row['path'])!=row['hash']:
            st.error('Imagem original alterada; recarregue a revisão.')
        else:
            save(draft_path,dict(proposal=str(selected),proposal_hash=sha(selected),image_hash=row['hash'],
                                base_review_hash=initial.attrs.get('review_hash'),points=points))
            st.success('Posições salvas como rascunho. Revisão completa ainda não incorporada ao treinamento.')
    return points
