"""Descriptive pilot retrieval. No fitting and no anatomical gate changes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .common import LAB_ROOT, REPO_ROOT, sha256_file
from .region_pilot import RUNS, FACE_EQUIVALENTS, key, image_input, render_overlay, verified
from .region_pilot_ui import display_frame


def features(ex, modality):
    points=({FACE_EQUIVALENTS[n]:n for n in FACE_EQUIVALENTS} if modality=='face' else {})
    if modality=='face':
        xy={points[p['idx']]:np.array([p['x'],p['y']]) for p in ex['points'] if p['idx'] in points}
    else:
        xy={p['numero']:np.array([p['x'],p['y']]) for p in ex['proposals'] if p['x'] is not None and p['y'] is not None}
    def distance(a,b): return float(np.linalg.norm(xy[a]-xy[b])) if a in xy and b in xy else np.nan
    scale=distance(5,6)
    if not np.isfinite(scale) or scale<=0: return {}
    return {name:distance(a,b)/scale for name,a,b in [
        ('largura_nasal_relativa',3,4),('separacao_medial_relativa',10,11),
        ('base_nasal_queixo_relativa',2,7),('largura_lateral_relativa',8,9)]}


def geometry(items, extracts):
    ids=sorted({i['id'] for i in items}); rows=[]
    for face in ids:
        f=features(extracts[(face,'face')],'face')
        for skull in ids:
            s=features(extracts[(skull,'cranio')],'cranio')
            terms={k:abs(f[k]-s[k]) for k in f.keys() & s.keys() if np.isfinite(f[k]) and np.isfinite(s[k])}
            distance=float(np.mean(list(terms.values()))) if len(terms)>=2 else np.nan
            rows.append(dict(face_id=face,cranio_id=skull,correto=face==skull,
                             score=1/(1+distance) if np.isfinite(distance) else np.nan,
                             distancia=distance,medidas=len(terms),**terms))
    return ranked(pd.DataFrame(rows))


def ranked(frame):
    frame=frame.sort_values(['face_id','score','cranio_id'],ascending=[True,False,True]).copy()
    # Worst rank within a tie avoids rewarding the known ID's sorting position.
    frame['rank']=frame.groupby('face_id').score.rank(method='max',ascending=False)
    return frame


def neural(ids):
    import joblib
    import torch
    from scripts.treinar_two_tower_torch import ProjectionTwoTower, image_tensor
    from torchvision.models import resnet18
    torch.set_num_threads(2)
    model_path=LAB_ROOT/'modelos/two_tower_resnet18_pca.pt'
    checkpoint=torch.load(model_path,map_location='cpu',weights_only=True)
    state=checkpoint['model']
    model=ProjectionTwoTower(checkpoint['face_dim'],state['face.3.weight'].shape[0])
    model.load_state_dict(state); model.eval()
    columns=checkpoint['pca_columns']
    frame=pd.read_csv(REPO_ROOT/'Treinamento_faces/resultados/features_matching_faces/features_matching_faces_final.csv')
    frame=frame.set_index('id_pessoa')
    if frame.index.duplicated().any(): raise ValueError('Features faciais com IDs duplicados')
    scaler=joblib.load(LAB_ROOT/'modelos/scaler_face_pca.joblib')
    paired=pd.read_csv(LAB_ROOT/'dataset/manifestos/dataset_pareado_matching.csv').set_index('id_pessoa')
    backbone=resnet18(weights=None)
    backbone.load_state_dict(torch.load(Path.home()/'.cache/torch/hub/checkpoints/resnet18-f37072fd.pth',map_location='cpu',weights_only=True))
    backbone.fc=torch.nn.Identity(); backbone.eval()
    tensors=[]
    for identity in ids:
        row=paired.loc[identity]; path=REPO_ROOT/row['imagem_cf']
        if not path.exists():
            alternatives=(REPO_ROOT/'Treinamento_cranios/cranios_normalizados_variantes').glob(f'*/{int(row["ano"])}/CF/{path.name}')
            path=next((p for p in alternatives if sha256_file(p)==row['imagem_cf_sha256']),path)
        if sha256_file(path)!=row['imagem_cf_sha256']:
            raise ValueError('Imagem craniana mudou desde o manifesto: '+identity)
        tensors.append(image_tensor(path))
    with torch.no_grad(): skulls=dict(zip(ids,backbone(torch.stack(tensors)).numpy()))
    missing=[i for i in ids if i not in frame.index or i not in skulls]
    if missing: raise ValueError('Entradas ausentes: '+', '.join(missing))
    raw=frame.loc[ids,columns]
    if raw.isna().any().any(): raise ValueError('Features com NaN; nao imputadas no piloto')
    # Apply the stored training mean/scale without refitting or version-dependent transform.
    if hasattr(scaler,'feature_names_in_') and list(scaler.feature_names_in_)!=columns:
        raise ValueError('Ordem das features incompatível com o scaler')
    face=torch.tensor(((raw.to_numpy()-scaler.mean_)/scaler.scale_).astype(np.float32))
    skull=torch.tensor(np.asarray([skulls[i] for i in ids]),dtype=torch.float32)
    with torch.no_grad():
        s,f=model.embeddings(skull,face); scores=(f@s.T).numpy()
    if not np.isfinite(scores).all(): raise ValueError('Score neural invalido')
    seen=pd.read_csv(LAB_ROOT/'resultados/matching/predicoes_two_tower.csv')
    train=set(seen.loc[seen.split.eq('treino'),'face_id'].astype(str)) | set(seen.loc[seen.split.eq('treino'),'cranio_id'].astype(str))
    rows=[dict(face_id=f,cranio_id=s,correto=f==s,score=float(scores[i,j]),
               face_vista_no_treino=f in train,cranio_visto_no_treino=s in train)
          for i,f in enumerate(ids) for j,s in enumerate(ids)]
    return ranked(pd.DataFrame(rows)),sha256_file(model_path)


def picture(item,ex,regions,marked=True):
    im,status=image_input(REPO_ROOT/item['path'])
    if im is None: st.error(status); return
    if marked: im=render_overlay(im,ex,regions,['Seleção regional','Propostas anteriores'])
    st.image(display_frame(im),channels='BGR',width='stretch',caption=item['id']+' · '+item['modalidade'])


def render_matching_pilot():
    st.subheader('Imagens teste piloto')
    paths=sorted([p.parent for p in RUNS.glob('*/relatorio.json') if verified(p)],reverse=True)
    if not paths: st.info('Conclua uma auditoria regional primeiro.'); return
    run=st.selectbox('Execução do piloto',paths,format_func=lambda p:p.name)
    items=json.loads((run/'entradas.json').read_text())
    extracts={}; regions={}; byid={}
    for item in items:
        identity=(item['id'],item['modalidade']); folder=run/key(item)
        if not verified(folder/'extracao/data.json') or not verified(folder/'regioes/data.json'):
            st.error('Execução incompleta ou corrompida'); return
        extracts[identity]=json.loads((folder/'extracao/data.json').read_text())
        regions[identity]=json.loads((folder/'regioes/data.json').read_text())
        byid[identity]=item
    ids=sorted({i['id'] for i in items})
    view=st.radio('Etapa do piloto',['Faces marcadas','Crânios marcados','Comparação dos pares','Matching'],horizontal=True)
    st.warning('Exploratório: pseudorrótulos e conjunto de desenvolvimento, não avaliação independente. Nenhum treinamento novo.')
    if view in ['Faces marcadas','Crânios marcados']:
        modality='face' if view=='Faces marcadas' else 'cranio'
        marked=st.checkbox('Mostrar marcações',True)
        columns=st.columns(3)
        for n,identity in enumerate(ids):
            with columns[n%3]: picture(byid[(identity,modality)],extracts[(identity,modality)],regions[(identity,modality)],marked)
        return
    query=st.selectbox('Face de referência',ids)
    if view=='Comparação dos pares':
        a,b=st.columns(2)
        with a: picture(byid[(query,'face')],extracts[(query,'face')],regions[(query,'face')])
        with b: picture(byid[(query,'cranio')],extracts[(query,'cranio')],regions[(query,'cranio')])
        values=pd.DataFrame({'face':features(extracts[(query,'face')],'face'),'crânio':features(extracts[(query,'cranio')],'cranio')})
        values['diferença absoluta']=(values.face-values['crânio']).abs()
        st.dataframe(values)
        st.info('Razões em pixels, divididas pela distância entre referências laterais dos olhos/órbitas de cada modalidade. Essas referências não são equivalentes anatômicos exatos. Sem estimativa de tecido mole em milímetros.')
    else:
        method=st.selectbox('Método',['Geometria descritiva','Duas torres existente'])
        if method=='Geometria descritiva':
            matrix=geometry(items,extracts)
            st.caption('Média das diferenças absolutas entre razões disponíveis; mínimo de duas medidas. Score = 1/(1+distância), não probabilidade. Candidatos com coberturas distintas exigem cautela.')
        else:
            st.info('Modelo legado: ResNet18 craniano + features PCA faciais, não duas torres RGB. Recalcula embeddings das imagens normalizadas do manifesto; não usa os novos pontos. Origem do ajuste da PCA ainda não auditada. Scaler legado aplicado pela média/escala salvas, sem reajuste.')
            try:
                matrix,checkpoint_hash=neural(ids)
                st.caption('Checkpoint SHA256: '+checkpoint_hash)
                st.caption('Score: similaridade cosseno, não probabilidade de identidade. As marcações exibidas são uma auditoria visual; não explicam causalmente a decisão deste modelo legado.')
            except Exception as error: st.error(f'Neural indisponível: {error}'); return
        query_rows=matrix[matrix.face_id.eq(query)]
        valid=matrix[matrix.correto & matrix['rank'].notna()]
        cols=st.columns(4)
        for col,k in zip(cols[:3],[1,3,5]): col.metric(f'Top-{k} descritivo',f'{(valid["rank"]<=k).sum()/len(ids):.1%}')
        cols[3].metric('MRR descritivo',f'{(1/valid["rank"]).sum()/len(ids):.3f}')
        st.caption(f'{len(valid)}/{len(ids)} pares corretos avaliáveis. Empates recebem a pior posição do grupo.')
        st.dataframe(query_rows,hide_index=True)
        candidate=st.selectbox('Crânio no ranking',query_rows.cranio_id.tolist(),format_func=lambda c:f"{c} {'(par conhecido)' if c==query else ''}")
        a,b=st.columns(2)
        with a: picture(byid[(query,'face')],extracts[(query,'face')],regions[(query,'face')])
        with b: picture(byid[(candidate,'cranio')],extracts[(candidate,'cranio')],regions[(candidate,'cranio')])
        st.dataframe(query_rows[query_rows.cranio_id.eq(candidate)],hide_index=True)
        with st.expander('Todos os 121 pares'):
            st.dataframe(matrix.pivot(index='face_id',columns='cranio_id',values='score'))
            st.download_button('Baixar comparações',matrix.to_csv(index=False).encode(),file_name='matching_piloto.csv')
    with st.expander('Relações osso–pele e literatura'):
        st.markdown('Canto ocular não é margem orbital; largura nasal externa não é abertura óssea. Barba e tecido mole afetam queixo e contorno. Ausências não são preenchidas. Não aplicar deslocamento fixo osso–pele.\n\n[MEPROCS: relações crânio–face](https://link.springer.com/chapter/10.1007/978-3-319-11137-7_3) fundamenta os cuidados anatômicos. [Cranio-ID](https://arxiv.org/html/2511.14411) inspira comparações regionais e globais; usa radiografias e não valida este acervo. Atenção cruzada, transporte ótimo e modelagem de tecido mole não estão implementados neste piloto.')
