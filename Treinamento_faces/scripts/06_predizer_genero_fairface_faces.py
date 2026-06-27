# -*- coding: utf-8 -*-
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; os.environ['GLOG_minloglevel']='3'; os.environ.setdefault('MPLCONFIGDIR','/tmp/treinamento_faces_mpl')
import argparse, subprocess, sys, warnings; warnings.filterwarnings('ignore')
import cv2, numpy as np, pandas as pd, mediapipe as mp
from tqdm import tqdm
from common_faces import *
MODEL_ID='11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB'; MODEL_NAME='res34_fair_align_multi_7_20190809.pt'
def ensure_model(root):
    p=root/'external_models'/'fairface'/MODEL_NAME; p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.stat().st_size>1000000: return p
    try: import gdown
    except Exception: subprocess.check_call([sys.executable,'-m','pip','install','gdown']); import gdown
    gdown.download(f'https://drive.google.com/uc?id={MODEL_ID}',str(p),quiet=False)
    if not p.exists() or p.stat().st_size<1000000: raise RuntimeError('Não consegui baixar o modelo FairFace. Baixe manualmente para '+str(p))
    return p
def load_model(p,device):
    import torch, torch.nn as nn, torchvision
    model=torchvision.models.resnet34(weights=None); model.fc=nn.Linear(model.fc.in_features,18); st=torch.load(str(p),map_location=device)
    if isinstance(st,dict) and 'state_dict' in st: st=st['state_dict']
    st={k.replace('module.',''):v for k,v in st.items()} if isinstance(st,dict) else st
    model.load_state_dict(st,strict=False); model.to(device); model.eval(); return model
def preprocess(img):
    import torch
    from torchvision import transforms
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB); tr=transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]); return tr(rgb).unsqueeze(0)
def predict(model,device,img):
    import torch
    x=preprocess(img).to(device)
    with torch.no_grad(): out=model(x).cpu().numpy().squeeze()
    logits=out[7:9]; e=np.exp(logits-np.max(logits)); probs=e/e.sum(); pm=float(probs[0]); pf=float(probs[1]); cl,conf=class_from_gender_probs(pm,pf); return pm,pf,cl,conf
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--years', nargs='*')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    root = as_root(args.root)
    ensure_base_dirs(root)
    out = genero_dir(root)
    out.mkdir(parents=True, exist_ok=True)

    status, dataset, _ = load_curadoria(root)
    # Usa dataset_matching_faces.csv como principal. Se ele estiver incompleto, usa status_matching_faces.csv com usar_matching=True.
    candidates = dataset.copy()
    if not status.empty and 'usar_matching' in status.columns:
        status_true = status[status['usar_matching'].astype(str).str.upper().isin(['TRUE', '1', 'SIM', 'YES'])].copy()
        candidates = pd.concat([candidates, status_true], ignore_index=True, sort=False)
    if candidates.empty:
        print('[ERRO] Não encontrei faces aprovadas para matching. Rode/aplique a curadoria antes.')
        return
    candidates['ano'] = candidates['ano'].astype(str)
    candidates['id_imagem'] = candidates['id_imagem'].astype(str)
    candidates = candidates.drop_duplicates(['ano', 'id_imagem'], keep='last')

    if args.years:
        candidates = candidates[candidates['ano'].isin([str(y) for y in args.years])].copy()

    old = read_csv_safe(out / 'predicoes_genero_fairface.csv')
    if not old.empty and {'ano', 'id_imagem'}.issubset(old.columns):
        old['ano'] = old['ano'].astype(str)
        old['id_imagem'] = old['id_imagem'].astype(str)
    else:
        old = pd.DataFrame()

    done = set() if args.force or old.empty else set(zip(old['ano'].astype(str), old['id_imagem'].astype(str)))
    pending = candidates.copy() if args.force else candidates[~candidates.apply(lambda r: (str(r['ano']), str(r['id_imagem'])) in done, axis=1)].copy()
    if args.limit and args.limit > 0:
        pending = pending.head(int(args.limit)).copy()

    print(f'[INFO] Faces aprovadas candidatas: {len(candidates)}')
    print(f'[INFO] Já processadas no CSV FairFace: {len(done)}')
    print(f'[INFO] Pendentes nesta execução: {len(pending)}')
    if pending.empty:
        print('[OK] Nenhuma imagem pendente. Use --force para refazer tudo.')
        return

    rows = [] if args.force or old.empty else old.to_dict('records')
    import torch
    device = torch.device('cpu')
    model = load_model(ensure_model(root), device)
    crops = out / 'crops_fairface'
    crops.mkdir(parents=True, exist_ok=True)

    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=.5) as fm:
        for _, r in tqdm(pending.iterrows(), total=len(pending), desc='FairFace'):
            y = str(r.get('ano', ''))
            stem = str(r.get('id_imagem', ''))
            ip = find_image_path(root, y, stem)
            row = {
                'ano': y, 'id_imagem': stem, 'imagem_original': str(ip) if ip else '',
                'status_final': str(r.get('status_final', '')), 'usar_matching': True,
                'modelo_ia': 'FAIRFACE_RESNET34_7RACE', 'classe_fairface': 'ERRO_DETECCAO',
                'prob_masculino_fairface': np.nan, 'prob_feminino_fairface': np.nan,
                'confianca_fairface': 'ERRO', 'status_genero_ia': 'REVISAR_MANUAL',
                'motivo_revisao_genero': '', 'crop_fairface': ''
            }
            try:
                if ip is None:
                    raise RuntimeError('imagem_nao_encontrada')
                img = read_image(ip)
                if img is None:
                    raise RuntimeError('imagem_nao_abriu')
                res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                crop = img
                if res.multi_face_landmarks:
                    crop = crop_face_by_landmarks(img, res.multi_face_landmarks[0].landmark)
                cp = crops / f'{y}_{stem}_crop.jpg'
                write_image(cp, crop)
                pm, pf, cl, conf = predict(model, device, crop)
                pmax = max(pm, pf)
                row.update({
                    'classe_fairface': cl,
                    'prob_masculino_fairface': round(pm, 6),
                    'prob_feminino_fairface': round(pf, 6),
                    'confianca_fairface': conf,
                    'status_genero_ia': 'ACEITAR_SUGESTAO' if pmax >= .65 and cl != 'INDEFINIDO' else 'REVISAR_MANUAL',
                    'motivo_revisao_genero': 'confianca_media_ou_alta' if pmax >= .65 else 'baixa_confianca_ou_indefinido',
                    'crop_fairface': str(cp)
                })
            except Exception as e:
                row['classe_fairface'] = 'ERRO_MODELO'
                row['motivo_revisao_genero'] = str(e)
            rows.append(row)

    final = pd.DataFrame(rows)
    if not final.empty:
        final['ano'] = final['ano'].astype(str)
        final['id_imagem'] = final['id_imagem'].astype(str)
        final = final.drop_duplicates(['ano', 'id_imagem'], keep='last').sort_values(['ano', 'id_imagem'])
    safe_to_csv(final, out / 'predicoes_genero_fairface.csv')
    print(f'[OK] FairFace atualizado: {out / "predicoes_genero_fairface.csv"}')

if __name__=='__main__': main()
