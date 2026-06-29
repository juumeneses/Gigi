# -*- coding: utf-8 -*-
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; os.environ['GLOG_minloglevel']='3'; os.environ.setdefault('MPLCONFIGDIR','/tmp/treinamento_faces_mpl')
import argparse,warnings; warnings.filterwarnings('ignore')
import pandas as pd, cv2, mediapipe as mp
from tqdm import tqdm
from common_faces import *
def save_lm(root,y,p,img,lms):
    h,w=img.shape[:2]; md=marks_dir(root,y); stem=p.stem; rows=[]
    for idx,lm in enumerate(lms): rows.append({'id_imagem':stem,'idx':idx,'x':lm.x,'y':lm.y,'z':lm.z,'x_px':lm.x*w,'y_px':lm.y*h,'image_width':w,'image_height':h})
    df=pd.DataFrame(rows); safe_to_csv(df,md/'landmarks_csv'/f'{stem}_landmarks.csv'); write_json(md/'landmarks_json'/f'{stem}_landmarks.json',{'id_imagem':stem,'landmarks':rows})
    cont=df[df.idx.isin(FACE_OVAL_ORDER)].copy(); order={idx:i for i,idx in enumerate(FACE_OVAL_ORDER)}; cont['contorno_ordem']=cont.idx.map(order); cont=cont.sort_values('contorno_ordem'); safe_to_csv(cont,md/'contorno_rosto_csv'/f'{stem}_contorno_rosto.csv')
    cran=cont.copy()
    if not cran.empty:
        cx,cy=cran.x_px.mean(),cran.y_px.mean(); scale=max(cran.x_px.max()-cran.x_px.min(),cran.y_px.max()-cran.y_px.min(),1.0); cran['x_norm_centro']=(cran.x_px-cx)/scale; cran['y_norm_centro']=(cran.y_px-cy)/scale
    safe_to_csv(cran,md/'contorno_cranio_like_csv'/f'{stem}_contorno_cranio_like.csv'); write_image(preview_path(root,y,stem),draw_landmarks_preview(img,lms))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--years',nargs='*'); ap.add_argument('--force',action='store_true'); ap.add_argument('--use-normalized',action='store_true'); args=ap.parse_args()
    if args.use_normalized: os.environ['FACES_USE_NORMALIZED']='1'
    root=as_root(args.root); ensure_base_dirs(root); fails=[]
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=.5) as fm:
        for y in discover_years(root,args.years):
            prepare_year(root,y); sums=[]; fl=[]
            for p in tqdm(list_images_for_year(root,y),desc=f'{y} landmarks'):
                out=marks_dir(root,y)/'landmarks_csv'/f'{p.stem}_landmarks.csv'
                if out.exists() and preview_path(root,y,p.stem).exists() and not args.force: sums.append({'ano':y,'id_imagem':p.stem,'status_landmarks':'ja_existia'}); continue
                img=read_image(p)
                if img is None: row={'ano':y,'id_imagem':p.stem,'erro':'nao abriu'}; fl.append(row); fails.append(row); continue
                res=fm.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                if not res.multi_face_landmarks: row={'ano':y,'id_imagem':p.stem,'erro':'FaceMesh nao detectou'}; fl.append(row); fails.append(row); continue
                save_lm(root,y,p,img,res.multi_face_landmarks[0].landmark); sums.append({'ano':y,'id_imagem':p.stem,'status_landmarks':'ok','num_landmarks':len(res.multi_face_landmarks[0].landmark)})
            safe_to_csv(pd.DataFrame(sums),marks_dir(root,y)/f'summary_landmarks_faces_{y}.csv'); safe_to_csv(pd.DataFrame(fl),marks_dir(root,y)/f'falhas_landmarks_faces_{y}.csv')
    safe_to_csv(pd.DataFrame(fails),resultados_root(root)/'falhas'/'falhas_landmarks_faces_todos_anos.csv'); print('[OK] landmarks')
if __name__=='__main__': main()
