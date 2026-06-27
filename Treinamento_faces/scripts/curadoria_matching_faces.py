# -*- coding: utf-8 -*-
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; os.environ['GLOG_minloglevel']='3'; os.environ.setdefault('MPLCONFIGDIR','/tmp/treinamento_faces_mpl')
import argparse, shutil, warnings; warnings.filterwarnings('ignore')
import cv2, numpy as np, pandas as pd, mediapipe as mp
from tqdm import tqdm
from common_faces import *
def detect(fm,img):
    r=fm.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); return None if not r.multi_face_landmarks else r.multi_face_landmarks[0].landmark
def rotate(img,ang):
    if abs(ang)<.01: return img.copy()
    h,w=img.shape[:2]; c=(w/2,h/2); m=cv2.getRotationMatrix2D(c,ang,1); cos=abs(m[0,0]); sin=abs(m[0,1]); nw=int(h*sin+w*cos); nh=int(h*cos+w*sin); m[0,2]+=nw/2-c[0]; m[1,2]+=nh/2-c[1]; return cv2.warpAffine(img,m,(nw,nh),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
def clahe(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab); l=cv2.createCLAHE(2.0,(8,8)).apply(l); return cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2BGR)
def sharp(img): return cv2.addWeighted(img,1.45,cv2.GaussianBlur(img,(0,0),1.2),-.45,0)
def candidates(img):
    out=[('original',img),('contraste_CLAHE',clahe(img)),('nitidez',sharp(img))]
    for bn,b in [('original',img),('CLAHE',clahe(img)),('nitidez',sharp(img))]:
        for a in [-10,-7,-5,-3,3,5,7,10]: out.append((f'{bn}_rot_{a:+d}',rotate(b,a)))
    return out
def try_fix(fm,img):
    best={'detected':False,'name':None,'image':None,'score':-1e9,'metrics':{},'landmarks':None}
    for n,c in candidates(img):
        lms=detect(fm,c)
        if lms is None: continue
        sc,mt=score_landmarks(c,lms)
        if sc>best['score']: best={'detected':True,'name':n,'image':c,'score':sc,'metrics':mt,'landmarks':lms}
    return best
def visually_good(mt):
    if not mt: return False,'sem_metricas'
    try:
        sc=float(mt.get('score',-100)); fora=float(mt.get('pontos_fora_imagem',1)); area=float(mt.get('area_face_normalizada',0)); inc=float(mt.get('inclinacao_olhos_graus',999)); ro=float(mt.get('razao_olhos_largura_face',0)); rb=float(mt.get('razao_boca_largura_face',0)); rw=float(mt.get('razao_largura_altura_face',0))
    except Exception: return False,'erro_metricas'
    bad=[]
    if sc<55: bad.append('score_baixo')
    if fora>.01: bad.append('pontos_fora')
    if not (.06<=area<=.90): bad.append('area')
    if inc>18: bad.append('inclinacao')
    if not (.42<=ro<=.85): bad.append('olhos')
    if not (.16<=rb<=.65): bad.append('boca')
    if not (.38<=rw<=1.20): bad.append('larg_alt')
    return (not bad), ';'.join(bad) if bad else 'landmarks_visualmente_coerentes'
def decide(q,od,osc,best,min_score,min_gain):
    if not best['detected']: return 'EXCLUIR_AUTO','NAO_USAR',False,'Nenhuma tentativa detectou face.'
    ok,mot=visually_good(best.get('metrics',{})); gain=best['score']-osc
    if str(q).upper()=='OK' and od and osc>=min_score: return 'OK_AUTO','USAR_ORIGINAL',True,'Já estava OK.'
    if best['name']=='original' and ok: return 'MANTER_ORIGINAL_AUTO','USAR_ORIGINAL',True,'Original com landmarks bons.'
    if best['name']!='original' and ok and gain>=min_gain: return 'CORRIGIR_AUTO','APLICAR_CORRECAO',True,'Correção melhorou.'
    if od and osc>=55: return 'MANTER_ORIGINAL_AUTO','USAR_ORIGINAL',True,'Original aceitável.'
    if ok: return 'CORRIGIR_AUTO','APLICAR_CORRECAO',True,'Sugestão aceitável.'
    return 'REVISAR_MANUAL','NAO_USAR',False,'Caso duvidoso.'
def apply_corr(root,y,stem,orig,img,name):
    corr=corrected_dir(root,y); bkp=backup_dir(root,y); corr.mkdir(parents=True,exist_ok=True); bkp.mkdir(parents=True,exist_ok=True); cp=corr/f'{stem}_corrigida_{name}{orig.suffix.lower()}'; bp=bkp/orig.name; write_image(cp,img); 
    if not bp.exists(): shutil.copy2(orig,bp)
    shutil.copy2(cp,orig); return cp,bp
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--years',nargs='*'); ap.add_argument('--apply',action='store_true'); ap.add_argument('--all',action='store_true'); ap.add_argument('--min-score',type=float,default=70); ap.add_argument('--min-gain',type=float,default=3); args=ap.parse_args(); root=as_root(args.root); ensure_base_dirs(root); years=discover_years(root,args.years); df=load_all_analysis(root,years)
    if df.empty: print('[ERRO] Rode análise antes'); return
    work=df.copy() if args.all else df[df.qualidade_landmarks.astype(str).str.upper().isin(['REVISAR','FALHA'])].copy(); out=resultados_root(root)/'curadoria_matching'; prev=out/'previews'; prev.mkdir(parents=True,exist_ok=True); rows=[]
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=.5) as fm:
      for _,it in tqdm(work.iterrows(),total=len(work),desc='Curadoria'):
        y=str(it.get('ano','')); stem=str(it.get('id_imagem','')); sm=str(it.get('status_manual','')).upper(); row={'ano':y,'id_imagem':stem,'qualidade_landmarks':str(it.get('qualidade_landmarks','')),'motivo_qualidade':str(it.get('motivo_qualidade','')),'status_manual':sm,'status_final':'','usar_matching':False,'acao_automatica':'','decisao_motivo':'','score_original':'','score_sugerido':'','ganho_score':'','sugestao':'','imagem_original':'','imagem_corrigida':'','backup_original':'','preview_antes':'','preview_depois':''}
        if sm in ['EXCLUIR','EXCLUIR_MANUAL']: row.update({'status_final':'EXCLUIR_MANUAL','acao_automatica':'NAO_USAR','decisao_motivo':'manual excluir'}); rows.append(row); continue
        if sm in ['OK_MANUAL','MANTER_ORIGINAL']: row.update({'status_final':'OK_MANUAL','usar_matching':True,'acao_automatica':'USAR_ORIGINAL','decisao_motivo':'manual ok'}); rows.append(row); continue
        ip=find_image_path(root,y,stem)
        if ip is None: row.update({'status_final':'EXCLUIR_AUTO','acao_automatica':'NAO_USAR','decisao_motivo':'imagem não encontrada'}); rows.append(row); continue
        row['imagem_original']=str(ip); img=read_image(ip)
        if img is None: row.update({'status_final':'EXCLUIR_AUTO','acao_automatica':'NAO_USAR','decisao_motivo':'imagem não abriu'}); rows.append(row); continue
        l=detect(fm,img)
        if l is None: od=False; osc=-100; omt={}
        else: od=True; osc,omt=score_landmarks(img,l); pa=prev/f'{y}_{stem}_antes.jpg'; write_image(pa,draw_landmarks_preview(img,l)); row['preview_antes']=str(pa)
        best=try_fix(fm,img); sf,ac,usar,mot=decide(row['qualidade_landmarks'],od,osc,best,args.min_score,args.min_gain); row.update({'status_final':sf,'acao_automatica':ac,'usar_matching':bool(usar),'decisao_motivo':mot,'score_original':round(float(osc),3),'sugestao':str(best.get('name',''))})
        if best['detected']:
            row['score_sugerido']=round(float(best['score']),3); row['ganho_score']=round(float(best['score']-osc),3); pdp=prev/f'{y}_{stem}_depois_{best["name"]}.jpg'; write_image(pdp,draw_landmarks_preview(best['image'],best['landmarks'])); row['preview_depois']=str(pdp)
        if ac=='APLICAR_CORRECAO':
            if args.apply: cp,bp=apply_corr(root,y,stem,ip,best['image'],str(best['name']).replace(' ','_')); row['imagem_corrigida']=str(cp); row['backup_original']=str(bp); row['status_final']='CORRIGIDO_AUTO'; row['usar_matching']=True
            else: row['status_final']='CORRIGIR_AUTO_PENDENTE'; row['usar_matching']=False
        for k,v in omt.items(): row[f'original_{k}']=v
        if best['detected']:
            for k,v in best['metrics'].items(): row[f'sugerido_{k}']=v
        rows.append(row); print(y,stem,row['status_final'])
    if not args.all:
        done_keys={(str(r.get('ano','')),str(r.get('id_imagem',''))) for r in rows}
        for _,it in df.iterrows():
            y=str(it.get('ano','')); stem=str(it.get('id_imagem',''))
            if (y,stem) in done_keys: continue
            sm=str(it.get('status_manual','')).upper()
            q=str(it.get('qualidade_landmarks','')).upper()
            row={'ano':y,'id_imagem':stem,'qualidade_landmarks':q,'motivo_qualidade':str(it.get('motivo_qualidade','')),'status_manual':sm,'status_final':'','usar_matching':False,'acao_automatica':'','decisao_motivo':'','score_original':'','score_sugerido':'','ganho_score':'','sugestao':'','imagem_original':'','imagem_corrigida':'','backup_original':'','preview_antes':'','preview_depois':''}
            ip=find_image_path(root,y,stem)
            row['imagem_original']=str(ip) if ip else ''
            if sm in ['EXCLUIR','EXCLUIR_MANUAL']:
                row.update({'status_final':'EXCLUIR_MANUAL','usar_matching':False,'acao_automatica':'NAO_USAR','decisao_motivo':'manual excluir'})
            elif sm in ['OK_MANUAL','MANTER_ORIGINAL']:
                row.update({'status_final':'OK_MANUAL','usar_matching':True,'acao_automatica':'USAR_ORIGINAL','decisao_motivo':'manual ok'})
            elif q=='OK':
                row.update({'status_final':'OK_AUTO','usar_matching':True,'acao_automatica':'USAR_ORIGINAL','decisao_motivo':'metricas_ok'})
            else:
                row.update({'status_final':'REVISAR_MANUAL','usar_matching':False,'acao_automatica':'NAO_USAR','decisao_motivo':'não processado'})
            rows.append(row)
    st=pd.DataFrame(rows); safe_to_csv(st,out/'status_matching_faces.csv'); use=st.usar_matching.astype(str).str.upper().isin(['TRUE','1','SIM']); safe_to_csv(st[use].copy(),out/'dataset_matching_faces.csv'); safe_to_csv(st.groupby(['status_final','usar_matching'],dropna=False).size().reset_index(name='qtd'),out/'resumo_curadoria_matching.csv'); print('[OK] curadoria')
if __name__=='__main__': main()
