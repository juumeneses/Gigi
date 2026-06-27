# -*- coding: utf-8 -*-
import argparse,re,shutil, pandas as pd
from common_faces import *
def infer_year(name):
    m=re.match(r'^(20\d{2}|19\d{2})',name) or re.search(r'(20\d{2}|19\d{2})',name); return m.group(1) if m else None
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--years',nargs='*'); args=ap.parse_args(); root=as_root(args.root); ensure_base_dirs(root); rows=[]; probs=[]
    for p in faces_root(root).rglob('*'):
        if not is_original_face_image(p): continue
        y=infer_year(p.name)
        if not y: probs.append({'arquivo':str(p),'problema':'ano não inferido'}); continue
        if args.years and y not in [str(a) for a in args.years]: continue
        prepare_year(root,y); dest=images_dir(root,y)/p.name; acao='ok'
        if p.resolve()!=dest.resolve():
            if not dest.exists(): shutil.copy2(p,dest); acao='copiado_para_images'
            else: acao='ja_existia'
        rows.append({'ano':y,'arquivo':str(p),'nome':p.name,'destino':str(dest),'acao':acao})
    out=resultados_root(root)/'checks'; safe_to_csv(pd.DataFrame(rows),out/'check_faces.csv'); safe_to_csv(pd.DataFrame(probs),out/'problemas_nomes_faces.csv'); print('[OK] check faces',len(rows))
if __name__=='__main__': main()
