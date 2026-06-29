# -*- coding: utf-8 -*-
import argparse, os, subprocess, sys
from pathlib import Path
from common_faces import as_root, cleanup_generated_outputs, discover_years, ensure_base_dirs
STEPS=['00_checar_dataset_faces.py','01_extrair_landmarks_faces.py','02_gerar_medidas_faces.py','05_analisar_padroes_faces.py','03_relatorio_landmarks_html.py']
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--years',nargs='*'); ap.add_argument('--force',action='store_true'); ap.add_argument('--no-open',action='store_true'); ap.add_argument('--serve-report',action='store_true'); ap.add_argument('--non-interactive-check',action='store_true'); ap.add_argument('--use-normalized',action='store_true'); args=ap.parse_args()
    if args.use_normalized: os.environ['FACES_USE_NORMALIZED']='1'
    root=as_root(args.root); ensure_base_dirs(root); years=discover_years(root,args.years)
    if not years:
        print('[ERRO] Nenhum ano encontrado em dados_originais/faces/<ANO>.')
        raise SystemExit(1)
    print('[INFO] Anos do pipeline:', ', '.join(years))
    if args.force:
        removed=cleanup_generated_outputs(root,years,keep_logs=True)
        print(f'[FORCE] Limpeza concluída: {len(removed)} arquivo(s)/pasta(s) gerados removidos. Originais preservados.')
    for name in STEPS:
        cmd=[sys.executable,str(root/'scripts'/name),'--root',str(root)]
        cmd+=['--years']+years
        if args.force and name=='01_extrair_landmarks_faces.py': cmd.append('--force')
        if args.use_normalized and name=='01_extrair_landmarks_faces.py': cmd.append('--use-normalized')
        if args.no_open and name=='03_relatorio_landmarks_html.py': cmd.append('--no-open')
        print('\n[ETAPA]',name)
        print('[CMD]',' '.join(cmd)); rc=subprocess.call(cmd,cwd=str(root))
        if rc:
            print('[ERRO] etapa falhou:',name,'codigo',rc)
            raise SystemExit(rc)
    print('[OK] pipeline completo')
if __name__=='__main__': main()
