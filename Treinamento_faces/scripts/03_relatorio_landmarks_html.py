# -*- coding: utf-8 -*-
import argparse
from common_faces import *
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--years',nargs='*'); ap.add_argument('--no-open',action='store_true'); args=ap.parse_args(); root=as_root(args.root); out=resultados_root(root)/'relatorios'; out.mkdir(parents=True,exist_ok=True); df=read_csv_safe(resultados_root(root)/'analises_faces'/'analise_padroes_faces_todos_anos.csv'); path=out/'relatorio_landmarks_faces_todos_anos.html'; path.write_text(df.to_html(index=False) if not df.empty else '<p>Sem dados</p>',encoding='utf-8'); print('[OK] HTML',path)
if __name__=='__main__': main()
