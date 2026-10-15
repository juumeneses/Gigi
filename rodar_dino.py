"""Start the lab using this interpreter on Windows, macOS or Linux."""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def check():
    errors=[]
    if sys.version_info < (3,11):errors.append('Use Python 3.11 ou superior (testado com 3.11).')
    for name in ('streamlit','torch','torchvision','numpy','pandas','PIL','cv2','sklearn','joblib','matplotlib'):
        if importlib.util.find_spec(name) is None:errors.append('Dependencia ausente: '+name)
    if not (ROOT/'Banco').is_dir():errors.append('Banco ausente: execute git lfs pull e confira a pasta Banco na raiz.')
    split=ROOT/'Experimentos_hibridos_cranio_face/dataset/splits/split_identidades.csv'
    if not split.exists():errors.append('Split por identidade ausente: '+str(split))
    for error in errors:print('ERRO:',error)
    if not errors:print('Pre-requisitos encontrados. Pesos externos sao preparados no painel, se ausentes.')
    return not errors


def main():
    parser=argparse.ArgumentParser(description='Laboratorio DINOv2 face-cranio')
    parser.add_argument('--port',type=int,default=8503)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if not check():return 1
    if args.check:return 0
    from Treinamento_DINOv2_face_cranio.portable import register
    register()
    return subprocess.call([sys.executable,'-m','streamlit','run',str(ROOT/'Treinamento_DINOv2_face_cranio/app.py'),
        '--server.port',str(args.port),'--server.address','127.0.0.1','--browser.gatherUsageStats','false'],cwd=ROOT)


if __name__=='__main__':raise SystemExit(main())
