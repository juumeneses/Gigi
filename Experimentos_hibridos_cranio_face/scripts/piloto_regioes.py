"""CLI for isolated regional audits, from any working directory."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gigi-matplotlib')
os.environ.setdefault('GLOG_minloglevel','2')
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.region_pilot import run_audit, STAGES

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--stage',choices=['all']+STAGES,default='all')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--force',action='store_true')
    p.add_argument('--limit',type=int,help='Parada controlada apos N imagens/etapas para testar retomada')
    a=p.parse_args()
    try:
        run_audit(a.run_dir,a.stage,a.resume,a.force,a.limit)
    except Exception as error:
        print(f'ERRO: {type(error).__name__}: {error}',flush=True)
        raise
