# -*- coding: utf-8 -*-
import argparse
from common_faces import *
def build(root,apply_high=False,high=.80):
    ok,msg=rebuild_genero_curado(root,apply_high=apply_high,high=high)
    print('[OK]' if ok else '[ERRO]', msg)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--apply-high',action='store_true'); ap.add_argument('--high-threshold',type=float,default=.80); args=ap.parse_args(); root=as_root(args.root); ensure_base_dirs(root); build(root,args.apply_high,args.high_threshold)
if __name__=='__main__': main()
