"""Isolated skull coordinate-regression smoke test on AI pseudolabels."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import LAB_ROOT, REPO_ROOT


def model():
    return nn.Sequential(nn.Conv2d(3,16,5,stride=2,padding=2),nn.ReLU(),
        nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU(),
        nn.Conv2d(32,32,3,stride=2,padding=1),nn.ReLU(),
        nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(512,64),nn.ReLU(),
        nn.Linear(64,22),nn.Sigmoid())


def prepare(source):
    meta=json.loads((source/'lote.json').read_text())
    points=pd.read_csv(source/'pontos.csv')
    samples=[]
    for entry in meta['entradas']:
        if entry['split'] not in ['treino','validacao']:
            raise ValueError('Teste final nao permitido neste piloto.')
        view=entry['views']['cranio']
        path=REPO_ROOT/view['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==view['hash']
        with Image.open(path) as raw:
            im=raw.convert('RGB').crop(tuple(view['box']))
        w,h=im.size
        scale=min(128/w,128/h)
        sw,sh=round(w*scale),round(h*scale)
        ox,oy=(128-sw)//2,(128-sh)//2
        canvas=Image.new('RGB',(128,128))
        canvas.paste(im.resize((sw,sh)),(ox,oy))
        frame=points[(points.id_pessoa==entry['id_pessoa'])&(points.modalidade=='cranio')].sort_values('numero')
        assert frame.numero.tolist()==list(range(1,12))
        mask=frame.estado.eq('PROPOSTA_IA').to_numpy()
        coords=frame[['x_px','y_px']].to_numpy(dtype=float)
        assert np.isfinite(coords[mask]).all()
        coords[:,0]=((coords[:,0]-view['box'][0])*sw/w+ox)/128
        coords[:,1]=((coords[:,1]-view['box'][1])*sh/h+oy)/128
        coords[~mask]=0  # Placeholder excluded from every loss and metric.
        samples.append(dict(id=entry['id_pessoa'],split=entry['split'],view=view,
            tensor=torch.tensor(np.array(canvas).transpose(2,0,1)/255,dtype=torch.float32),
            target=torch.tensor(coords,dtype=torch.float32),mask=torch.tensor(mask),
            transform=[w,h,sw,sh,ox,oy],image=im))
    return samples


def masked_loss(pred,target,mask):
    return ((pred-target).square().sum(-1)*mask).sum()/mask.sum().clamp_min(1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    assert args.epochs>0
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    source=args.source.resolve()
    samples=prepare(source)
    train=[s for s in samples if s['split']=='treino']
    val=[s for s in samples if s['split']=='validacao']
    assert train and val and not ({s['id'] for s in train}&{s['id'] for s in val})
    out=LAB_ROOT/'resultados/piloto_localizador_ia'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    def log(message):
        print(message,flush=True)
        with (out/'execucao.log').open('a') as stream:
            stream.write(message+'\n')
    net=model()
    optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    x=torch.stack([s['tensor'] for s in train])
    y=torch.stack([s['target'] for s in train])
    m=torch.stack([s['mask'] for s in train])
    baseline=(y*m[...,None]).sum(0)/m.sum(0).clamp_min(1)[...,None]
    config=dict(seed=args.seed,epochs=args.epochs,lr=.001,input_size=128,
        architecture=str(net),torch_version=torch.__version__,source=str(source.relative_to(LAB_ROOT)),
        pseudolabel_hash=hashlib.sha256((source/'pontos.csv').read_bytes()).hexdigest(),
        train_ids=[s['id'] for s in train],validation_ids=[s['id'] for s in val],
        finalidade='SMOKE_TEST_PSEUDORROTULOS',validacao_anatomica=False,matching_liberado=False,
        checkpoint_selection='Ultima epoca; numero de epocas fixado antes da execucao',
        limitations='9 treino e 1 validacao com rotulos IA; sem teste independente; nao prediz visibilidade.')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    log(f"Cranio apenas: {len(train)} treino, {len(val)} validacao. Matching bloqueado.")
    history=[]
    for epoch in range(1,args.epochs+1):
        net.train(); optimizer.zero_grad()
        pred=net(x).reshape(-1,11,2)
        loss=masked_loss(pred,y,m)
        assert torch.isfinite(loss)
        loss.backward(); optimizer.step()
        history.append(dict(epoch=epoch,train_mse=float(loss.detach())))
        if epoch==1 or epoch%10==0:
            log(f'Epoca {epoch}/{args.epochs}: loss treino={float(loss.detach()):.6f}')
    pd.DataFrame(history).to_csv(out/'curva_treino.csv',index=False)
    torch.save(dict(state_dict=net.state_dict(),config=config,baseline=baseline),out/'ultimo.pt')
    net.eval()
    rows=[]
    for s in samples:
        with torch.no_grad():
            pred=net(s['tensor'][None]).reshape(11,2).numpy()
        w,h,sw,sh,ox,oy=s['transform']
        overlay=s['image'].copy()
        draw=ImageDraw.Draw(overlay)
        def local(arr):
            return np.stack([(arr[:,0]*128-ox)*w/sw,(arr[:,1]*128-oy)*h/sh],axis=1)
        p=local(pred); t=local(s['target'].numpy()); b=local(baseline.numpy())
        diagonal=np.hypot(w,h)
        for i in range(11):
            visible=bool(s['mask'][i])
            error=float(np.linalg.norm(p[i]-t[i])) if visible else None
            base=float(np.linalg.norm(b[i]-t[i])) if visible else None
            rows.append(dict(id_pessoa=s['id'],split=s['split'],numero=i+1,
                erro_px_vs_ia=error,nme_vs_ia=error/diagonal if visible else None,
                baseline_nme_vs_ia=base/diagonal if visible else None,
                referencia_disponivel=visible,x_original=float(p[i,0]+s['view']['box'][0]),
                y_original=float(p[i,1]+s['view']['box'][1])))
            px,py=p[i]
            draw.ellipse((px-4,py-4,px+4,py+4),outline='magenta',width=2)
            if visible:
                tx,ty=t[i]
                draw.ellipse((tx-4,ty-4,tx+4,ty+4),outline='cyan',width=2)
                draw.line((px,py,tx,ty),fill='white',width=1)
        overlay.thumbnail((700,850))
        overlay.save(out/(s['id']+'_comparacao.jpg'))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'predicoes.csv',index=False)
    summary=frame.groupby('split')[['nme_vs_ia','baseline_nme_vs_ia']].mean().reset_index()
    summary.to_csv(out/'metricas.csv',index=False)
    log(summary.to_string(index=False))
    log('Concluido. Erros medem concordancia com IA, nao acuracia anatomica.')
    (out/'concluido.json').write_text(json.dumps(dict(status='CONCLUIDO',matching_liberado=False)))
    print(out,flush=True)


if __name__=='__main__':
    main()
