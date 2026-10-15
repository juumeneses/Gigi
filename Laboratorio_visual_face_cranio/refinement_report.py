"""Before/after report using originals and exact saved coordinates."""
import base64
import html
from pathlib import Path
import cv2
import numpy as np
from .core import atomic, image, load, overlay, view_frame, write_image


def report(run):
    run=Path(run)
    blocks=['<!doctype html><meta charset="utf-8"><title>Revisão nasal</title>',
            '<style>body{font:16px sans-serif;max-width:1000px;margin:auto}img{width:100%}section{margin:32px 0}</style>',
            '<h1>Propostas nasais: antes e depois</h1>',
            '<p>50 faces e 50 crânios. Amarelo: coordenadas propostas. Ausências permanecem sem pontos. '
            'Revisão visual por IA, sem aprovação anatômica independente. Demais pontos não aprovados.</p>']
    for path in sorted(run.glob('*/proposal.json'),key=lambda p:(load(p)['modality'],load(p)['person'])):
        row=load(path)
        parent=load(row['parent_proposal'])
        old=parent.get('previous_points',parent['points'])
        new=row['points']
        im=image(row['path'])
        # One shared viewport includes old and new locations, avoiding misleading crops.
        p=[(q['x'],q['y']) for q in old+new if q['numero'] in (2,3,4) and q.get('x') is not None]
        xy=np.array(p)
        low,high=xy.min(0),xy.max(0)
        margin=max(float((high-low).max())*.4,12)
        region=dict(box=[*(low-margin),*(high+margin)])
        before=view_frame(overlay(im,[q for q in old if q['numero'] in (2,3,4)]),region)
        after=view_frame(overlay(im,[q for q in new if q['numero'] in (2,3,4)]),region)
        joined=np.hstack([before,after])
        for label,x in [('Anterior',12),('Proposta revisada por IA',612)]:
            cv2.rectangle(joined,(x-5,0),(x+360,30),(0,0,0),-1)
            cv2.putText(joined,label,(x,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        write_image(run/row['key']/'nose_comparison.jpg',joined)
        encoded=base64.b64encode(cv2.imencode('.jpg',joined)[1]).decode()
        blocks.append(f'<section><h2>{html.escape(row["person"])} · {row["modality"]}</h2>'
                      f'<p>{html.escape(row.get("visual_audit",{}).get("summary",""))}</p>'
                      f'<img src="data:image/jpeg;base64,{encoded}"></section>')
    atomic(run/'report.html','\n'.join(blocks).encode())
