"""Reproducible facial proposals; preserve source batches and review provenance."""
from pathlib import Path
import argparse
import copy
import cv2
import numpy as np

from .core import MESH_MAP, image, load, new_run, regions_for, save, sha, write_image


def refine(source):
    import mediapipe as mp
    source = Path(source).resolve()
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1)
    run = new_run('annotation_batches')
    config = load(source/'config.json')
    config.update(parent_batch=str(source), refinement='mediapipe_static_refined_v1',
                  mediapipe_version=mp.__version__, mesh_map=MESH_MAP,
                  parameters=dict(static_image_mode=True, refine_landmarks=True, max_num_faces=1))
    save(run/'config.json', config)
    save(run/'inputs.json', load(source/'inputs.json'))
    with mesh:
        for path in sorted(source.glob('*/proposal.json')):
            row = copy.deepcopy(load(path))
            if sha(row['path']) != row['hash']:
                raise ValueError('Input changed: '+row['path'])
            row['parent_proposal'] = str(path)
            row['parent_proposal_hash'] = sha(path)
            row['refinement_status'] = 'UNCHANGED_SKULL'
            if row['modality'] == 'face':
                im = image(row['path'])
                detection = mesh.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                if detection.multi_face_landmarks:
                    lm = detection.multi_face_landmarks[0].landmark
                    row['previous_points'] = row['points']
                    row['mesh'] = [[float(p.x*row['width']), float(p.y*row['height'])] for p in lm]
                    row['points'] = [dict(numero=n+1, x=row['mesh'][i][0], y=row['mesh'][i][1],
                                          estado='MEDIAPIPE_CANDIDATO', visibility='NAO_AVALIADO',
                                          method='mediapipe', mesh_index=i) for n,i in enumerate(MESH_MAP)]
                    row['origin'] = 'PROPOSTA_MEDIAPIPE'
                    row['refinement_status'] = 'FACE_REESTIMATED'
                else:
                    row['refinement_status'] = 'NO_FACE_DETECTED_OLD_PROPOSAL_RETAINED'
                row['regions'] = regions_for(row['points'], row['width'], row['height'])
            target = run/row['key']/'proposal.json'
            save(target, row)
            save(target.parent/'proposal.sha.json', dict(sha256=sha(target)))
            print(row['filename'], row['refinement_status'], flush=True)
    save(run/'status.json', dict(state='CONCLUIDO', anatomical_validation=False))
    sheets(run)
    return run


def sheets(run, modality='face'):
    run = Path(run)
    rows = sorted([load(p) for p in run.glob('*/proposal.json') if load(p)['modality']==modality], key=lambda r:r['person'])
    mapping = []
    for start in range(0, len(rows), 10):
        canvas = np.full((5*230, 2*360, 3), 245, np.uint8)
        for k, row in enumerate(rows[start:start+10]):
            im = image(row['path'])
            pts = {p['numero']:p for p in row['points']}
            xs = [pts[n]['x'] for n in [2,3,4]]
            ys = [pts[n]['y'] for n in [2,3,4]]
            cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
            extent = max(max(xs)-min(xs), 30)*2.4
            scale = 300/extent
            matrix = np.array([[scale,0,180-cx*scale],[0,scale,120-cy*scale]])
            tile = cv2.warpAffine(im,matrix,(360,230))
            for n in [2,3,4]:
                x,y = (matrix @ np.array([pts[n]['x'],pts[n]['y'],1])).round().astype(int)
                cv2.circle(tile,(x,y),3,(0,255,255),1)
                cv2.putText(tile,str(n),(x+4,y-4),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
            cv2.rectangle(tile,(0,0),(359,24),(245,245,245),-1)
            cv2.putText(tile,f'{start+k}: {row["person"]}',(5,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
            canvas[(k//2)*230:(k//2+1)*230,(k%2)*360:(k%2+1)*360] = tile
            mapping.append(dict(index=start+k,key=row['key'],person=row['person'],matrix=matrix.tolist()))
        write_image(run/f'{modality}_nose_sheet_{start//10}.jpg',canvas)
    save(run/f'{modality}_sheet_mapping.json',mapping)


def skull_sheets(run):
    run=Path(run)
    rows=sorted([load(p) for p in run.glob('*/proposal.json') if load(p)['modality']=='cranio'],key=lambda r:r['person'])
    mapping=[]
    for start in range(0,len(rows),10):
        canvas=np.full((1500,720,3),245,np.uint8)
        for k,row in enumerate(rows[start:start+10]):
            im=image(row['path'])
            mask=cv2.threshold(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),30,255,cv2.THRESH_BINARY)[1]
            x,y,w,h=cv2.boundingRect(mask)
            scale=min(340/max(w,1),400/max(h,1))
            matrix=np.array([[scale,0,180-(x+w/2)*scale],[0,scale,130-(y+h*.48)*scale]])
            tile=cv2.warpAffine(im,matrix,(360,300))
            cv2.rectangle(tile,(0,0),(359,24),(245,245,245),-1)
            cv2.putText(tile,f'{start+k}: {row["person"]}',(5,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
            canvas[(k//2)*300:(k//2+1)*300,(k%2)*360:(k%2+1)*360]=tile
            mapping.append(dict(index=start+k,key=row['key'],person=row['person'],matrix=matrix.tolist()))
        write_image(run/f'cranio_context_{start//10}.jpg',canvas)
    save(run/'cranio_context_mapping.json',mapping)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source')
    args=parser.parse_args()
    print(refine(args.source))
