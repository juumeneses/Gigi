from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .core import (MESH_MAP, VERSION, digest, image, letterbox, load, new_run, overlay,
                   regions_for, save, sha, write_image)


def process(rows, action, training_run=None, reference=None, run=None, resume=False,
            size=256, stop=None, log=print):
    if not rows:
        raise ValueError('Nenhuma imagem selecionada')
    if action not in ('normalize', 'apply', 'mesh', 'regions', 'contours'):
        raise ValueError('Acao desconhecida')
    model = state = mesh = None
    mesh_error = None
    checkpoint_hash = None
    if action == 'apply':
        from .learning import predictor
        model, state = predictor(training_run)
        checkpoint_hash = sha(Path(training_run) / 'best.pt')
        if any(r['modality'] != state['config']['modality'] or r['view'] != state['config']['view'] for r in rows):
            raise ValueError('Modelo incompativel com modalidade/vista selecionada')
    if action == 'mesh':
        if any(r['modality'] != 'face' or r['view'] not in ('FFN', 'FFS') for r in rows):
            raise ValueError('Extracao MediaPipe habilitada apenas para fotos frontais')
        import mediapipe as mp
        try:
            mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                 refine_landmarks=True, min_detection_confidence=.5)
        except RuntimeError as exc:
            mesh_error = str(exc)
    config = dict(version=VERSION, action=action, size=size,
                  training_run=str(training_run) if training_run else None,
                  checkpoint_hash=checkpoint_hash,
                  reference=str(reference) if reference else None,
                  reference_hash=sha(reference) if reference else None,
                  keys=[r['key'] for r in rows], paths=[r['path'] for r in rows])
    run = Path(run) if run else new_run('processing')
    if resume:
        if load(run / 'config.json') != config:
            raise ValueError('Retomada incompativel com configuracao ou checkpoint')
    else:
        if (run / 'config.json').exists():
            raise ValueError('Execucao existente; use retomar ou uma pasta nova')
        save(run / 'config.json', config)
        save(run / 'inputs.json', rows)
    ref_map = {}
    if reference:
        from .data import validate_references
        for item in validate_references(reference)['samples']:
            ref_map[item['key']] = item
            for alias in item['bank_keys']:
                ref_map[alias] = item
    results = []
    save(run / 'status.json', dict(state='RODANDO', completed=0, total=len(rows)))
    try:
        for index, row in enumerate(rows):
            if stop and Path(stop).exists():
                break
            folder = run / row['key']
            record_path = folder / 'result.json'
            token = digest([config, row, sha(row['path']) if Path(row['path']).is_file() else None])
            if resume and record_path.exists():
                old = load(record_path)
                valid = old.get('fingerprint') == token and old.get('status') != 'ERRO'
                valid = valid and all(Path(p).is_file() and sha(p) == h for p, h in old.get('assets', {}).items())
                if valid:
                    results.append(old)
                    log(f'[{index+1}/{len(rows)}] RETOMADO {row["filename"]}')
                    continue
            result = dict(item=row, fingerprint=token, action=action, points=[], regions={},
                          input_hash=sha(row['path']) if Path(row['path']).is_file() else None,
                          contours=[], assets={}, status='OK_TECNICO', reason='')
            assets = []
            try:
                im = image(row['path'])
                h, w = im.shape[:2]
                if action == 'normalize':
                    normalized, matrix = letterbox(im, size)
                    target = folder / 'normalizada.png'
                    write_image(target, normalized)
                    assets.append(target)
                    result.update(image=str(target), transform=matrix.tolist(), original_size=[w, h])
                elif action == 'apply':
                    from .learning import predict
                    result['points'] = predict(im, model, state)
                    result['regions'] = regions_for(result['points'], w, h)
                    result['status'] = 'PROPOSTA_MODELO'
                elif action == 'mesh':
                    if mesh_error:
                        raise ValueError('MEDIAPIPE_INDISPONIVEL: ' + mesh_error)
                    detection = mesh.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    if not detection.multi_face_landmarks:
                        result.update(status='ATENCAO', reason='FACE_NAO_DETECTADA')
                    else:
                        landmarks = detection.multi_face_landmarks[0].landmark
                        result['mesh'] = [dict(idx=i, x=float(p.x*w), y=float(p.y*h), z=float(p.z))
                                          for i, p in enumerate(landmarks)]
                        result['points'] = [dict(numero=n+1, x=float(landmarks[i].x*w),
                                                 y=float(landmarks[i].y*h), estado='MEDIAPIPE_CANDIDATO',
                                                 visibility='NAO_AVALIADO') for n, i in enumerate(MESH_MAP)]
                        for edges in (mp.solutions.face_mesh.FACEMESH_FACE_OVAL,
                                      mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
                                      mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
                                      mp.solutions.face_mesh.FACEMESH_LIPS):
                            result['contours'].extend([[[landmarks[a].x*w, landmarks[a].y*h],
                                                       [landmarks[b].x*w, landmarks[b].y*h]] for a, b in edges])
                        result['regions'] = regions_for(result['points'], w, h)
                        result.update(status='PROPOSTA_MEDIAPIPE', mediapipe_version=mp.__version__,
                                      actual_points=len(landmarks),
                                      parameters=dict(static_image_mode=True, refine_landmarks=True))
                elif action == 'regions':
                    ref = ref_map.get(row['key'])
                    if ref is None or sha(row['path']) != ref['hash']:
                        raise ValueError('Referencia vinculada por hash indisponivel')
                    result['points'] = ref['points']
                    result['regions'] = regions_for(ref['points'], w, h)
                    result['status'] = 'PROPOSTA_IA'
                else:
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    result['contours'] = [c.reshape(-1, 2).tolist() for c in contours]
                    target = folder / 'mask.png'
                    write_image(target, mask)
                    assets.append(target)
                    result.update(status='ATENCAO', reason='PRIMEIRO_PLANO_CANDIDATO' if contours else 'MASCARA_VAZIA',
                                  border_contact=bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()))
                preview = folder / 'preview.png'
                marked = overlay(im, result['points'], result['regions'], result['contours'])
                write_image(preview, letterbox(marked)[0])
                assets.append(preview)
                result['preview'] = str(preview)
            except Exception as exc:
                result.update(status='ERRO', reason=str(exc))
            result['assets'] = {str(p): sha(p) for p in assets}
            save(record_path, result)
            results.append(result)
            save(run / 'status.json', dict(state='RODANDO', completed=len(results), total=len(rows)))
            log(f'[{index+1}/{len(rows)}] {row["filename"]}: {result["status"]} {result["reason"]}')
    finally:
        if mesh:
            mesh.close()
    save(run / 'results.json', results)
    errors = sum(r['status'] == 'ERRO' for r in results)
    save(run / 'status.json', dict(state='INTERROMPIDO' if len(results) < len(rows) else
                                   ('CONCLUIDO_COM_ERROS' if errors else 'CONCLUIDO'),
                                   completed=len(results), total=len(rows), errors=errors))
    log(f'Saidas: {run}; erros: {errors}/{len(results)}')
    return run
