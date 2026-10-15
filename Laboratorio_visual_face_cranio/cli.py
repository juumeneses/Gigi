"""All UI actions call this CLI with a persisted JSON request."""
from __future__ import annotations

import argparse
import sys
import traceback
import fcntl
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Laboratorio_visual_face_cranio.core import ARTIFACTS, load, save


def execute(spec, stop=None):
    action = spec['action']
    args = spec.get('args', {})
    if action == 'prepare':
        from Laboratorio_visual_face_cranio.data import prepare
        return prepare(**args)
    if action == 'train':
        from Laboratorio_visual_face_cranio.learning import train
        return train(**args, stop=stop)
    if action == 'evaluate':
        from Laboratorio_visual_face_cranio.learning import evaluate
        evaluate(**args)
        return Path(args['run'])
    if action == 'process':
        from Laboratorio_visual_face_cranio.processing import process
        return process(**args, stop=stop)
    if action in ('expand', 'promote'):
        from Laboratorio_visual_face_cranio.references import expand, promote
        return expand(**args, stop=stop) if action == 'expand' else promote(**args)
    if action in ('train_matching', 'evaluate_matching', 'compare_matching'):
        from Laboratorio_visual_face_cranio.matching import train, evaluate, compare
        if action == 'train_matching':
            return train(**args, stop=stop)
        if action == 'evaluate_matching':
            return evaluate(**args, stop=stop)
        from Laboratorio_visual_face_cranio.core import new_run
        output = new_run('matching_comparisons')
        save(output/'inputs.json', args)
        save(output/'result.json', compare(**args, stop=stop))
        save(output/'inputs.json', args)
        return output
    if action in ('train_cranial_sex', 'apply_cranial_sex'):
        from Laboratorio_visual_face_cranio.cranial_sex import train, apply
        return train(**args, stop=stop) if action == 'train_cranial_sex' else apply(**args, stop=stop)
    raise ValueError('Acao desconhecida: ' + action)


def main():
    parser = argparse.ArgumentParser(description='Laboratorio: comandos definidos em JSON; ver README')
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    job = args.request.parent
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    lock = (ARTIFACTS / 'worker.lock').open('a')
    started = time.time()
    save(job / 'status.json', dict(state='RODANDO'))
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Outra execucao esta em andamento; aguarde ou solicite a parada.') from None
        save(job / 'process.json', dict(pid=os.getpid(), started_at=started))
        output = execute(load(args.request), job / 'stop.flag')
        result = dict(state='CONCLUIDO', output=str(output), duration_seconds=round(time.time()-started, 2))
        if (Path(output) / 'status.json').exists():
            result['state'] = load(Path(output) / 'status.json')['state']
        save(job / 'status.json', result)
        print('Resultado:', output, flush=True)
    except Exception as exc:
        traceback.print_exc()
        save(job / 'status.json', dict(state='ERRO', error=str(exc), duration_seconds=round(time.time()-started, 2)))
        return 1
    finally:
        lock.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
