"""Shared CLI and background worker for the DINOv2 panel."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Treinamento_DINOv2_face_cranio.portable import Path
from Treinamento_DINOv2_face_cranio import file_lock as fcntl
from Treinamento_DINOv2_face_cranio.core import ARTIFACTS, load, new_run, save


def execute(action, args, stop=None):
    if action == 'retrieval_pipeline':
        from Treinamento_DINOv2_face_cranio.retrieval_pipeline import run_pipeline
        return run_pipeline(**args,stop=stop)
    if action == 'matching_breakdown':
        from Treinamento_DINOv2_face_cranio.matching_breakdown import build
        return build(**args,allow_inference=True,stop=stop)
    if action == 'blend_models':
        from Treinamento_DINOv2_face_cranio.rank_blend import search
        return search(**args, stop=stop)
    if action == 'quality_audit':
        from Treinamento_DINOv2_face_cranio.quality import audit
        return audit(**args, stop=stop)
    if action == 'improvement_pipeline':
        from Treinamento_DINOv2_face_cranio.improvement_pipeline import run_pipeline
        return run_pipeline(**args,stop=stop)
    if action == 'search_weights':
        from Treinamento_DINOv2_face_cranio.weight_search import search_weights
        return search_weights(**args, stop=stop)
    if action in ('train_cranial_sex', 'apply_cranial_sex'):
        from Laboratorio_visual_face_cranio import cranial_sex
        function = cranial_sex.train if action == 'train_cranial_sex' else cranial_sex.apply
        return function(**args, stop=stop, log=lambda message: print(message, flush=True))
    if action == 'visual_report':
        from Treinamento_DINOv2_face_cranio.visuals import visual_report
        return visual_report(**args, stop=stop)
    if action in ('train_components', 'train_combinations', 'all_combinations', 'fuse', 'compare_fusion'):
        from Treinamento_DINOv2_face_cranio import fusion
        return getattr(fusion, action)(**args, stop=stop)
    from Treinamento_DINOv2_face_cranio import backbone, data, engine
    if action == 'inventory':
        return data.inventory(**args, stop=stop)
    if action == 'prepare':
        return backbone.prepare_variant(args.get('backbone', 'dinov2_vits14'))
    if action == 'extract':
        return backbone.extract(**args, stop=stop)
    if action in ('train', 'suite', 'evaluate'):
        return getattr(engine, action)(**args, stop=stop)
    if action == 'freeze':
        return engine.freeze(**args)
    if action in ('compare', 'explain'):
        target = new_run('comparisons' if action == 'compare' else 'explanations')
        save(target/'inputs.json', args)
        result = getattr(engine, action)(**args, stop=stop)
        save(target/'result.json', result)
        return target
    raise ValueError('Acao desconhecida: '+action)


def main():
    p = argparse.ArgumentParser(description='DINOv2: etapas isoladas, sem pontos obrigatorios')
    p.add_argument('action', nargs='?', choices=['inventory', 'prepare', 'extract', 'train', 'suite', 'evaluate', 'compare', 'explain', 'freeze', 'train_components', 'train_combinations', 'all_combinations', 'fuse', 'compare_fusion', 'visual_report', 'search_weights', 'improvement_pipeline', 'blend_models', 'quality_audit', 'matching_breakdown', 'retrieval_pipeline'])
    p.add_argument('--args', type=Path, help='Parametros da etapa em JSON')
    p.add_argument('--request', type=Path, help='Request do job: action e args')
    p.add_argument('--run-dir', type=str)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--force', action='store_true', help='Criar nova execucao preservando a anterior')
    options = p.parse_args()
    if options.resume and options.force:
        p.error('--resume e --force sao exclusivos')
    spec = load(options.request) if options.request else dict(action=options.action, args=load(options.args) if options.args else {})
    if not spec['action']:
        p.error('Informe uma etapa ou --request')
    args = spec.setdefault('args', {})
    if options.run_dir:
        args['resume_dir' if spec['action'] == 'suite' else 'run_dir'] = options.run_dir
    if options.resume and not (options.run_dir or args.get('run_dir') or args.get('resume_dir')):
        p.error('--resume requer a pasta da execucao')
    if options.resume and spec['action'] in ('train', 'extract'):
        args['resume'] = True
    if options.force:
        args.pop('run_dir', None); args.pop('resume_dir', None); args.pop('resume', None)
    job = options.request.parent if options.request else new_run('jobs')
    if spec['action'] in ('train_components', 'train_combinations') and not args.get('run_dir'):
        args['run_dir'] = str(new_run('fusion_training'))
    if spec['action'] == 'improvement_pipeline' and not args.get('run_dir'):
        args['run_dir'] = str(new_run('improvement_pipelines'))
    if spec['action'] == 'retrieval_pipeline' and not args.get('run_dir'):
        args['run_dir'] = str(new_run('retrieval_pipelines'))
    # Reserve a durable output before starting so even a failed job can resume.
    if spec['action'] in ('train', 'extract') and not args.get('run_dir'):
        args['run_dir'] = str(new_run('runs' if spec['action'] == 'train' else 'extractions'))
    if spec['action'] == 'suite' and not args.get('resume_dir'):
        suite_dir = new_run('suites')
        save(suite_dir/'request.json', dict(**args))
        args['resume_dir'] = str(suite_dir)
        # suite() normalizes omitted defaults before validating its request.
        saved = {k: v for k, v in args.items() if k != 'resume_dir'}
        saved.setdefault('experiment', 'E1'); saved.setdefault('seeds', [42]); saved.setdefault('mode', 'matched')
        save(suite_dir/'request.json', saved)
    save(job/'request.json', spec)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (ARTIFACTS/'worker.lock').open('a') as lock:
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                save(job/'status.json', dict(state='OCUPADO',
                     message='Outra execucao esta em andamento. Esta tentativa nao foi iniciada.', duration=time.time()-started))
                print('Outra execucao esta em andamento; tentativa duplicada nao iniciada.', flush=True)
                return 0
            save(job/'process.json', dict(pid=os.getpid(), started=started))
            save(job/'status.json', dict(state='RODANDO'))
            output = execute(spec['action'], args, job/'stop.flag')
            state = load(Path(output)/'status.json')['state'] if (Path(output)/'status.json').exists() else 'CONCLUIDO'
            if state == 'CONCLUIDO' and spec['action'] == 'compare_fusion':
                from Treinamento_DINOv2_face_cranio.matching_breakdown import build
                try:
                    build(output,allow_inference=True,stop=job/'stop.flag')
                except (InterruptedError,ValueError,OSError,RuntimeError) as exc:
                    save(Path(output)/'breakdown_warning.json',dict(error=str(exc)))
                    print('Ranking principal salvo; tabela por vistas incompleta: '+str(exc),flush=True)
            if state == 'CONCLUIDO' and spec['action'] in ('train', 'suite', 'train_components', 'train_combinations', 'compare', 'compare_fusion'):
                from Treinamento_DINOv2_face_cranio.visuals import visual_report, matching_maps
                try:
                    print('Gerando mapas visuais por oclusao (nao altera pesos ou ranking).', flush=True)
                    if spec['action'] in ('compare', 'compare_fusion'):
                        matching_maps(output, combined=spec['action']=='compare_fusion', stop=job/'stop.flag')
                    else:
                        visual_report(output, stop=job/'stop.flag')
                except (InterruptedError, ValueError, OSError, RuntimeError) as exc:
                    save(Path(output)/'visual_warning.json', dict(error=str(exc)))
                    print('Resultado principal salvo; mapas incompletos: '+str(exc), flush=True)
            save(job/'status.json', dict(state=state, output=str(output), duration=time.time()-started))
            print('Resultado: '+str(output), flush=True)
            return 0
        except InterruptedError:
            save(job/'status.json', dict(state='INTERROMPIDO', duration=time.time()-started,
                 output=args.get('resume_dir', args.get('run_dir'))))
            return 0
        except Exception as exc:
            traceback.print_exc()
            save(job/'status.json', dict(state='ERRO', error=str(exc), duration=time.time()-started,
                                         output=args.get('resume_dir', args.get('run_dir'))))
            return 1


if __name__ == '__main__':
    sys.exit(main())
