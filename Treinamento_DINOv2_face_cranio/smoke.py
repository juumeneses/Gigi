"""Small real-image integration check; never a scientific benchmark."""
from .core import ARTIFACTS, load, save, sha, new_run
from .data import cohort
from .engine import suite, sample_gallery, compare, explain


def main():
    manifest = sorted((ARTIFACTS/'inventories').glob('*/manifest.json'))[-1]
    eligible = cohort(manifest, 'E6')['people']
    selected = [p for p in eligible if p['split'] == 'treino'][:8] + [p for p in eligible if p['split'] == 'validacao'][:4]
    hashes = {v['path']: sha(v['path']) for p in selected for v in p['views'].values()}
    run = suite(str(manifest), experiment='E6', seeds=[42], ids=[p['person'] for p in selected],
                epochs=3, batch=4, smoke=True, device='cpu')
    model = run/'E6_seed42'
    gallery = sample_gallery([p for p in selected if p['split'] == 'validacao'], count=4, seed=42)
    result = compare(model, **gallery)
    out = new_run('comparisons')
    save(out/'inputs.json', dict(run=str(model), **gallery))
    save(out/'result.json', result)
    target = new_run('explanations')
    candidate = gallery['candidates'][0]
    save(target/'inputs.json', dict(run=str(model), query=gallery['query'], candidate=candidate))
    save(target/'result.json', explain(model, gallery['query'], candidate, grid=3))
    intact = all(sha(p) == h for p, h in hashes.items())
    assert intact
    report = dict(suite=str(run), train=8, validation=4, epochs=3, originals_unchanged=intact,
                  comparison=str(out), explanation=str(target), scientific_benchmark=False,
                  evaluations={exp: load(run/f'{exp}_seed42/evaluation.json')['full'] for exp in ('E0', 'E1', 'E6')})
    save(ARTIFACTS/'smoke_report.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
