"""Shared gallery inference and frozen-backbone two-tower experiments."""
from itertools import combinations
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .core import ARTIFACTS, GROUPS, coordinates, digest, image, letterbox, load, new_run, regions_for, save, sha
from .learning import checkpoint_load, checkpoint_save, predictor, predict
from .references import paired

PAIRS = list(combinations(range(11), 2))
FACE_VIEWS = ['FFN', 'FFS', 'FL']
SKULL_VIEWS = ['CF', 'CE', 'CD']
WEIGHTS = Path.home()/'.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'


def geometry(points, width, height):
    xy = coordinates(points)
    diagonal = np.hypot(width, height)
    values = np.zeros(55, np.float32)
    mask = np.zeros(55, bool)
    for k, (a, b) in enumerate(PAIRS):
        if np.isfinite(xy[[a, b]]).all():
            values[k] = np.linalg.norm(xy[a]-xy[b])/diagonal
            mask[k] = True
    return values, mask


def sample_gallery(items, count=10, seed=42):
    items = sorted(items, key=lambda r: r['person'])
    if count < 2 or count > len(items) or len({r['person'] for r in items}) != len(items):
        raise ValueError('Escolha de 2 ate o total de pessoas distintas disponiveis')
    rng = random.Random(seed)
    query = rng.choice(items)
    candidates = rng.sample([r for r in items if r['person'] != query['person']], count-1) + [query]
    rng.shuffle(candidates)
    return dict(query=query, candidates=candidates, seed=seed)


def retrieval(scores, ids):
    ranks = []
    for i, row in enumerate(scores):
        if not np.isfinite(row[i]):
            ranks.append(None)
            continue
        order = sorted(range(len(ids)), key=lambda j: (-float(row[j]) if np.isfinite(row[j]) else float('inf'), ids[j]))
        ranks.append(order.index(i)+1)
    return rank_metrics(ranks)


def rank_metrics(ranks):
    rr = np.array([1/r if r else 0 for r in ranks])
    rng = np.random.default_rng(42)
    samples = [rr[rng.integers(len(rr), size=len(rr))].mean() for _ in range(500)] if len(rr) else [0]
    return dict(n=len(ranks), ranks=ranks, mrr=float(rr.mean()) if len(rr) else 0,
                mrr_ci95=np.quantile(samples, [.025, .975]).tolist(),
                **{f'recall_{k}': sum(r is not None and r <= k for r in ranks)/max(1, len(ranks)) for k in [1, 3, 5, 10]})


def triplet_semihard(face, skull, margin=.2):
    distances = 1-face@skull.T
    if len(face) < 2:
        raise ValueError('Triplet precisa de duas identidades distintas no lote')
    losses = []
    for matrix in (distances, distances.T):
        for i in range(len(matrix)):
            pos = matrix[i, i]
            negatives = torch.cat([matrix[i, :i], matrix[i, i+1:]])
            eligible = negatives[(negatives > pos) & (negatives < pos+margin)]
            negative = eligible.min() if len(eligible) else negatives.min()
            losses.append(F.relu(pos-negative+margin))
    return torch.stack(losses).mean()


class Towers(nn.Module):
    def __init__(self, kind='image', dimension=64, regional=False):
        super().__init__()
        from Modelo_novo.src.model import TwoTower
        self.visual = TwoTower(embedding_dim=dimension, pretrained=False, freeze_backbone=True, dropout=0.)
        if not WEIGHTS.exists():
            raise ValueError('Pesos ResNet18 locais ausentes: ' + str(WEIGHTS))
        weights = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
        weights = {k: v for k, v in weights.items() if not k.startswith('fc.')}
        for encoder in (self.visual.face_encoder, self.visual.skull_encoder):
            encoder.backbone.load_state_dict(weights)
            if regional:
                encoder.projection = nn.Sequential(nn.Linear(1029, dimension), nn.LayerNorm(dimension))
        self.kind = kind
        self.regional = regional
        self.face_geometry = nn.Sequential(nn.Linear(110, 128), nn.GELU(), nn.Linear(128, dimension))
        self.skull_geometry = nn.Sequential(nn.Linear(110, 128), nn.GELU(), nn.Linear(128, dimension))
        self.face_fusion = nn.Linear(2*dimension, dimension)
        self.skull_fusion = nn.Linear(2*dimension, dimension)

    def encode(self, features, views, geom, modality):
        encoder = self.visual.face_encoder if modality == 'face' else self.visual.skull_encoder
        values = encoder.projection(features)
        values = (values*views[..., None]).sum(1)/views.sum(1).clamp_min(1)[..., None]
        if self.kind == 'hybrid':
            branch = self.face_geometry if modality == 'face' else self.skull_geometry
            fusion = self.face_fusion if modality == 'face' else self.skull_fusion
            values = fusion(torch.cat([values, branch(geom)], -1))
        return F.normalize(values, dim=-1)

    def head_state(self):
        return {k: v for k, v in self.state_dict().items() if '.backbone.' not in k}

    def load_heads(self, values):
        missing, unexpected = self.load_state_dict(values, strict=False)
        if unexpected or any('.backbone.' not in k for k in missing):
            raise ValueError('Cabecas do checkpoint incompativeis')


def pixel_tensor(im, size=224):
    canvas = letterbox(im, size)[0]
    x = torch.tensor(canvas[:, :, ::-1].copy().transpose(2, 0, 1), dtype=torch.float32)/255
    return (x-torch.tensor([.485, .456, .406])[:, None, None])/torch.tensor([.229, .224, .225])[:, None, None]


def extract(item, modality, localizer, visual=None, multiview=False, regional=False, exclusions=None):
    names = FACE_VIEWS if modality == 'face' else SKULL_VIEWS
    if not multiview:
        names = names[:1]
    feats, views, points, masks = [], [], {}, []
    geom, available = np.zeros(55, np.float32), np.zeros(55, bool)
    failures = []
    for view in names:
        row = item['views'].get(view)
        embedding = np.zeros(1029 if regional else 512, np.float32)
        present = 0.
        if row:
            try:
                if row.get('hash') and row['hash'] != sha(row['path']):
                    raise ValueError('Hash alterado')
                im = image(row['path'])
                current = []
                if localizer and view == names[0]:
                    current = predict(im, *localizer)
                    # Human visibility exclusions are tied to this exact image, not guessed by the model.
                    excluded = set(row.get('excluded_points', [])) | set((exclusions or {}).get(sha(row['path']), []))
                    for p in current:
                        if p['numero'] in excluded:
                            p.update(x=None, y=None, estado='AUSENCIA_REVISADA')
                    geom, available = geometry(current, im.shape[1], im.shape[0])
                points[view] = current
                regions = regions_for(current, im.shape[1], im.shape[0])
                if visual:
                    encoder = visual.visual.face_encoder if modality == 'face' else visual.visual.skull_encoder
                    with torch.no_grad():
                        base = encoder.backbone(pixel_tensor(im)[None])[0].numpy()
                        embedding = base
                        if regional:
                            crops, region_mask = [], []
                            for name in GROUPS:
                                box = regions[name].get('box')
                                if box:
                                    x1, y1, x2, y2 = map(int, box)
                                    crop = im[y1:y2, x1:x2]
                                    if crop.size:
                                        crops.append(encoder.backbone(pixel_tensor(crop)[None])[0].numpy())
                                        region_mask.append(1.)
                                        continue
                                region_mask.append(0.)
                            local = np.mean(crops, axis=0) if crops else np.zeros(512)
                            embedding = np.r_[base, local, region_mask].astype(np.float32)
                present = 1.
            except (ValueError, OSError) as exc:
                failures.append(dict(view=view, error=str(exc)))
        feats.append(embedding)
        views.append(present)
        masks.append(view)
    return dict(features=np.stack(feats).tolist(), views=views, view_names=masks,
                geometry=geom.tolist(), availability=available.tolist(), points=points, failures=failures)


def localizers(config):
    result = {}
    for modality in ['face', 'cranio']:
        path = config.get(modality+'_run')
        if path:
            expected = 'FFN' if modality == 'face' else 'CF'
            model, state = predictor(path)
            if state['config']['modality'] != modality or state['config']['view'] != expected:
                raise ValueError('Localizador incompativel')
            result[modality] = (model, state)
        else:
            result[modality] = None
    return result


def standardize(items, modality):
    values = np.array([r[modality]['geometry'] for r in items], np.float32)
    mask = np.array([r[modality]['availability'] for r in items], bool)
    count = mask.sum(0)
    mean = (values*mask).sum(0)/np.maximum(count, 1)
    variance = (((values-mean)**2)*mask).sum(0)/np.maximum(count, 1)
    return dict(mean=mean.tolist(), scale=np.maximum(np.sqrt(variance), .001).tolist(), supported=(count > 0).tolist())


def geometry_tensor(items, modality, stats):
    values = np.array([r[modality]['geometry'] for r in items], np.float32)
    mask = np.array([r[modality]['availability'] for r in items], bool) & np.array(stats['supported'])
    values = ((values-np.array(stats['mean']))/np.array(stats['scale']))*mask
    return torch.tensor(np.c_[values, mask.astype(float)], dtype=torch.float32)


def encode_items(model, items, modality, stats):
    values = torch.tensor(np.array([r[modality]['features'] for r in items]), dtype=torch.float32)
    views = torch.tensor(np.array([r[modality]['views'] for r in items]), dtype=torch.float32)
    return model.encode(values, views, geometry_tensor(items, modality, stats), modality)


def ridge_scores(query, candidates, state, coverage=.6):
    f = geometry_tensor(query, 'face', state['stats']['face']).numpy()
    predicted = np.c_[f, np.ones(len(f))]@np.asarray(state['ridge'])
    target = geometry_tensor(candidates, 'cranio', state['stats']['cranio']).numpy()[:, :55]
    scores = np.full((len(query), len(candidates)), -np.inf)
    for i, face in enumerate(query):
        fm = np.array(face['face']['availability']) & np.array(state['stats']['face']['supported'])
        for j, skull in enumerate(candidates):
            sm = np.array(skull['cranio']['availability']) & np.array(state['stats']['cranio']['supported'])
            common = fm & sm
            if common.sum() >= 5 and common.sum()/max(1, fm.sum()) >= coverage:
                scores[i, j] = -float(np.sqrt(np.mean((predicted[i, common]-target[j, common])**2)))
    return scores


def scores_for(items, state, model=None, queries=None):
    queries = queries or items
    if state['config']['kind'] == 'geometry':
        scores = ridge_scores(queries, items, state)
    else:
        with torch.no_grad():
            a = encode_items(model, queries, 'face', state['stats']['face'])
            b = encode_items(model, items, 'cranio', state['stats']['cranio'])
        scores = (a@b.T).numpy()
    for i, row in enumerate(queries):
        if not row['face']['views'][0]:
            scores[i, :] = -np.inf
    for j, row in enumerate(items):
        if not row['cranio']['views'][0]:
            scores[:, j] = -np.inf
    return scores


def validate_items(items):
    hashes = {}
    people = set()
    for item in items:
        if item['person'] in people:
            raise ValueError('Identidade duplicada no manifesto')
        people.add(item['person'])
        for row in item['views'].values():
            token = sha(row['path'])
            if row.get('hash') and row['hash'] != token:
                raise ValueError('Imagem alterada desde a preparacao')
            if token in hashes and hashes[token] != item['split']:
                raise ValueError('Mesma imagem em splits diferentes')
            hashes[token] = item['split']


def train(rows, face_run=None, cranio_run=None, kind='image', loss='infonce', multiview=False,
          regional=False, epochs=100, lr=.0003, batch_size=16, seed=42, dimension=64,
          temperature=.07, margin=.2, patience=10, ridge_alpha=10., resume=None, stop=None, log=print):
    if kind not in ('geometry', 'image', 'hybrid') or loss not in ('infonce', 'triplet'):
        raise ValueError('Experimento desconhecido')
    if epochs < 1 or batch_size < 2 or lr <= 0 or temperature <= 0 or margin <= 0:
        raise ValueError('Parametros invalidos')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    people = [p for p in paired(rows) if p['split'] in ('treino', 'validacao')]
    exclusions = {}
    for review_path in sorted((ARTIFACTS/'point_reviews').glob('*/review.json')):
        decision = load(review_path)
        exclusions[decision['image_hash']] = [p['numero'] for p in decision['points'] if p['visibility'] != 'VISIVEL']
    if kind != 'image' or regional:
        if not face_run or not cranio_run:
            raise ValueError('Geometria/regioes requerem os dois localizadores frontais')
    for p in people:
        for row in p['views'].values():
            row['hash'] = sha(row['path'])
    validate_items(people)
    config = dict(version='matching_lab_v1', kind=kind, loss=loss, multiview=multiview, regional=regional,
                  epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, dimension=dimension,
                  temperature=temperature, margin=margin, patience=patience, ridge_alpha=ridge_alpha,
                  face_run=face_run, cranio_run=cranio_run,
                  exclusions=exclusions,
                  localizer_hashes={m: sha(Path(p)/'best.pt') if p else None for m, p in [('face', face_run), ('cranio', cranio_run)]},
                  weights_hash=sha(WEIGHTS) if kind != 'geometry' else None,
                  input_fingerprint=digest(people), train_ids=[p['person'] for p in people if p['split'] == 'treino'],
                  validation_ids=[p['person'] for p in people if p['split'] == 'validacao'])
    protocol = {k: v for k, v in config.items() if k != 'epochs'}
    config['fingerprint'] = digest(protocol)
    run = Path(resume) if resume else new_run('matching')
    if resume and load(run/'config.json')['fingerprint'] != config['fingerprint']:
        raise ValueError('Retomada exige os mesmos dados, pesos e parametros')
    save(run/'config.json', config)
    save(run/'inputs.json', people)
    save(run/'inputs_rows.json', rows)
    sink = log
    def log(message):
        with (run/'execution.log').open('a') as stream:
            stream.write(message+'\n')
        sink(message)
    model = Towers(kind, dimension, regional) if kind != 'geometry' else None
    if model:
        model.eval()
    loc = localizers(config)
    # A localizer cannot have learned labels of matching validation identities.
    train_ids = set(config['train_ids'])
    for value in loc.values():
        if value and not set(value[1]['config']['train_ids']).issubset(train_ids):
            raise ValueError('Localizador treinado fora do split de treino deste matching')
    cache, failures = [], []
    for i, item in enumerate(people):
        if stop and Path(stop).exists():
            save(run/'status.json', dict(state='INTERROMPIDO', stage='features'))
            return run
        path = run/'features'/f'{digest(item["person"])[:20]}.json'
        integrity = path.with_suffix('.sha.json')
        if resume and path.exists() and integrity.exists() and sha(path) == load(integrity)['sha256']:
            result = load(path)
        else:
            result = dict(person=item['person'], year=item['year'], split=item['split'])
            for modality in ['face', 'cranio']:
                result[modality] = extract(item, modality, loc[modality], model, multiview, regional, exclusions)
            save(path, result)
            save(integrity, dict(sha256=sha(path)))
        if not result['face']['views'][0] or not result['cranio']['views'][0]:
            failures.append(dict(person=item['person'], reason='VISTA_FRONTAL_ILEGIVEL', details=[result['face']['failures'], result['cranio']['failures']]))
            if result['split'] == 'validacao':
                cache.append(result)
        else:
            cache.append(result)
        log(f'Features {i+1}/{len(people)}: {item["person"]}')
    save(run/'features.json', cache)
    save(run/'failures.json', failures)
    tr = [r for r in cache if r['split'] == 'treino']
    va = [r for r in cache if r['split'] == 'validacao']
    if len(tr) < 2 or sum(bool(r['face']['views'][0] and r['cranio']['views'][0]) for r in va) < 2:
        raise ValueError('Matching requer pelo menos duas identidades de treino e duas de validacao')
    stats = {m: standardize(tr, m) for m in ['face', 'cranio']}
    diagnostic = []
    for k, (a, b) in enumerate(PAIRS):
        positive, negative = [], []
        for i, record in enumerate(tr):
            other = tr[(i+1) % len(tr)]
            if record['face']['availability'][k] and record['cranio']['availability'][k]:
                positive.append(abs(record['face']['geometry'][k]-record['cranio']['geometry'][k]))
            if record['face']['availability'][k] and other['cranio']['availability'][k]:
                negative.append(abs(record['face']['geometry'][k]-other['cranio']['geometry'][k]))
        diagnostic.append(dict(points=[a+1, b+1], n_positive=len(positive), n_negative=len(negative),
                               positive_median=float(np.median(positive)) if positive else None,
                               negative_median=float(np.median(negative)) if negative else None,
                               interpretation='Diferenca entre medidas; nao erro anatomico'))
    save(run/'regional_diagnostic.json', diagnostic)
    state = dict(config=config, stats=stats)
    ids = [r['person'] for r in va]
    history, start, best, stale = [], 0, -1., 0
    if kind == 'geometry':
        f = geometry_tensor(tr, 'face', stats['face']).numpy()
        s = geometry_tensor(tr, 'cranio', stats['cranio']).numpy()[:, :55]
        mask = np.array([r['cranio']['availability'] for r in tr])
        X = np.c_[f, np.ones(len(f))]
        coefficients = np.zeros((111, 55))
        penalty = np.eye(111)*ridge_alpha
        penalty[-1, -1] = 0
        for j in range(55):
            keep = mask[:, j]
            if keep.any():
                coefficients[:, j] = np.linalg.lstsq(X[keep].T@X[keep]+penalty, X[keep].T@s[keep, j], rcond=None)[0]
        state['ridge'] = torch.tensor(coefficients)
        metrics = retrieval(scores_for(va, state), ids)
        state.update(epoch=1, metrics=metrics)
        checkpoint_save(run/'best.pt', state)
        save(run/'evaluation.json', dict(**metrics, split='validacao', failures=failures))
    else:
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=.0001)
        if resume and (run/'last.pt').exists():
            old = checkpoint_load(run/'last.pt')
            model.load_heads(old['heads'])
            optimizer.load_state_dict(old['optimizer'])
            start, best, stale, history = old['epoch'], old['best'], old['stale'], old['history']
            if epochs <= start:
                raise ValueError('Total de epocas deve superar a ultima epoca salva')
        for epoch in range(start, epochs):
            if stop and Path(stop).exists():
                break
            torch.manual_seed(seed+epoch)
            model.train()
            order = torch.randperm(len(tr)).tolist()
            batches = [order[j:j+batch_size] for j in range(0, len(order), batch_size)]
            if len(batches) > 1 and len(batches[-1]) == 1:
                batches[-2].extend(batches.pop())
            losses = []
            for indices in batches:
                batch = [tr[j] for j in indices]
                f = encode_items(model, batch, 'face', stats['face'])
                s = encode_items(model, batch, 'cranio', stats['cranio'])
                if loss == 'infonce':
                    from Modelo_novo.src.model import symmetric_infonce
                    value = symmetric_infonce(f, s, temperature)
                else:
                    value = triplet_semihard(f, s, margin)
                if not torch.isfinite(value):
                    raise ValueError('Perda nao finita')
                optimizer.zero_grad()
                value.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(float(value.detach()))
            model.eval()
            metrics = retrieval(scores_for(va, state, model), ids)
            improved = metrics['mrr'] > best
            best, stale = (metrics['mrr'], 0) if improved else (best, stale+1)
            history.append(dict(epoch=epoch+1, loss=float(np.mean(losses)), val_mrr=metrics['mrr']))
            snapshot = dict(**state, heads=model.head_state(), optimizer=optimizer.state_dict(), epoch=epoch+1,
                            best=best, stale=stale, history=history, metrics=metrics)
            checkpoint_save(run/'last.pt', snapshot)
            if improved:
                checkpoint_save(run/'best.pt', snapshot)
                save(run/'evaluation.json', dict(**metrics, split='validacao', failures=failures, epoch=epoch+1))
            save(run/'history.json', history)
            save(run/'status.json', dict(state='RODANDO', epoch=epoch+1, target_epochs=epochs))
            log(f'Epoca {epoch+1}: loss={np.mean(losses):.5f}; validacao MRR={metrics["mrr"]:.4f}')
            if stale >= patience:
                break
    save(run/'status.json', dict(state='INTERROMPIDO' if stop and Path(stop).exists() else 'CONCLUIDO'))
    return run


def load_model(run):
    state = checkpoint_load(Path(run)/'best.pt')
    config = state['config']
    if config['weights_hash'] and sha(WEIGHTS) != config['weights_hash']:
        raise ValueError('Pesos do backbone mudaram')
    for m in ['face', 'cranio']:
        p = config[m+'_run']
        if p and sha(Path(p)/'best.pt') != config['localizer_hashes'][m]:
            raise ValueError('Localizador mudou desde o treino de matching')
    model = None
    if config['kind'] != 'geometry':
        model = Towers(config['kind'], config['dimension'], config['regional'])
        model.load_heads(state['heads'])
        model.eval()
    return state, model


def compare(run, query, candidates, reference_sex=None, candidate_sexes=None, stop=None, log=print):
    if not candidates or len({r['person'] for r in candidates}) != len(candidates):
        raise ValueError('Galeria vazia ou IDs de candidatos repetidos')
    for item in [query]+candidates:
        for row in item['views'].values():
            if row.get('hash') and row['hash'] != sha(row['path']):
                raise ValueError('Imagem mudou desde a selecao: ' + row['path'])
            row['hash'] = sha(row['path'])
    state, model = load_model(run)
    config = state['config']
    loc = localizers(config)
    q = dict(person=query['person'], face=extract(query, 'face', loc['face'], model, config['multiview'], config['regional'], config.get('exclusions')))
    items = []
    for item in candidates:
        if stop and Path(stop).exists():
            raise ValueError('Comparacao interrompida; ranking incompleto nao publicado')
        items.append(dict(person=item['person'], cranio=extract(item, 'cranio', loc['cranio'], model, config['multiview'], config['regional'], config.get('exclusions'))))
        log('Candidato: ' + item['person'])
    if not q['face']['views'][0]:
        raise ValueError('Face frontal nao pode ser processada')
    # encode_items only needs its own modality; scores_for takes separate query/candidates.
    matrix = scores_for(items, state, model, [q])[0]
    result = []
    for i, item in enumerate(items):
        valid = bool(item['cranio']['views'][0]) and np.isfinite(matrix[i])
        sex = (candidate_sexes or {}).get(item['person'])
        excluded = reference_sex in ('MASCULINO', 'FEMININO') and sex in ('MASCULINO', 'FEMININO') and reference_sex != sex
        result.append(dict(person=item['person'], score=float(matrix[i]) if valid else None,
                           excluded=excluded, reason='SEXO_DOCUMENTADO_DIVERGENTE' if excluded else ('' if valid else 'DADOS_INSUFICIENTES'),
                           sex_recorded=sex, views=item['cranio']['views'], points=item['cranio']['points'],
                           coverage=sum(item['cranio']['availability'])/55, failures=item['cranio']['failures']))
    ordered = sorted([r for r in result if r['score'] is not None], key=lambda r: (-r['score'], r['person']))
    included_rank = 0
    for rank, row in enumerate(ordered, 1):
        row['original_rank'] = rank
        row['difference_to_leader'] = ordered[0]['score']-row['score']
        if not row['excluded']:
            included_rank += 1
            row['rank'] = included_rank
        else:
            row['rank'] = None
    for row in result:
        row.setdefault('rank', None)
        row.setdefault('original_rank', None)
    correct = next((r for r in result if r['person'] == query['person']), None)
    return dict(run=str(run), query_points=q['face']['points'], query_views=q['face']['views'], ranking=sorted(result, key=lambda r: r['original_rank'] or 10**9),
                correct_rank=correct['rank'] if correct else None, correct_excluded=bool(correct and correct['excluded']))


def evaluate(run, rows, split='validacao', gallery_size=10, seed=42, stop=None, log=print):
    if split not in ('validacao', 'teste'):
        raise ValueError('Avaliacao deve usar validacao ou teste')
    state, model = load_model(run)
    items = [p for p in paired(rows) if p['split'] == split]
    train_ids = set(state['config']['train_ids'])
    if any(r['person'] in train_ids for r in items):
        raise ValueError('Avaliacao contem identidades usadas no treino')
    if gallery_size < 2:
        raise ValueError('Galeria precisa de pelo menos dois candidatos')
    if len(items) < 2:
        raise ValueError('Split com menos de duas pessoas')
    validate_items(items)
    loc = localizers(state['config'])
    extracted = []
    for item in items:
        if stop and Path(stop).exists():
            raise ValueError('Avaliacao interrompida; nenhum resultado final publicado')
        record = dict(person=item['person'], year=item['year'])
        for m in ['face', 'cranio']:
            record[m] = extract(item, m, loc[m], model, state['config']['multiview'], state['config']['regional'], state['config'].get('exclusions'))
        extracted.append(record)
        log('Avaliando ' + item['person'])
    scores = scores_for(extracted, state, model)
    for i, row in enumerate(extracted):
        if not row['face']['views'][0]:
            scores[i, :] = -np.inf
        if not row['cranio']['views'][0]:
            scores[:, i] = -np.inf
    ids = [r['person'] for r in extracted]
    metrics = retrieval(scores, ids)
    rng = random.Random(seed)
    ranks = []
    n = min(gallery_size, len(items))
    for i in range(len(items)):
        subset = rng.sample([j for j in range(len(items)) if j != i], n-1)+[i]
        order = sorted(subset, key=lambda j: (-float(scores[i, j]) if np.isfinite(scores[i, j]) else float('inf'), ids[j]))
        ranks.append(order.index(i)+1 if np.isfinite(scores[i, i]) else None)
    coverage_groups = {}
    for i, record in enumerate(extracted):
        coverage = np.mean(np.array(record['face']['availability']) & np.array(record['cranio']['availability']))
        label = 'alta_80pct' if coverage >= .8 else ('parcial_50pct' if coverage >= .5 else 'baixa_ou_sem_pontos')
        coverage_groups.setdefault(label, []).append(metrics['ranks'][i])
    metrics.update(gallery_size=n, gallery_mrr=sum(1/r if r else 0 for r in ranks)/len(ranks),
                   gallery_recall_1=sum(r == 1 for r in ranks)/len(ranks), split=split, seed=seed,
                   per_year={y: rank_metrics([metrics['ranks'][i] for i, r in enumerate(extracted) if r['year'] == y]) for y in sorted({r['year'] for r in extracted})},
                   per_coverage={k: rank_metrics(v) for k, v in coverage_groups.items()})
    target = new_run('matching_evaluations')
    save(target/'metrics.json', metrics)
    save(target/'cases.json', [dict(person=ids[i], rank=metrics['ranks'][i], gallery_rank=ranks[i],
                                   face_failures=r['face']['failures'], skull_failures=r['cranio']['failures']) for i, r in enumerate(extracted)])
    save(target/'config.json', dict(run=str(run), checkpoint_hash=sha(Path(run)/'best.pt'), split=split))
    return target
