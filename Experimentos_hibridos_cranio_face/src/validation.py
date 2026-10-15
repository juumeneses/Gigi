"""Independent visual review; never infer approval from detector confidence."""
import hashlib
import json
import pandas as pd


def review_sample(manifest, splits, seed=20260915):
    pool = manifest.merge(splits[['id_pessoa', 'split']], on='id_pessoa', validate='one_to_one')
    pool = pool[pool.split.isin(['treino', 'validacao'])]
    selected = []
    for year, group in pool.groupby('ano'):
        group = group.sort_values('id_pessoa').sample(n=min(20, len(group)), random_state=seed + int(year)).copy()
        group['grupo_revisao'] = ['desenvolvimento' if i % 2 == 0 else 'aceitacao' for i in range(len(group))]
        selected.append(group)
    return pd.concat(selected, ignore_index=True) if selected else pool


def point_fingerprint(row, input_hash):
    values = {k: row.get(k) for k in ['ponto', 'x_px', 'y_px', 'metodo', 'confianca']}
    values['input_hash'] = input_hash
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


def wilson(correct, total):
    if not total:
        return (None, None)
    z = 1.96
    p = correct / total
    center = (p + z*z/(2*total))/(1+z*z/total)
    half = z*((p*(1-p)/total + z*z/(4*total*total))**.5)/(1+z*z/total)
    return center-half, center+half
