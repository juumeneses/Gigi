"""Read-only reference reader; no image inference or training."""
from pathlib import Path
import pandas as pd
from .core import load, sha

def references(path):
    path=Path(path)
    if sha(path/'reference.csv') != load(path/'config.json')['hash']:
        raise ValueError('Arquivo de referência alterado')
    return pd.read_csv(path/'reference.csv', dtype=str).set_index('id_pessoa')
