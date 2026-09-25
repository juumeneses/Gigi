import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.cranios import detect_geometry
from src.cranios import _orbital_prior


def test_detector_nao_inventa_sem_imagem():
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    metrics, points, mask, preview = detect_geometry(image)
    assert metrics["status_cranio_geo"] == "FALHA"
    assert points == []


def test_detector_tolerante_a_topo_cortado():
    image = np.zeros((400, 300, 3), dtype=np.uint8)
    image[:360, 40:260] = 180
    metrics, points, _, _ = detect_geometry(image)
    assert metrics["topo_truncado"]
    assert any(p["ponto"] == "mento" for p in points)


def test_prior_orbital_respeita_faixa_anatomica():
    rows = [
        ({"confianca_par_orbital": .8}, [
            {"ponto": "orbita_direita", "visivel": True, "x_norm": .40, "y_norm": .25},
            {"ponto": "orbita_esquerda", "visivel": True, "x_norm": .60, "y_norm": .25},
        ])
    ]
    prior = _orbital_prior(rows)
    assert .39 <= prior["ry"] <= .44
    assert .34 <= prior["sep"] <= .46
