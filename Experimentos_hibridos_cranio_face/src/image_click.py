"""Coordinates in the displayed crop map back to original pixels."""
import math


def map_click(u, v, bounds):
    if not all(math.isfinite(float(n)) for n in (u, v)) or not (0 <= u <= 1 and 0 <= v <= 1):
        raise ValueError('Clique fora da imagem.')
    left, top, right, bottom = bounds
    if right <= left or bottom <= top:
        raise ValueError('Regiao vazia.')
    return (min(right-1, left+int(u*(right-left))),
            min(bottom-1, top+int(v*(bottom-top))))
