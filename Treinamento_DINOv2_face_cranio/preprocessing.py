"""Conservative trimming of a uniform background; no anatomical detector."""
import numpy as np
from PIL import Image


def prepare_image(image, mode='original'):
    if mode not in ('original', 'uniform_background'):
        raise ValueError('Enquadramento desconhecido')
    w, h = image.size
    info = dict(mode=mode, box=[0,0,w,h], removed_fraction=0., applied=False, reason='original')
    if mode == 'original':
        return image, info
    # Inspect every side. Heterogeneous backgrounds keep the full image.
    thumb = image.copy()
    thumb.thumbnail((320,320), Image.Resampling.BILINEAR)
    a = np.asarray(thumb.convert('RGB')).astype(float)
    border = np.concatenate([a[0], a[-1], a[:,0], a[:,-1]])
    background = np.median(border, axis=0)
    if np.mean(np.linalg.norm(border-background, axis=1) < 35) < .65:
        info['reason'] = 'fundo_nao_uniforme'
        return image, info
    mask = np.linalg.norm(a-background, axis=2) >= 35
    ys, xs = np.where(mask)
    if len(xs) < max(16, .01*mask.size):
        info['reason'] = 'primeiro_plano_insuficiente'
        return image, info
    x0, x1 = xs.min()*w/a.shape[1], (xs.max()+1)*w/a.shape[1]
    y0, y1 = ys.min()*h/a.shape[0], (ys.max()+1)*h/a.shape[0]
    mx, my = max(4, .1*(x1-x0)), max(4, .1*(y1-y0))
    box = [max(0,int(x0-mx)), max(0,int(y0-my)), min(w,int(np.ceil(x1+mx))), min(h,int(np.ceil(y1+my)))]
    removed = 1-(box[2]-box[0])*(box[3]-box[1])/(w*h)
    if removed < .02:
        info['reason'] = 'sem_bordas_removiveis'
        return image, info
    info.update(box=box, removed_fraction=removed, applied=True, reason='bordas_uniformes', margin=.1)
    return image.crop(box), info
