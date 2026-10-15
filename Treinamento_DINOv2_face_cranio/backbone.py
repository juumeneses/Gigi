from __future__ import annotations

import os
os.environ.setdefault('XFORMERS_DISABLED', '1')
import shutil
import urllib.request
import zipfile
from .portable import Path

import numpy as np
import torch

from .core import (ARTIFACTS, atomic, check_stop, digest, letterbox, load, new_run,
                   np_save, read_image, save, sha)

REVISION = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
WEIGHT_URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'
MODEL_DIR = ARTIFACTS/'models'/REVISION
MODELS = {'dinov2_vits14': ('DINOv2 Small', 384), 'dinov2_vitb14': ('DINOv2 Base', 768),
          'convnext_tiny': ('ConvNeXt Tiny', 768)}


def model_dir(model):
    if model not in MODELS:
        raise ValueError('Extrator desconhecido')
    return MODEL_DIR if model == 'dinov2_vits14' else ARTIFACTS/'models'/model


def prepare_variant(model, log=print):
    if model == 'dinov2_vits14':
        return prepare(log)
    folder = model_dir(model)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'model.json').exists():
        verify_variant(model)
        return folder
    if model == 'dinov2_vitb14':
        prepare(log)
        url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
    else:
        from torchvision.models import ConvNeXt_Tiny_Weights
        url = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.url
    log('Baixando pesos oficiais: '+model, flush=True)
    path = folder/'weights.pth'
    tmp = path.with_suffix('.download')
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    import torchvision
    save(folder/'model.json', dict(model=model, dimension=MODELS[model][1], weight_url=url,
        weight_hash=sha(path), revision=REVISION if model.startswith('dinov2') else torchvision.__version__.split('+')[0]))
    return folder


def verify_variant(model):
    if model == 'dinov2_vits14':
        return verify_model()
    folder = model_dir(model)
    if not (folder/'model.json').exists():
        raise ValueError('Prepare o extrator selecionado primeiro: '+model)
    meta = load(folder/'model.json')
    if sha(folder/'weights.pth') != meta['weight_hash']:
        raise ValueError('Pesos alterados: '+model)
    if model == 'dinov2_vitb14':
        meta = dict(meta, source=verify_model()['source_hashes'])
    else:
        import torchvision
        if torchvision.__version__.split('+')[0] != meta['revision'].split('+')[0]:
            raise ValueError('Versao torchvision mudou; prepare uma execucao compativel.')
    return meta


def prepare(log=print):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metadata = MODEL_DIR/'model.json'
    if metadata.exists():
        verify_model()
        return MODEL_DIR
    archive = MODEL_DIR/'source.zip'
    source = MODEL_DIR/'source'
    if not source.exists():
        log('Baixando codigo oficial DINOv2 fixado em ' + REVISION, flush=True)
        urllib.request.urlretrieve('https://codeload.github.com/facebookresearch/dinov2/zip/'+REVISION, archive)
        staging = MODEL_DIR/'staging'
        staging.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if not (staging/member.filename).resolve().is_relative_to(staging.resolve()):
                    raise ValueError('Caminho invalido no arquivo do modelo')
            z.extractall(staging)
        shutil.move(str(staging/('dinov2-'+REVISION)), source)
        archive.unlink()
    weight = MODEL_DIR/'dinov2_vits14.pth'
    if not weight.exists():
        log('Baixando pesos oficiais ViT-S/14 (aproximadamente 84 MB)', flush=True)
        tmp = weight.with_suffix('.download')
        urllib.request.urlretrieve(WEIGHT_URL, tmp)
        tmp.replace(weight)
    save(metadata, dict(model='dinov2_vits14', revision=REVISION, weight_url=WEIGHT_URL,
                        weight_hash=sha(weight), source_hashes={str(p.relative_to(source)): sha(p)
                            for p in sorted(source.rglob('*.py'))}, dimension=384))
    return MODEL_DIR


def verify_model():
    path = MODEL_DIR/'model.json'
    if not path.exists():
        raise ValueError('Prepare o modelo DINOv2 primeiro.')
    data = load(path)
    if sha(MODEL_DIR/'dinov2_vits14.pth') != data['weight_hash']:
        raise ValueError('Pesos DINOv2 alterados')
    for name, expected in data['source_hashes'].items():
        if sha(MODEL_DIR/'source'/name) != expected:
            raise ValueError('Codigo DINOv2 alterado: '+name)
    return data


def tensor(im, size):
    canvas, transform = letterbox(im, size)
    value = torch.from_numpy(np.array(canvas).copy()).permute(2, 0, 1).float()/255
    return (value-torch.tensor([.485, .456, .406])[:, None, None])/torch.tensor([.229, .224, .225])[:, None, None], transform


class Extractor:
    def __init__(self, size=224, device='auto', backbone='dinov2_vits14', preprocessing='original'):
        if size < 56 or size % 14:
            raise ValueError('Resolucao precisa ser multipla de 14 e >=56')
        torch.set_num_threads(2)
        self.meta, self.size = verify_variant(backbone), size
        self.dimension = self.meta['dimension']
        self.fingerprint = digest(dict(model=self.meta, size=size, preprocessing='exif_rgb_letterbox_v1'))
        if preprocessing not in ('original', 'uniform_background'):
            raise ValueError('Enquadramento desconhecido')
        self.preprocessing = preprocessing
        if preprocessing != 'original':
            from . import preprocessing as preparation
            self.fingerprint = digest([self.fingerprint, preprocessing, sha(preparation.__file__)])
        if backbone == 'convnext_tiny':
            from torchvision.models import convnext_tiny
            self.model = convnext_tiny(weights=None)
            self.model.load_state_dict(torch.load(model_dir(backbone)/'weights.pth', map_location='cpu', weights_only=True))
            self.model.classifier[2] = torch.nn.Identity()
        else:
            self.model = torch.hub.load(str(MODEL_DIR/'source'), backbone, source='local', pretrained=False)
            weights = MODEL_DIR/'dinov2_vits14.pth' if backbone == 'dinov2_vits14' else model_dir(backbone)/'weights.pth'
            self.model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        self.model.eval().requires_grad_(False)
        choices = ([d for d, ok in [('cuda', torch.cuda.is_available()),
                     ('mps', torch.backends.mps.is_available())] if ok] + ['cpu']) if device == 'auto' else [device]
        for d in choices:
            try:
                self.model.to(d)
                with torch.inference_mode():
                    output = self.model(torch.zeros(1, 3, size, size, device=d))
                if not torch.isfinite(output).all():
                    raise ValueError('Saida nao finita')
                self.device = d
                break
            except (RuntimeError, ValueError):
                if d == choices[-1]:
                    raise

    def encode_image(self, im):
        from .preprocessing import prepare_image
        im, crop = prepare_image(im, self.preprocessing)
        x, transform = tensor(im, self.size)
        transform['crop'] = crop
        with torch.inference_mode():
            value = self.model(x[None].to(self.device))[0].float().cpu().numpy()
        if value.shape != (self.dimension,) or not np.isfinite(value).all():
            raise ValueError('Caracteristica DINOv2 invalida')
        return value, transform

    def feature(self, row):
        actual = sha(row['path'])
        if row.get('hash') and row['hash'] != actual:
            raise ValueError('Imagem alterada depois do inventario')
        key = digest([actual, self.fingerprint])
        path = ARTIFACTS/'features'/self.fingerprint/(key+'.npz')
        if path.exists():
            try:
                metadata = load(path.with_suffix('.json'))
                if metadata.get('feature_hash') != sha(path):
                    raise ValueError('Cache alterado')
                with np.load(path, allow_pickle=False) as stored:
                    value = stored['embedding'].copy()
                    if value.shape == (self.dimension,) and np.isfinite(value).all() and str(stored['key']) == key:
                        return value
            except (ValueError, OSError, KeyError, zipfile.BadZipFile):
                pass
        value, transform = self.encode_image(read_image(row['path']))
        np_save(path, embedding=value, key=key)
        save(path.with_suffix('.json'), dict(input_hash=actual, extractor=self.fingerprint,
                                            transform=transform, device=self.device, feature_hash=sha(path)))
        return value


def extract(manifest, size=224, device='auto', run_dir=None, resume=False, stop=None, log=print, backbone='dinov2_vits14'):
    from .data import manifest as read_manifest
    source = read_manifest(manifest)
    model = Extractor(size, device) if backbone == 'dinov2_vits14' else Extractor(size, device, backbone=backbone)
    run = Path(run_dir) if run_dir else new_run('extractions')
    config = dict(manifest=manifest, manifest_hash=sha(manifest), size=size, extractor=model.fingerprint)
    if (run/'config.json').exists() and (not resume or load(run/'config.json') != config):
        raise ValueError('Extracao existente ou retomada incompativel')
    save(run/'config.json', config)
    save(run/'status.json', dict(state='RODANDO', device=model.device))
    results = []
    try:
        for i, row in enumerate(source['rows']):
            if row['view'] == 'CE':
                continue
            check_stop(stop)
            try:
                model.feature(row)
                result = dict(path=row['path'], status='OK')
            except Exception as exc:
                result = dict(path=row['path'], status='ERRO', error=str(exc))
            results.append(result)
            save(run/'results.json', results)
            log(f'Extracao {i+1}/{len(source["rows"])} {result["status"]}', flush=True)
    except InterruptedError:
        save(run/'status.json', dict(state='INTERROMPIDO'))
        return run
    save(run/'status.json', dict(state='CONCLUIDO'))
    return run
