"""Resolve historical repository paths without rewriting signed JSON/checkpoints."""
import json
import os
from functools import lru_cache
from pathlib import Path as NativePath

REPO = NativePath(__file__).resolve().parent.parent
REGISTRY = REPO/'Treinamento_DINOv2_face_cranio/portable_roots.json'


@lru_cache(maxsize=8)
def registry_values(stamp):
    return json.loads(REGISTRY.read_text(encoding='utf-8')).get('source_roots',[]) if stamp is not None else []


def roots():
    try:
        stat=REGISTRY.stat();stamp=(str(REGISTRY),stat.st_mtime_ns,stat.st_size)
    except FileNotFoundError:stamp=None
    values = list(registry_values(stamp))
    if os.environ.get('GIGI_SOURCE_ROOT'): values.append(os.environ['GIGI_SOURCE_ROOT'])
    return sorted({str(p).replace('\\','/').rstrip('/') for p in values},key=len,reverse=True)


def relocate(value):
    text = os.fspath(value)
    normalized = text.replace('\\','/')
    for old in roots():
        if normalized == old:return str(REPO)
        if normalized.startswith(old+'/'):
            return str(REPO.joinpath(*normalized[len(old)+1:].split('/')))
    return text


class Path(type(NativePath())):
    def __new__(cls,*args):
        if args:args=(relocate(args[0]),*args[1:])
        return super().__new__(cls,*args)

    def __init__(self,*args):
        # Python 3.12+ moved pathlib argument parsing from __new__ to __init__.
        if NativePath.__init__ is not object.__init__:
            if args:args=(relocate(args[0]),*args[1:])
            super().__init__(*args)


def register():
    """Remember this clone's root so future runs also work on the other machine."""
    data=json.loads(REGISTRY.read_text(encoding='utf-8')) if REGISTRY.exists() else dict(source_roots=[])
    current=str(REPO)
    if current not in data['source_roots']:
        data['source_roots'].append(current)
        REGISTRY.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return REGISTRY


if __name__=='__main__':print(register())
