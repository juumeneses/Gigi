import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path as NativePath
from unittest.mock import patch

import torch

from Treinamento_DINOv2_face_cranio import portable, core, data, engine, file_lock
from Treinamento_DINOv2_face_cranio.improvement_pipeline import check_signatures


class PortableTest(unittest.TestCase):
    def test_move_preserves_manifest_checkpoint_and_file_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=NativePath(tmp);registry=root/'portable_roots.json'
            registry.write_text(json.dumps(dict(source_roots=['/old/Gigi','C:\\old\\Gigi'])))
            with patch.object(portable,'REPO',root),patch.object(portable,'REGISTRY',registry):
                self.assertEqual(portable.Path('/old/Gigi/Banco/a.jpg'),root/'Banco/a.jpg')
                self.assertEqual(portable.Path('C:\\old\\Gigi\\Banco\\a.jpg'),root/'Banco/a.jpg')
                self.assertEqual(str(portable.Path('/old/Gigantic/file')), '/old/Gigantic/file')
                file=root/'Banco/a.jpg';file.parent.mkdir();file.write_bytes(b'not-an-image')
                token=core.sha(file)
                self.assertEqual(core.sha('/old/Gigi/Banco/a.jpg'),token)
                check_signatures({'/old/Gigi/Banco/a.jpg':token})
                manifest=dict(people=[],identity_hash_collisions={},split_hash='fixed')
                manifest['fingerprint']=core.digest(manifest)
                core.save(root/'manifest.json',manifest)
                before=core.sha(root/'manifest.json')
                self.assertEqual(data.manifest('/old/Gigi/manifest.json')['fingerprint'],manifest['fingerprint'])
                self.assertEqual(core.sha(root/'manifest.json'),before)
                folder=root/'run';folder.mkdir()
                cfg=dict(experiment='E1',dimension=384,manifest='/old/Gigi/manifest.json',epochs=1)
                core.save(folder/'config.json',cfg)
                model=engine.Towers(384)
                engine.checkpoint(folder/'best.pt',dict(config=cfg,model=model.state_dict()))
                before=core.sha(folder/'best.pt')
                loaded,_=engine.load_run('/old/Gigi/run')
                self.assertEqual(loaded,cfg)
                self.assertEqual(core.sha(folder/'best.pt'),before)

    def test_lock_is_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=NativePath(tmp)/'worker.lock'
            code='''
import sys
from Treinamento_DINOv2_face_cranio import file_lock as f
with open(sys.argv[1],'a') as lock:
    try:f.flock(lock,f.LOCK_EX|f.LOCK_NB)
    except BlockingIOError:sys.exit(7)
    f.flock(lock,f.LOCK_UN)
'''
            with path.open('a') as lock:
                file_lock.flock(lock,file_lock.LOCK_EX|file_lock.LOCK_NB)
                self.assertEqual(subprocess.run([sys.executable,'-c',code,str(path)]).returncode,7)
                file_lock.flock(lock,file_lock.LOCK_UN)
            self.assertEqual(subprocess.run([sys.executable,'-c',code,str(path)]).returncode,0)
