import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from Laboratorio_visual_face_cranio.point_editor import canvas, map_click
from Laboratorio_visual_face_cranio.core import write_image, save, sha, load


class PointEditorTests(unittest.TestCase):
    def test_click_coordinates_and_padding(self):
        for shape in [(800,400),(300,900)]:
            im=np.zeros((*shape,3),np.uint8)
            for region in [None,dict(box=[50,60,180,200])]:
                _,matrix,bounds=canvas(im,[],region,2)
                point=np.array([100.,120.,1.])
                screen=matrix @ point
                np.testing.assert_allclose(map_click(*(screen/600),matrix,bounds),point[:2])
                self.assertIsNone(map_click(-.1,.5,matrix,bounds))
        _,matrix,bounds=canvas(np.zeros((800,400,3),np.uint8),[],None,2)
        self.assertIsNone(map_click(0,0,matrix,bounds))

    def test_partial_draft_is_saved_and_reloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'image.png'
            write_image(path,np.zeros((80,100,3),np.uint8))
            proposal=root/'proposal.json'
            points=[dict(numero=n,x=None,y=None,visibility='NAO_AVALIADO') for n in range(1,12)]
            row=dict(path=str(path),hash=sha(path),width=100,height=80,points=points)
            save(proposal,row)
            script=f'''
import pandas as pd
from Laboratorio_visual_face_cranio.core import load,image
from Laboratorio_visual_face_cranio.point_editor import editor
from pathlib import Path
p=Path({str(proposal)!r})
r=load(p)
editor(p,r,image(r['path']),pd.DataFrame(r['points']))
'''
            with patch('Laboratorio_visual_face_cranio.point_editor.ARTIFACTS',root):
                app=AppTest.from_string(script).run()
                self.assertFalse(app.exception)
                next(b for b in app.button if b.label=='Salvar rascunho').click().run()
                self.assertFalse(app.exception)
                saved=list(root.glob('point_drafts/*/draft.json'))
                self.assertEqual(len(saved),1)
                self.assertEqual(load(saved[0])['points'][1]['x'],None)
                self.assertFalse(list(root.glob('point_reviews/*')))
                self.assertFalse(AppTest.from_string(script).run().exception)


if __name__=='__main__':
    unittest.main()
