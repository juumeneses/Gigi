import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from src import region_pilot as p


class RegionPilotTests(unittest.TestCase):
    def test_manifest_no_test(self):
        items=p.source_manifest()
        self.assertEqual(len(items),22)
        self.assertEqual(len({i['id'] for i in items}),11)
        self.assertFalse(any(i['split']=='teste' for i in items))
        for item in items:
            self.assertEqual(p.sha256_file(p.REPO_ROOT/item['path']),item['expected_hash'])

    def test_coordinates_and_aspect(self):
        im=np.zeros((100,300,3),np.uint8)
        points=[[0,0],[120,40],[299,99]]
        for name in p.VARIANTS:
            _,matrix=p.variant(im,name)
            restored=p.transform_points(p.transform_points(points,matrix),cv2.invertAffineTransform(matrix))
            np.testing.assert_allclose(restored,points,atol=1e-10)
        output,meta=p.letterbox(im)
        self.assertEqual(output.shape,(256,256,3))
        self.assertLess(abs(meta['size'][0]/meta['size'][1]-3),.03)

    def test_boxes_missing_cut_and_empty_mask(self):
        b=p.box_for({1:[0,0],2:[10,20]},[1,2],100,100)
        self.assertEqual(b['margin'],[4.,4.])
        self.assertGreater(b['outside_fraction'],0)
        self.assertEqual(p.box_for({},[1],100,100)['status'],'REGIAO_INDISPONIVEL')
        self.assertEqual(p.box_for({1:[float('nan'),0]},[1],100,100)['status'],'REGIAO_INDISPONIVEL')
        self.assertTrue(p.clipped_by_transform([0,0,100,100],[[1,0,10],[0,1,0]],100,100))
        mask,contours,_=p.segment(np.zeros((50,50,3),np.uint8))
        self.assertFalse(mask.any()); self.assertFalse(contours)

    def test_invalid_inputs_and_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.assertEqual(p.image_input(root/'absent')[1],'AUSENTE')
            p.atomic(root/'pointer.jpg',b'version https://git-lfs.github.com/spec/v1\n')
            self.assertEqual(p.image_input(root/'pointer.jpg')[1],'LFS_NAO_BAIXADO')
            p.atomic(root/'invalid.jpg',b'not an image')
            self.assertEqual(p.image_input(root/'invalid.jpg')[1],'ILEGIVEL')
            p.seal(root/'data.json',{'status':'OK_TECNICO'},[root/'invalid.jpg'],'x')
            self.assertTrue(p.verified(root/'data.json','x'))
            value=json.loads((root/'data.json').read_text()); value['status']='tampered'
            p.json_save(root/'data.json',value)
            self.assertFalse(p.verified(root/'data.json','x'))
            p.seal(root/'ints.json',{'points':{1:[1.,2.],10:[3.,4.],2:[5.,6.]}},[],'y')
            self.assertTrue(p.verified(root/'ints.json','y'))

    def test_no_detection_and_review_visibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); image=root/'face.jpg'
            p.image_save(image,np.zeros((100,100,3),np.uint8))
            item=dict(id='sample',modalidade='face',path=str(image),expected_hash=p.sha256_file(image),proposals=[])
            mesh=p.Mesh()
            with patch.object(mesh,'detect',return_value=[]):
                ex,_=p.extract(root,item,mesh,{'mapping':{'landmarks':{}}})
            self.assertEqual(ex['reason'],'FACE_NAO_DETECTADA')
            regions,_=p.make_regions(root,item,ex)
            self.assertTrue(all(b['status']=='REGIAO_INDISPONIVEL' for b in regions['regions'].values()))
            path=root/'sample__face/regioes/data.json'
            p.seal(path,regions,[],'test')
            review=p.save_review(root,item,'mandibula_queixo','A','test','OCULTO','barba',['barba'])
            self.assertFalse(json.loads(review.read_text())['validado'])
            self.assertTrue(all(b.get('visibilidade','NAO_AVALIADO')=='NAO_AVALIADO' for b in regions['regions'].values()))

    def test_interrupted_resume_retains_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); item=dict(id='fake',modalidade='face',path=str(root/'absent.jpg'),
                                      expected_hash='absent',proposals=[],ano=2017,split='treino')
            with patch.object(p,'source_manifest',return_value=[item]), patch.object(p,'RUNS',root/'locks'):
                run=p.run_audit(root/'partial',limit=1)
                first=(run/'fake__face/extracao/data.json').read_bytes()
                p.run_audit(run,resume=True)
                self.assertEqual(first,(run/'fake__face/extracao/data.json').read_bytes())
                complete=p.run_audit(root/'complete')
                for stage in p.STAGES[:3]:
                    self.assertEqual((run/f'fake__face/{stage}/data.json').read_bytes(),
                                     (complete/f'fake__face/{stage}/data.json').read_bytes())
                rows=json.loads((run/'fake__face/estabilidade/data.json').read_text())['rows']
                self.assertEqual(len(rows),7)
                self.assertTrue(all(r['status']=='ERRO' for r in rows))
                forced=p.run_audit(run,force=True,limit=1)
                self.assertNotEqual(forced,run)
                self.assertEqual(first,(run/'fake__face/extracao/data.json').read_bytes())
                with self.assertRaisesRegex(ValueError,'Pre-requisito'):
                    p.run_audit(root/'no-extraction',stage='regioes')

    def test_panel(self):
        app=AppTest.from_file(str(p.LAB_ROOT/'revisao_cranios.py')).run(timeout=45)
        self.assertFalse(app.exception)
        if not list(p.RUNS.glob('*/relatorio.html')): return
        get=lambda label:next(s for s in app.selectbox if s.label==label)
        completed=sorted(p.RUNS.glob('*/relatorio.html'))[-1].parent
        get('Execução').set_value(completed).run()
        get('Região').set_value('nariz').run()
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label=='Próximo').click().run()
        self.assertFalse(app.exception)
        next(r for r in app.radio if r.label=='Visualização').set_value('Revisão cega').run()
        get('Sessão').set_value('B').run()
        self.assertFalse(app.exception)


if __name__=='__main__': unittest.main()
