import unittest
import numpy as np
from src.region_pilot_ui import display_frame, viewing_bounds, free_zoom_bounds, regional_zoom_defaults, region_outline


class DisplayTests(unittest.TestCase):
    def test_eye_outline_and_initial_zoom_share_bounds(self):
        ex={'proposals':[]}
        raw=[300,300,440,308]
        reg={'regions':{'olho_direito':{'crop_bounds':raw}}}
        box=viewing_bounds(raw,'olho_direito',ex,'cranio')
        self.assertEqual(box[2]-box[0],box[3]-box[1])
        zoom,x,y=regional_zoom_defaults(1000,800,'olho_direito',ex,reg,'cranio')
        viewport=free_zoom_bounds(1000,800,x,y,zoom)
        self.assertLess(viewport[0],box[0])
        self.assertLess(viewport[1],box[1])
        self.assertGreater(viewport[2],box[2])
        self.assertGreater(viewport[3],box[3])

    def test_outline_preserves_source_and_missing_bounds(self):
        image=np.zeros((100,100,3),dtype=np.uint8)
        result=region_outline(image,[20,30,70,80])
        self.assertTrue(result[30,20].any())
        self.assertFalse(image.any())
        self.assertFalse(region_outline(image,None).any())

    def test_region_selection_initializes_zoom(self):
        ex={'proposals':[]}
        reg={'regions':{'olho_esquerdo':{'crop_bounds':[500,200,600,220]}}}
        z,x,y=regional_zoom_defaults(800,600,'olho_esquerdo',ex,reg,'cranio')
        self.assertGreater(z,1)
        self.assertGreater(x,50)
        self.assertLess(y,50)
        self.assertEqual(regional_zoom_defaults(800,600,'Imagem inteira',ex,reg,'cranio'),(1.,50,50))
        self.assertGreater(regional_zoom_defaults(800,600,'nariz',ex,reg,'cranio')[0],1)

    def test_free_zoom_without_points(self):
        self.assertEqual(free_zoom_bounds(800,400,50,50,1),[0,0,800,400])
        self.assertEqual(free_zoom_bounds(800,400,50,50,2),[200,100,600,300])
        self.assertEqual(free_zoom_bounds(800,400,100,0,2),[400,0,800,200])

    def test_region_specific_context(self):
        ex={'proposals':[{'numero':n,'x':x,'y':y} for n,x,y in
                         [(2,100,100),(7,100,180),(5,60,60),(6,140,60)]]}
        jaw=viewing_bounds([20,40,180,200],'mandibula_queixo',ex,'cranio')
        self.assertGreater(jaw[1],100)
        self.assertLess(jaw[3],200)
        nose=viewing_bounds([80,85,120,110],'nariz',ex,'cranio')
        self.assertLess(nose[1],60)
        self.assertLess(nose[0],80)
        image=np.zeros((220,220,3),np.uint8)
        self.assertEqual(display_frame(image,jaw,square_context=False).shape,(600,600,3))

    def test_fixed_size_for_all_shapes(self):
        for shape in [(900,300,3),(100,800,3),(600,600,3)]:
            image=np.zeros(shape,dtype=np.uint8)
            self.assertEqual(display_frame(image).shape,(600,600,3))
            self.assertEqual(display_frame(image,[0,0,80,4]).shape,(600,600,3))

    def test_eye_strip_shows_surrounding_context(self):
        image=np.zeros((200,200,3),dtype=np.uint8)
        image[70:130,50:150]=255
        before=image.copy()
        frame=display_frame(image,[50,98,150,102],100)
        self.assertEqual(frame.shape,(100,100,3))
        self.assertTrue(frame[25,50].all())
        self.assertFalse(frame[0,50].any())
        np.testing.assert_array_equal(image,before)


if __name__=='__main__': unittest.main()
