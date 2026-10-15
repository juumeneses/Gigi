import unittest
from streamlit.testing.v1 import AppTest
from src.file_browser_ui import inventory


class FileBrowserTests(unittest.TestCase):
    def test_inventory_includes_uncatalogued(self):
        frame = inventory('face')
        self.assertIn('2017_33_FDCD_FFN.jpg', frame.arquivo.tolist())
        self.assertIn('2023_21_APS_FFN.jpg', frame.arquivo.tolist())
        self.assertFalse(frame.caminho.duplicated().any())

    def test_filters_and_navigation(self):
        app = AppTest.from_string("from src.file_browser_ui import render_files\nrender_files('face')").run(timeout=30)
        self.assertFalse(app.exception)
        app.text_input[0].set_value('2017_33_FD').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.selectbox[2].options), 1)
        self.assertFalse(app.error)
        app.text_input[0].set_value('not_an_existing_filename_123').run()
        self.assertFalse(app.exception)
        self.assertTrue(app.info)


if __name__ == '__main__':
    unittest.main()
