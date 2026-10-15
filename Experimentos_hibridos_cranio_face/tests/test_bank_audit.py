import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from src.bank_audit import inspect


class InventoryTests(unittest.TestCase):
    def test_audit_button(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
from unittest.mock import patch
from src.bank_audit import render_report
import streamlit as st
def fake_execute(command):
    st.session_state['audit_command'] = [str(c) for c in command]
    return 0
with patch('src.region_pilot_ui.execute', side_effect=fake_execute):
    render_report()
''').run(timeout=30)
        app.button(key='run_bank_audit').click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(app.success)
        self.assertIn('--run-dir', app.session_state['audit_command'])
        self.assertTrue(any(c.endswith('/scripts/auditar_banco.py') for c in app.session_state['audit_command']))

    def test_uncatalogued_image_explained(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
from src.bank_audit import render_file_detail
render_file_detail(dict(id='exemplo', vista='face', status='ERRO', motivo='NAO_CATALOGADO', caminho=''))
''').run()
        self.assertFalse(app.exception)
        self.assertIn('Sem imagem para exibir', app.warning[0].value)

    def test_missing_file_explained(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
from src.bank_audit import render_file_detail
render_file_detail(dict(id='exemplo', vista='CF', status='ERRO', motivo='AUSENTE', caminho='/tmp/gigi-test-file-that-does-not-exist.png'))
''').run()
        self.assertFalse(app.exception)
        self.assertIn('Arquivo não encontrado', app.error[0].value)

    def test_missing_and_lfs(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'image.png'
            self.assertEqual(inspect(path)['motivo'], 'AUSENTE')
            path.write_text('version https://git-lfs.github.com/spec/v1\n')
            self.assertEqual(inspect(path)['status'], 'ERRO')

    def test_flat_small_image(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'image.png'
            cv2.imwrite(str(path), np.zeros((64,64,3), dtype=np.uint8))
            result = inspect(path)
            self.assertEqual(result['status'], 'ATENCAO')
            self.assertIn('RESOLUCAO_BAIXA', result['motivo'])
            self.assertEqual(result['largura'], 64)
