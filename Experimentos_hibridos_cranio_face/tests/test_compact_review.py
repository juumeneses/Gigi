import unittest
from streamlit.testing.v1 import AppTest
from src.common import LAB_ROOT


class CompactReviewTests(unittest.TestCase):
    def test_navigation(self):
        app = AppTest.from_file(str(LAB_ROOT/'revisao_cranios.py')).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.radio[0].options, ['Faces', 'Crânios', 'Comparação', 'Matching'])
        for area in ['Crânios', 'Comparação', 'Matching']:
            app.radio[0].set_value(area).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
        self.assertTrue(any('indisponível' in message.value for message in app.info))


if __name__ == '__main__':
    unittest.main()
