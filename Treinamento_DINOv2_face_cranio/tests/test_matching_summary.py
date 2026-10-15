import unittest

from Treinamento_DINOv2_face_cranio.matching_summary import query_summary


class MatchingSummaryTest(unittest.TestCase):
    def test_known_missing_and_upload(self):
        result = dict(query={'person': '2019_12_A'}, candidates=[{'person': '2019_12_A'}],
                      ranking=[dict(person='other', score=.8, rank=1), dict(person='2019_12_A', score=.6, rank=2)])
        summary = query_summary(result)
        self.assertEqual(summary['Ano'], '2019')
        self.assertEqual(summary['Top-1'], 'Nao')
        self.assertEqual(summary['Top-3'], 'Sim')
        result['ranking'][1]['score'] = None
        self.assertIsNone(query_summary(result)['Posicao do correto'])
        self.assertEqual(query_summary(result)['Top-10'], 'Nao')
        result['query'] = {'person': 'consulta'}
        self.assertEqual(query_summary(result)['Top-1'], 'Nao avaliavel')

