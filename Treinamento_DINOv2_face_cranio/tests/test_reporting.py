import tempfile
import unittest
from pathlib import Path

from Treinamento_DINOv2_face_cranio.core import save
from Treinamento_DINOv2_face_cranio.reporting import comparison_rows, metadata


class ReportingTest(unittest.TestCase):
    def test_dates_parameters_and_comparable_leader(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = []
            for date, value, cohort in [('20261014T150000000000Z', .2, 'a'),
                                        ('20261014T160000000000Z', .3, 'a'),
                                        ('20261014T170000000000Z', .9, 'b')]:
                run = Path(tmp)/date/'frontal'; run.mkdir(parents=True)
                save(run/'config.json', dict(cohort=cohort, experiment='E1', loss='infonce', seed=42, smoke=False))
                metric = dict(n=2,mrr=value,**{f'recall_{k}':value for k in (1,3,5,10)})
                save(run/'evaluation.json', dict(full=metric, gallery=metric,gallery_size=2,
                                                 rows=[dict(person='one'),dict(person='two')],galleries={}))
                save(run/'history.json', [dict(epoch=1,val_mrr=value)])
                runs.append(run)
            self.assertEqual(metadata(runs[0])['Data (Sao Paulo)'], '14/10/2026 12:00:00')
            self.assertEqual(metadata(runs[0])['Melhor epoca (MRR)'], 1)
            rows = comparison_rows(runs)
            self.assertEqual(rows[0]['Evolucao'], 'Primeira referencia')
            self.assertAlmostEqual(rows[1]['Ganho MRR vs melhor anterior'], .1)
            self.assertAlmostEqual(rows[2]['Diferenca MRR vs melhor atual'], -.1)
            self.assertEqual(rows[2]['Melhor atual (MRR)'], rows[1]['Execucao'])


if __name__ == '__main__':
    unittest.main()
