# Onde fica cada parte

Caminhos abaixo são relativos a codigo_referencia/ e à pasta Gigi do projeto original.

| Arquivo | Responsabilidade |
|---|---|
| Treinamento_DINOv2_face_cranio/app.py | Navegação, início, banco, treino de matching, resultados e comparador |
| Treinamento_DINOv2_face_cranio/sex_controls.py | Informação opcional da consulta e filtro do ranking preservando o original |
| Treinamento_DINOv2_face_cranio/matching_summary.py | Resumo da consulta e rótulos do histórico |
| Treinamento_DINOv2_face_cranio/log_view.py | Apresentação dos logs |
| Treinamento_DINOv2_face_cranio/ui_help.py | Ajuda de parâmetros e tabelas |
| Treinamento_DINOv2_face_cranio/cli.py | Encaminhamento dos trabalhos de matching em segundo plano |
| Treinamento_DINOv2_face_cranio/engine.py | Treino, avaliação e comparação de matching |
| Treinamento_DINOv2_face_cranio/fusion.py | Modelos por vista e combinações de scores |
| Treinamento_DINOv2_face_cranio/backbone.py | Extratores visuais de matching |
| Treinamento_DINOv2_face_cranio/data.py | Inventário, coortes e leitura dos splits |
| Laboratorio_visual_face_cranio/registered.py | União dos resultados históricos por identificador |
| Laboratorio_visual_face_cranio/registered_ui.py | Filtros, tabelas e download dos resultados históricos |
| Laboratorio_visual_face_cranio/references_io.py | Leitura e integridade das referências |

## Entradas existentes no computador original, não incluídas

- Banco/: imagens originais.
- Experimentos_hibridos_cranio_face/dataset/splits/split_identidades.csv: divisão de identidades.
- Treinamento_faces/resultados/genero_faces/: curadoria e resultados faciais já registrados.
- Treinamento_cranios/resultados/sexo_cranio/: resultados cranianos já registrados.
- Treinamento_DINOv2_face_cranio/artifacts/: inventários, trabalhos e modelos de matching.
- Laboratorio_visual_face_cranio/artifacts/: referências e resultados já registrados.

Em app.py, procurar as funções home_page(), sex_page(), matching_page() e matching_viewer(). Em sex_controls.py, procurar matching_controls() e matching_filter().
