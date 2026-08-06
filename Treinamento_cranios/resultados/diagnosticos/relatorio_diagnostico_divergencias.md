# Diagnostico das imagens divergentes - sexo craniano
Fonte: `Treinamento_cranios/resultados/sexo_cranio/comparacao_genero_face_cranio.csv`.
Observacao importante: este diagnostico usa a comparacao detalhada atualmente salva, que corresponde a normalizacao `crop_pad_percentile`, nao a melhor run `atual`.
- Divergencias analisadas: **75**
- Normalizacao diagnosticada: **crop_pad_percentile**

## Distribuicao das divergencias
- Face `FEMININO` -> Cranio `MASCULINO`: **62**
- Face `MASCULINO` -> Cranio `FEMININO`: **13**

## Confianca das divergencias
- BAIXA: **29**
- INDEFINIDO: **20**
- MEDIA: **18**
- ALTA: **8**

## Alertas automaticos mais frequentes
- CE_original_possivel_corte: **36**
- CE_normalizada_possivel_corte: **36**
- CD_original_possivel_corte: **29**
- CF_normalizada_possivel_corte: **26**
- CD_normalizada_possivel_corte: **25**
- CF_original_possivel_corte: **23**
- sem_alerta_automatico: **17**
- CD_original_borrada: **7**
- CF_original_borrada: **6**
- CE_original_borrada: **6**
- CF_segmentacao_suspeita: **1**
- CE_CD_mesma_direcao_visual_direita: **1**
- CE_segmentacao_suspeita: **1**
- faltando_original_CF+CE: **1**
- faltando_normalizada_CF+CE: **1**
- faltando_original_CF+CE+CD: **1**
- faltando_normalizada_CF+CE+CD: **1**

## Leitura critica
- O principal problema visual detectado foi o enquadramento/corte agressivo da normalizacao `crop_pad_percentile`. Muitas imagens normalizadas ficam muito coladas nas bordas, especialmente nas laterais CE/CD.
- Nao apareceu evidencia de troca massiva entre CE e CD. Na inspeccao automatica, apenas um caso foi marcado com direcao visual claramente igual entre CE e CD: `2017_4_AC`. Alguns casos de baixa confianca devem ser revisados manualmente nas pranchas.
- Existem imagens com baixa nitidez/blur no original, mas esse nao parece ser o fator dominante. Os alertas de borrado foram bem menos frequentes que os de corte/enquadramento.
- Ha casos sem alerta automatico: nesses, a divergencia pode vir de morfologia ambigua, rotulo de genero vindo da face, ou limite real do classificador.
- Como a melhor run foi `atual` e esta comparacao detalhada e da `crop_pad_percentile`, o diagnostico reforca por que `crop_pad_percentile` piorou: ela altera muito o enquadramento e provavelmente remove/realca informacoes de forma instavel.

## Arquivos gerados
- Resumo por pessoa: `Treinamento_cranios/resultados/diagnosticos/diagnostico_divergencias_resumo.csv`
- Detalhe por lado/imagem: `Treinamento_cranios/resultados/diagnosticos/diagnostico_imagens_divergentes_por_lado.csv`
- Prancha visual 01: `Treinamento_cranios/resultados/diagnosticos/prancha_divergentes_crop_pad_percentile_01.jpg`
- Prancha visual 02: `Treinamento_cranios/resultados/diagnosticos/prancha_divergentes_crop_pad_percentile_02.jpg`
- Prancha visual 03: `Treinamento_cranios/resultados/diagnosticos/prancha_divergentes_crop_pad_percentile_03.jpg`
- Prancha visual 04: `Treinamento_cranios/resultados/diagnosticos/prancha_divergentes_crop_pad_percentile_04.jpg`
- Prancha visual 05: `Treinamento_cranios/resultados/diagnosticos/prancha_divergentes_crop_pad_percentile_05.jpg`
- Prancha visual 06: `Treinamento_cranios/resultados/diagnosticos/prancha_divergentes_crop_pad_percentile_06.jpg`
- Prancha visual 07: `Treinamento_cranios/resultados/diagnosticos/prancha_divergentes_crop_pad_percentile_07.jpg`
- Prancha visual 08: `Treinamento_cranios/resultados/diagnosticos/prancha_divergentes_crop_pad_percentile_08.jpg`
- Prancha visual 09: `Treinamento_cranios/resultados/diagnosticos/prancha_divergentes_crop_pad_percentile_09.jpg`
- Prancha visual 10: `Treinamento_cranios/resultados/diagnosticos/prancha_divergentes_crop_pad_percentile_10.jpg`

## Casos prioritarios para revisao manual
- `2021 | 2021_16_ARRDS`: face `FEMININO`, cranio `MASCULINO`, M=45.0%, confianca `BAIXA`, flags: <Flags(allows_duplicate_labels=True)>
- `2017 | 2017_4_AC`: face `FEMININO`, cranio `MASCULINO`, M=36.7%, confianca `BAIXA`, flags: <Flags(allows_duplicate_labels=True)>
- `2019 | 2019_121_GLBSB`: face `FEMININO`, cranio `MASCULINO`, M=51.5%, confianca `INDEFINIDO`, flags: <Flags(allows_duplicate_labels=True)>
- `2019 | 2019_207_RDAM`: face `FEMININO`, cranio `MASCULINO`, M=67.4%, confianca `MEDIA`, flags: <Flags(allows_duplicate_labels=True)>
- `2019 | 2019_234_SFLL`: face `FEMININO`, cranio `MASCULINO`, M=81.8%, confianca `ALTA`, flags: <Flags(allows_duplicate_labels=True)>
- `2021 | 2021_70_MCT`: face `MASCULINO`, cranio `FEMININO`, M=29.9%, confianca `MEDIA`, flags: <Flags(allows_duplicate_labels=True)>
- `2022 | 2022_118_MRAB`: face `FEMININO`, cranio `MASCULINO`, M=67.2%, confianca `MEDIA`, flags: <Flags(allows_duplicate_labels=True)>
- `2022 | 2022_62_GADMB`: face `FEMININO`, cranio `MASCULINO`, M=39.8%, confianca `BAIXA`, flags: <Flags(allows_duplicate_labels=True)>
- `2023 | 2023_105_MLR`: face `MASCULINO`, cranio `FEMININO`, M=31.2%, confianca `MEDIA`, flags: <Flags(allows_duplicate_labels=True)>
- `2023 | 2023_44_DKSDL`: face `MASCULINO`, cranio `FEMININO`, M=25.1%, confianca `MEDIA`, flags: <Flags(allows_duplicate_labels=True)>
- `2023 | 2023_9_ADJSO`: face `FEMININO`, cranio `MASCULINO`, M=41.5%, confianca `BAIXA`, flags: <Flags(allows_duplicate_labels=True)>
- `2017 | 2017_33_FD`: face `FEMININO`, cranio `MASCULINO`, M=44.3%, confianca `BAIXA`, flags: <Flags(allows_duplicate_labels=True)>
- `2021 | 2021_19_BCLMF`: face `FEMININO`, cranio `MASCULINO`, M=37.0%, confianca `BAIXA`, flags: <Flags(allows_duplicate_labels=True)>
- `2021 | 2021_63_LGDM`: face `FEMININO`, cranio `MASCULINO`, M=60.8%, confianca `BAIXA`, flags: <Flags(allows_duplicate_labels=True)>
- `2022 | 2022_108_MDBR`: face `FEMININO`, cranio `MASCULINO`, M=43.8%, confianca `BAIXA`, flags: <Flags(allows_duplicate_labels=True)>
- `2022 | 2022_135_RACDS`: face `MASCULINO`, cranio `FEMININO`, M=26.7%, confianca `MEDIA`, flags: <Flags(allows_duplicate_labels=True)>
- `2017 | 2017_96_TF`: face `FEMININO`, cranio `MASCULINO`, M=62.5%, confianca `BAIXA`, flags: <Flags(allows_duplicate_labels=True)>
- `2019 | 2019_3_ACCM`: face `FEMININO`, cranio `MASCULINO`, M=88.0%, confianca `ALTA`, flags: <Flags(allows_duplicate_labels=True)>
- `2019 | 2019_68_FGL`: face `FEMININO`, cranio `MASCULINO`, M=50.5%, confianca `INDEFINIDO`, flags: <Flags(allows_duplicate_labels=True)>
- `2021 | 2021_98_TSDA`: face `FEMININO`, cranio `MASCULINO`, M=65.1%, confianca `MEDIA`, flags: <Flags(allows_duplicate_labels=True)>
