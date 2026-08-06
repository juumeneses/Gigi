# Diagnostico das divergencias - melhor run atual

- Fonte: `Treinamento_cranios/resultados/sexo_cranio/comparacao_genero_face_cranio.csv`
- Normalizacao: **atual**
- Divergencias: **42** de 731 comparados

## Distribuicao
- Face `FEMININO` -> cranio `MASCULINO`: **28**
- Face `MASCULINO` -> cranio `FEMININO`: **14**

## Confianca
- MEDIA: **16**
- ALTA: **11**
- BAIXA: **10**
- INDEFINIDO: **5**

## Alertas automaticos mais frequentes
- CE_normalizada_possivel_corte: **18**
- CF_normalizada_possivel_corte: **16**
- CE_original_possivel_corte: **16**
- CF_original_possivel_corte: **14**
- CD_normalizada_possivel_corte: **13**
- CD_original_possivel_corte: **12**
- sem_alerta_automatico: **10**
- CD_original_borrada: **6**
- CF_original_borrada: **5**
- CE_original_borrada: **5**
- faltando_original_CF+CE: **1**
- faltando_normalizada_CF+CE: **1**
- faltando_original_CF+CE+CD: **1**
- faltando_normalizada_CF+CE+CD: **1**

## Casos prioritarios para revisao manual
- `2019 | 2019_234_SFLL`: face `FEMININO`, cranio `MASCULINO`, M=92.7%, confianca `ALTA`, flags: <Flags(allows_duplicate_labels=True)>
- `2021 | 2021_63_LGDM`: face `FEMININO`, cranio `MASCULINO`, M=66.0%, confianca `MEDIA`, flags: <Flags(allows_duplicate_labels=True)>
- `2022 | 2022_62_GADMB`: face `FEMININO`, cranio `MASCULINO`, M=57.0%, confianca `BAIXA`, flags: <Flags(allows_duplicate_labels=True)>
- `2023 | 2023_119_PTC`: face `FEMININO`, cranio `MASCULINO`, M=66.5%, confianca `MEDIA`, flags: <Flags(allows_duplicate_labels=True)>
- `2017 | 2017_33_FD`: face `FEMININO`, cranio `MASCULINO`, M=54.4%, confianca `INDEFINIDO`, flags: <Flags(allows_duplicate_labels=True)>
- `2023 | 2023_44_DKSDL`: face `MASCULINO`, cranio `FEMININO`, M=12.8%, confianca `ALTA`, flags: <Flags(allows_duplicate_labels=True)>
- `2023 | 2023_7_ACSDS`: face `FEMININO`, cranio `MASCULINO`, M=61.2%, confianca `BAIXA`, flags: <Flags(allows_duplicate_labels=True)>
- `2019 | 2019_3_ACCM`: face `FEMININO`, cranio `MASCULINO`, M=99.0%, confianca `ALTA`, flags: <Flags(allows_duplicate_labels=True)>
- `2022 | 2022_138_RCCA`: face `MASCULINO`, cranio `FEMININO`, M=29.9%, confianca `MEDIA`, flags: <Flags(allows_duplicate_labels=True)>
- `2022 | 2022_36_DFBN`: face `MASCULINO`, cranio `FEMININO`, M=5.2%, confianca `ALTA`, flags: <Flags(allows_duplicate_labels=True)>
- `2022 | 2022_78_JESA`: face `MASCULINO`, cranio `FEMININO`, M=2.1%, confianca `ALTA`, flags: <Flags(allows_duplicate_labels=True)>
- `2023 | 2023_90_LGSG`: face `FEMININO`, cranio `MASCULINO`, M=55.5%, confianca `BAIXA`, flags: <Flags(allows_duplicate_labels=True)>
- `2023 | 2023_92_MADA`: face `FEMININO`, cranio `MASCULINO`, M=53.0%, confianca `INDEFINIDO`, flags: <Flags(allows_duplicate_labels=True)>
- `2023 | 2023_99_MGDSN`: face `FEMININO`, cranio `MASCULINO`, M=77.4%, confianca `MEDIA`, flags: <Flags(allows_duplicate_labels=True)>
- `2017 | 2017_11_BB`: face `MASCULINO`, cranio `FEMININO`, M=33.2%, confianca `MEDIA`, flags: <Flags(allows_duplicate_labels=True)>
- `2017 | 2017_52_JDRV`: face `MASCULINO`, cranio `FEMININO`, M=39.2%, confianca `BAIXA`, flags: <Flags(allows_duplicate_labels=True)>
- `2017 | 2017_70_MLBDA`: face `FEMININO`, cranio `MASCULINO`, M=54.4%, confianca `INDEFINIDO`, flags: <Flags(allows_duplicate_labels=True)>
- `2017 | 2017_95_TDOES`: face `FEMININO`, cranio `MASCULINO`, M=85.2%, confianca `ALTA`, flags: <Flags(allows_duplicate_labels=True)>
- `2017 | 2017_98_UDSLB`: face `FEMININO`, cranio `MASCULINO`, M=67.3%, confianca `MEDIA`, flags: <Flags(allows_duplicate_labels=True)>
- `2019 | 2019_68_FGL`: face `FEMININO`, cranio `MASCULINO`, M=55.1%, confianca `BAIXA`, flags: <Flags(allows_duplicate_labels=True)>
