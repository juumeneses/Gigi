# Relatório completo do repositório Gigi

Gerado em: 2026-08-18 21:07

> Relatório local para orientar próximos passos. Como há fotografias de rostos e crânios, manter este material em ambiente privado.

## 0. Trilha principal: estamos usando o `Modelo_novo`

Este projeto agora deve ser pensado a partir da pasta `Modelo_novo`. Os treinamentos antigos de faces e crânios continuam importantes, mas funcionam como **fontes de dados e modelos auxiliares**. A pesquisa principal de matching face-crânio fica no `Modelo_novo`, porque ele foi criado para não mexer nem sobrescrever `Treinamento_faces`, `Treinamento_cranios` ou `Matching_cranio_face`.

### Como o `Modelo_novo` deve funcionar

1. **Construir índice de pares**: lê as faces aprovadas, o sexo estimado/curado das faces, os crânios disponíveis, as previsões de sexo craniano e monta `Modelo_novo/data/manifests/metadata_pares.csv`.
2. **Criar splits sem vazamento**: separa treino, validação e teste por pessoa, não por imagem. Também cria testes leave-one-year-out para medir generalização por ano.
3. **Pré-processar imagens**: gera versões padronizadas/normalizadas sem apagar originais. Face e crânio entram em formatos comparáveis para o modelo.
4. **Rodar baselines**: compara aleatório, apenas sexo e shape64 atual para saber se o modelo neural realmente melhora algo.
5. **Treinar modelo two-tower**: usa uma torre para face e outra para crânio, gera embeddings e calcula similaridade. O resultado é ranking, não conclusão pericial.
6. **Avaliar retrieval**: usa Top-1, Top-3, Top-5, Recall@10 e MRR; também avalia por ano e por estudos de ablação.
7. **Usar na prática**: na tela “Comparar 10 crânios”, a pessoa envia uma face frontal e três vistas de cada candidato (`CF`, `CE`, `CD`), e a app retorna o ranking de similaridade aprendida.

### O que já está funcionando no `Modelo_novo`

- App Streamlit em `Modelo_novo/app.py`, com telas: Comparar 10 crânios, Resumo, Dados e pares, Pré-processamento, Treinamento, Avaliação, Ranking, Análise de erros e Logs.
- CLI em `Modelo_novo/cli.py` com comandos: `build-index`, `create-splits`, `preprocess`, `baselines`, `train`, `ablate-views`, `loyo`, `multiseed-loyo`, `ensemble-rankings`, `simulate-gallery`, `audit-domains`, `normalize-images`.
- `metadata_pares.csv` já existe e concentra os pares face-crânio auditáveis.
- Splits congelados já existem para treino, validação, teste e leave-one-year-out por ano.
- Baselines já foram gerados em `Modelo_novo/reports/baselines/`.
- Auditoria visual por domínio/ano já foi gerada em `Modelo_novo/reports/domain_audit/`.
- Estudos já existem: gallery-10, domain normalized, hard negatives, ensembles, leave-one-year-out, multiseed e ablação por vistas.
- O fluxo preserva os resultados de sexo/gênero já existentes, mas para escrita científica o termo recomendado é **sexo estimado**.

### O que ainda precisa amadurecer no `Modelo_novo`

- Melhorar a portabilidade removendo caminhos absolutos legados dos CSVs antigos.
- Decidir quais resultados devem ficar versionados e quais devem ser regeneráveis, para evitar conflitos no Git.
- Consolidar uma configuração final oficial antes de usar o teste como número final.
- Explicar claramente que o score é **similaridade/ranking experimental**, não probabilidade de identificação.


## 1. Visão geral do repositório

| area | scripts_py | csvs | imagens | modelos_embeddings |
| --- | --- | --- | --- | --- |
| Treinamento_faces | 22 | 2505 | 2538 | 68 |
| Treinamento_cranios | 14 | 1597 | 32068 | 3 |
| Matching_cranio_face | 2 | 0 | 0 | 0 |
| Modelo_novo | 18 | 327 | 20 | 0 |
| Painel_Geral | 1 | 0 | 0 | 0 |

### Bases de imagens por ano

| base | ano | imagens_total |
| --- | --- | --- |
| faces | 2017 | 198 |
| faces | 2019 | 552 |
| faces | 2021 | 204 |
| faces | 2022 | 332 |
| faces | 2023 | 305 |
| cranios | 2017 | 294 |
| cranios | 2019 | 826 |
| cranios | 2021 | 306 |
| cranios | 2022 | 498 |
| cranios | 2023 | 457 |

### Imagens de exemplo

**Faces originais**

![Faces originais](assets_relatorio_gigi/faces_originais_exemplos.jpg)

**Marcações faciais**

![Marcações faciais](assets_relatorio_gigi/faces_marcacoes_exemplos.jpg)

**Curadoria matching**

![Curadoria matching](assets_relatorio_gigi/curadoria_matching_exemplos.jpg)

**Crops FairFace**

![Crops FairFace](assets_relatorio_gigi/fairface_crops_exemplos.jpg)

**Crânios originais**

![Crânios originais](assets_relatorio_gigi/cranios_originais_exemplos.jpg)

**Auditoria visual Modelo_novo**

![Auditoria visual Modelo_novo](assets_relatorio_gigi/modelo_novo_domain_audit.jpg)

## 2. Treinamento de faces

- Estado geral: cadeia funcional com landmarks, qualidade, curadoria de matching, FairFace, curadoria de sexo estimado, modelo próprio, comparação e features.
- `OK_AUTO` e `OK_MANUAL` aparecem como dados aproveitáveis quando `usar_matching=True`/`usar_para_treino_genero=True`.
- Atenção: há caminhos absolutos em CSVs; scripts futuros devem resolver caminhos relativos à raiz do repo.

### Scripts principais de faces

| arquivo | o_que_faz | funcoes_classes |
| --- | --- | --- |
| Treinamento_faces/scripts/00_checar_dataset_faces.py | Confere estrutura/nomenclatura das imagens faciais. | infer_year, main |
| Treinamento_faces/scripts/01_extrair_landmarks_faces.py | Extrai landmarks MediaPipe e previews das faces. | save_lm, main |
| Treinamento_faces/scripts/02_gerar_medidas_faces.py | Gera medidas e features geométricas da face. | dist, area, per, proc, main |
| Treinamento_faces/scripts/03_relatorio_landmarks_html.py | Gera relatório HTML visual dos landmarks. | main |
| Treinamento_faces/scripts/05_analisar_padroes_faces.py | Analisa qualidade/outliers das faces para matching. | main |
| Treinamento_faces/scripts/06_predizer_genero_fairface_faces.py | Roda FairFace nas faces aprovadas e salva probabilidades M/F. | ensure_model, load_model, preprocess, predict, main |
| Treinamento_faces/scripts/07_curadoria_genero_faces.py | Aplica curadoria automática/manual do sexo estimado da face. | build, main |
| Treinamento_faces/scripts/08_preparar_features_genero_faces.py | Monta dataset de treino do modelo próprio de sexo da face. | lm_feats, main |
| Treinamento_faces/scripts/09_treinar_modelo_genero_faces.py | Treina modelo próprio de sexo da face com métricas. | runs_dir, append_run_history, main |
| Treinamento_faces/scripts/10_comparar_genero_fairface_modelo.py | Compara FairFace x modelo próprio e lista divergências. | base, stc, main |
| Treinamento_faces/scripts/11_exportar_matching_faces_com_genero.py | Exporta features de face com sexo final curado para matching. | main |
| Treinamento_faces/scripts/12_exportar_features_faces_final.py | Exporta features finais de face para matching. | person_id_from_stem, resample_closed_contour, normalize_landmarks_for_pca, shape64_from_landmarks, main |
| Treinamento_faces/scripts/13_preprocess_faces_normalizadas.py | Gera versões normalizadas/melhoradas de faces. | normalize_face_image, main |
| Treinamento_faces/scripts/analisar_padroes_faces.py | Módulo de apoio/experimento do projeto. |  |
| Treinamento_faces/scripts/app_revisao_faces.py | App Streamlit de revisão/curadoria de faces. | guess_root, script_path, project_python, build_cmd, bgr_to_pil, get_face_mesh, detect_local, stop_command_flag |
| Treinamento_faces/scripts/checar_dataset_faces.py | Módulo de apoio/experimento do projeto. |  |
| Treinamento_faces/scripts/common_faces.py | Módulo de apoio/experimento do projeto. | as_root, faces_root, resultados_root, year_dir, images_dir, marks_dir, previews_dir, corrected_dir |
| Treinamento_faces/scripts/curadoria_matching_faces.py | Módulo de apoio/experimento do projeto. | detect, rotate, clahe, sharp, candidates, try_fix, visually_good, decide |
| Treinamento_faces/scripts/extrair_landmarks_faces.py | Módulo de apoio/experimento do projeto. |  |
| Treinamento_faces/scripts/gerar_medidas_faces.py | Módulo de apoio/experimento do projeto. |  |
| Treinamento_faces/scripts/relatorio_landmarks_html.py | Módulo de apoio/experimento do projeto. |  |
| Treinamento_faces/scripts/run_pipeline_faces.py | Roda a cadeia completa de faces. | main |

### Resumo da curadoria de matching facial por ano

| ano | registros | ok_qualidade | revisar_qualidade | ok_auto | ok_manual | usar_matching |
| --- | --- | --- | --- | --- | --- | --- |
| 2017 | 99 | 99 | 0 | 92 | 7 | 99 |
| 2019 | 276 | 276 | 0 | 268 | 8 | 276 |
| 2021 | 102 | 102 | 0 | 98 | 4 | 102 |
| 2022 | 166 | 166 | 0 | 163 | 3 | 166 |
| 2023 | 152 | 152 | 0 | 134 | 18 | 152 |

### Status encontrados na curadoria de matching facial

| campo | valor | qtde |
| --- | --- | --- |
| qualidade_landmarks | OK | 795 |
| status_final | OK_AUTO | 755 |
| status_final | OK_MANUAL | 40 |
| status_manual | nan | 755 |
| status_manual | OK_MANUAL | 40 |
| usar_matching | True | 795 |

### Resumo de sexo estimado/curado em faces

| ano | registros | masculino | feminino | indefinido | alta_fairface | manual | usar_treino |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2017 | 99 | 37 | 62 | 0 | 81 | 18 | 99 |
| 2019 | 276 | 134 | 142 | 0 | 217 | 59 | 276 |
| 2021 | 102 | 35 | 67 | 0 | 75 | 27 | 102 |
| 2022 | 166 | 59 | 107 | 0 | 125 | 41 | 166 |
| 2023 | 152 | 51 | 101 | 0 | 111 | 41 | 152 |

### Métricas do modelo próprio de sexo da face

| modelo | n_total | n_treino | n_teste | test_size | random_state | metrica_selecao | accuracy | precision_masculino | recall_masculino | f1_masculino |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 795 | 596 | 199 | 0.25 | 42 | f1_masculino | 0.8291457286432161 | 0.8172043010752689 | 0.8172043010752689 | 0.8172043010752689 |
| svm_rbf | 795 | 596 | 199 | 0.25 | 42 | f1_masculino | 0.7738693467336684 | 0.7553191489361702 | 0.7634408602150538 | 0.7593582887700535 |
| random_forest | 795 | 596 | 199 | 0.25 | 42 | f1_masculino | 0.7336683417085427 | 0.7325581395348837 | 0.6774193548387096 | 0.7039106145251397 |

### Matriz de confusão do modelo próprio de sexo da face

| classe_real | pred_FEMININO | pred_MASCULINO |
| --- | --- | --- |
| real_FEMININO | 92 | 14 |
| real_MASCULINO | 13 | 80 |

### Exemplo da comparação FairFace x modelo próprio

| ano | id_imagem | genero_final | classe_fairface | classe_modelo | concordancia_fairface_modelo | prob_masculino_fairface | prob_feminino_fairface | prob_masculino_modelo | prob_feminino_modelo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2017 | 2017_10_AWS_FFN | MASCULINO | MASCULINO_PROVAVEL | MASCULINO_PROVAVEL | CONCORDANTE_MEDIA | 0.999999 | 1e-06 | 0.710543 | 0.289457 |
| 2017 | 2017_11_BB_FFN | MASCULINO | MASCULINO_PROVAVEL | MASCULINO_PROVAVEL | CONCORDANTE_MEDIA | 0.999876 | 0.000124 | 0.568379 | 0.431621 |
| 2017 | 2017_12_BDSA_FFN | FEMININO | FEMININO_PROVAVEL | FEMININO_PROVAVEL | CONCORDANTE_ALTA | 0.006327 | 0.993673 | 0.162392 | 0.837608 |
| 2017 | 2017_13_BTN_FFN | MASCULINO | MASCULINO_PROVAVEL | MASCULINO_PROVAVEL | CONCORDANTE_MEDIA | 1.0 | 0.0 | 0.792709 | 0.207291 |
| 2017 | 2017_14_CAFM_FFN | MASCULINO | MASCULINO_PROVAVEL | MASCULINO_PROVAVEL | CONCORDANTE_ALTA | 0.999999 | 1e-06 | 0.994049 | 0.005951 |
| 2017 | 2017_15_CBL_FFN | FEMININO | INDEFINIDO | FEMININO_PROVAVEL | INDEFINIDO_REVISAR | 0.499115 | 0.500885 | 0.121851 | 0.878149 |
| 2017 | 2017_16_CFDC_FFN | FEMININO | FEMININO_PROVAVEL | FEMININO_PROVAVEL | CONCORDANTE_MEDIA | 0.162245 | 0.837755 | 0.366801 | 0.633199 |
| 2017 | 2017_17_CG_FFN | FEMININO | FEMININO_PROVAVEL | FEMININO_PROVAVEL | CONCORDANTE_ALTA | 0.01954 | 0.98046 | 0.154978 | 0.845022 |
| 2017 | 2017_18_CHPM_FFN | FEMININO | FEMININO_PROVAVEL | FEMININO_PROVAVEL | CONCORDANTE_ALTA | 0.03466 | 0.96534 | 0.153308 | 0.846692 |
| 2017 | 2017_19_CPPDS_FFN | FEMININO | FEMININO_PROVAVEL | FEMININO_PROVAVEL | CONCORDANTE_ALTA | 0.125423 | 0.874577 | 0.1166 | 0.8834 |
| 2017 | 2017_1_AALF_FFN | MASCULINO | MASCULINO_PROVAVEL | MASCULINO_PROVAVEL | CONCORDANTE_MEDIA | 0.999988 | 1.2e-05 | 0.605871 | 0.394129 |
| 2017 | 2017_20_CRAU_FFN | MASCULINO | MASCULINO_PROVAVEL | MASCULINO_PROVAVEL | CONCORDANTE_ALTA | 0.999999 | 1e-06 | 0.976892 | 0.023108 |

## 3. Treinamento de crânios

- Estado geral: pipeline com checagem de tripletas, contorno, shape 64, medidas, qualidade, silhuetas, treino de sexo craniano, comparação de normalizações e features.
- A base contém vistas `CF`, `CE` e `CD`; o matching principal pode começar em `CF` e evoluir para multivista.
- Atenção: alguns CSVs têm caminhos absolutos de outra máquina (`/Users/giovanni/...`).

### Scripts principais de crânios

| arquivo | o_que_faz | funcoes_classes |
| --- | --- | --- |
| Treinamento_cranios/scripts/00_checar_dataset_cranios.py | Confere estrutura, anos, lados e tripletas de crânios. | main |
| Treinamento_cranios/scripts/01_extrair_contorno_cranios.py | Extrai contorno automático dos crânios. | main |
| Treinamento_cranios/scripts/02_gerar_shape_cranios_64pts.py | Amostra shape/contorno do crânio em 64 pontos. | main |
| Treinamento_cranios/scripts/03_gerar_medidas_cranios.py | Calcula medidas geométricas do crânio. | polygon_area, process_contour, main |
| Treinamento_cranios/scripts/04_analisar_qualidade_cranios.py | Analisa qualidade e status de uso dos crânios. | main |
| Treinamento_cranios/scripts/05_preparar_silhuetas_sexo_cranio.py | Gera silhuetas 256 para classificador de sexo craniano. | prepare_silhouettes_for_kind, main |
| Treinamento_cranios/scripts/06_juntar_genero_faces_cranios.py | Cruza sexo curado das faces com crânios por pessoa. | col_or_default, face_person_id, main |
| Treinamento_cranios/scripts/07_treinar_sexo_cranio.py | Treina classificador de sexo craniano. | stop_flag_path, stop_requested, _metric_text, epoch_progress_callback, runs_dir, append_run_history, keras_history_summary, fallback_predictions |
| Treinamento_cranios/scripts/08_exportar_features_cranios_final.py | Exporta features finais dos crânios para matching. | normalize_keys, main |
| Treinamento_cranios/scripts/09_preprocess_cranios_normalizados.py | Gera normalizações/variantes dos crânios. | to_gray, gray_to_bgr, foreground_mask, percentile_clip_gray, normalize_atual, normalize_monografia_equalizacao, normalize_clahe_suave, normalize_pe... |
| Treinamento_cranios/scripts/10_comparar_normalizacoes_sexo_cranio.py | Compara normalizações do treino craniano. | run_step, main |
| Treinamento_cranios/scripts/app_revisao_cranios.py | App Streamlit de revisão/treino de crânios. | guess_root, project_python, stop_training_flag, current_process_pid_path, training_lock_path, _pid_alive, force_stop_pid, force_stop_training |
| Treinamento_cranios/scripts/common_cranios.py | Módulo de apoio/experimento do projeto. | as_root, dados_root, normalizados_root, normalized_variants_root, resultados_root, side_dir, normalized_side_dir, normalized_variant_side_dir |
| Treinamento_cranios/scripts/run_pipeline_cranios.py | Roda a cadeia completa de crânios. | main |

### Resumo de features/sexo craniano por ano

| ano | registros | pessoas | ok | usar_matching | masculino | feminino | confianca_alta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2017 | 97 | 97 | 94 | 94 | 29 | 65 | 73 |
| 2019 | 265 | 265 | 237 | 237 | 98 | 136 | 185 |
| 2021 | 100 | 100 | 95 | 95 | 30 | 65 | 79 |
| 2022 | 166 | 166 | 164 | 164 | 57 | 107 | 137 |
| 2023 | 151 | 151 | 146 | 146 | 46 | 98 | 117 |

### Métricas do modelo de sexo craniano

| accuracy | balanced_accuracy | f1_masculino | precision_masculino | recall_masculino | accuracy_pct | balanced_accuracy_pct | f1_pct | precision_pct | recall_pct | run_id | normalizacao | modelo | n_individuos | n_silhuetas | n_treino_individuos | n_validacao_individuos | n_teste_individuos | threshold_masculino | estrategia_voto | class_weight_feminino | class_weight_masculino | learning_rate | fine_tuning_layers | melhor_val_accuracy_pct | train_loss | train_accuracy | val_loss | val_accuracy | best_train_loss | best_train_accuracy | best_val_loss | best_val_accuracy | train_accuracy_pct | val_accuracy_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.868852 | 0.849935 | 0.809524 | 0.836066 | 0.784615 | 86.89 | 84.99 | 80.95 | 83.61 | 78.46 | cranio_genero_vgg16_finetune_20260722_144701 | atual | vgg16_finetune_block5_imagenet_silhueta_256_threshold_multivista | 731 | 2193 | 438 | 110 | 183 | 0.5 | ponderada | 0.779359 | 1.394904 | 1e-05 | block5_conv1,block5_conv2,block5_conv3,block5_pool | 82.73 | 0.036028 | 0.999239 | 0.412875 | 0.818182 | 0.036028 | 1.0 | 0.40404 | 0.827273 | 100.0 | 82.73 |

### Métricas por vista do modelo de sexo craniano

| run_id | modelo | lado | n_vistas_teste | accuracy | balanced_accuracy | f1_masculino | precision_masculino | recall_masculino | accuracy_pct | balanced_accuracy_pct | f1_pct | precision_pct | recall_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cranio_genero_vgg16_finetune_20260722_144701 | vgg16_finetune_block5_imagenet_silhueta_256_threshold_multivista | CD | 183 | 0.863388 | 0.856063 | 0.81203 | 0.794118 | 0.830769 | 86.34 | 85.61 | 81.2 | 79.41 | 83.08 |
| cranio_genero_vgg16_finetune_20260722_144701 | vgg16_finetune_block5_imagenet_silhueta_256_threshold_multivista | CE | 183 | 0.808743 | 0.78605 | 0.724409 | 0.741935 | 0.707692 | 80.87 | 78.6 | 72.44 | 74.19 | 70.77 |
| cranio_genero_vgg16_finetune_20260722_144701 | vgg16_finetune_block5_imagenet_silhueta_256_threshold_multivista | CF | 183 | 0.868852 | 0.843025 | 0.803279 | 0.859649 | 0.753846 | 86.89 | 84.3 | 80.33 | 85.96 | 75.38 |

### Matriz de confusão do sexo craniano

| classe_real | pred_FEMININO | pred_MASCULINO |
| --- | --- | --- |
| real_FEMININO | 108 | 10 |
| real_MASCULINO | 14 | 51 |

### Comparativo de normalizações cranianas

| accuracy | balanced_accuracy | f1_masculino | precision_masculino | recall_masculino | accuracy_pct | balanced_accuracy_pct | f1_pct | precision_pct | recall_pct | run_id | normalizacao | modelo | n_individuos | n_silhuetas | n_treino_individuos | n_validacao_individuos | n_teste_individuos | threshold_masculino | estrategia_voto | class_weight_feminino | class_weight_masculino | learning_rate | fine_tuning_layers | melhor_val_accuracy_pct | train_loss | train_accuracy | val_loss | val_accuracy | best_train_loss | best_train_accuracy | best_val_loss | best_val_accuracy | train_accuracy_pct | val_accuracy_pct | status | comparados | acertos | divergencias | indefinidos | acertos_pct | divergencias_pct | indefinidos_pct | melhor_normalizacao |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.874317 | 0.878594 | 0.836879 | 0.786667 | 0.893939 | 87.43 | 87.86 | 83.69 | 78.67 | 89.39 | cranio_genero_vgg16_finetune_20260721_033635 | atual | vgg16_finetune_block5_imagenet_silhueta_256_threshold_multivista | 731 | 2193 | 438 | 110 | 183 | 0.5 | laterais | 0.779359 | 1.394904 | 1e-05 | block5_conv1,block5_conv2,block5_conv3,block5_pool | 81.82 | 0.048107 | 0.999239 | 0.488161 | 0.818182 | 0.047776 | 1.0 | 0.484351 | 0.818182 | 100.0 | 81.82 | OK | 731 | 689 | 42 | 0 | 94.25 | 5.75 | 0.0 | True |
| 0.874317 | 0.871989 | 0.832117 | 0.802817 | 0.863636 | 87.43 | 87.2 | 83.21 | 80.28 | 86.36 | cranio_genero_vgg16_finetune_20260721_050654 | monografia_equalizacao | vgg16_finetune_block5_imagenet_silhueta_256_threshold_multivista | 731 | 2193 | 438 | 110 | 183 | 0.5 | laterais | 0.779359 | 1.394904 | 1e-05 | block5_conv1,block5_conv2,block5_conv3,block5_pool | 80.3 | 0.067607 | 0.996956 | 0.507952 | 0.80303 | 0.067607 | 0.996956 | 0.503152 | 0.80303 | 99.7 | 80.3 | OK | 731 | 683 | 48 | 0 | 93.43 | 6.57 | 0.0 | False |
| 0.868852 | 0.871018 | 0.828571 | 0.783784 | 0.878788 | 86.89 | 87.1 | 82.86 | 78.38 | 87.88 | cranio_genero_vgg16_finetune_20260721_070620 | clahe_suave | vgg16_finetune_block5_imagenet_silhueta_256_threshold_multivista | 731 | 2193 | 438 | 110 | 183 | 0.5 | ponderada | 0.779359 | 1.394904 | 1e-05 | block5_conv1,block5_conv2,block5_conv3,block5_pool | 81.52 | 0.036108 | 0.999239 | 0.486846 | 0.80303 | 0.036108 | 0.999239 | 0.471003 | 0.815152 | 99.92 | 81.52 | OK | 731 | 689 | 42 | 0 | 94.25 | 5.75 | 0.0 | False |
| 0.825137 | 0.843434 | 0.789474 | 0.697674 | 0.909091 | 82.51 | 84.34 | 78.95 | 69.77 | 90.91 | cranio_genero_vgg16_finetune_20260721_083658 | percentile_clip | vgg16_finetune_block5_imagenet_silhueta_256_threshold_multivista | 731 | 2193 | 438 | 110 | 183 | 0.4 | laterais | 0.779359 | 1.394904 | 1e-05 | block5_conv1,block5_conv2,block5_conv3,block5_pool | 81.21 | 0.069746 | 0.99239 | 0.51024 | 0.812121 | 0.069746 | 0.993912 | 0.485175 | 0.812121 | 99.39 | 81.21 | OK | 731 | 670 | 61 | 0 | 91.66 | 8.34 | 0.0 | False |
| 0.808743 | 0.830614 | 0.774194 | 0.674157 | 0.909091 | 80.87 | 83.06 | 77.42 | 67.42 | 90.91 | cranio_genero_vgg16_finetune_20260721_012454 | original | vgg16_finetune_block5_imagenet_silhueta_256_threshold_multivista | 731 | 2193 | 438 | 110 | 183 | 0.35 | laterais | 0.779359 | 1.394904 | 1e-05 | block5_conv1,block5_conv2,block5_conv3,block5_pool | 81.21 | 0.043956 | 1.0 | 0.509086 | 0.787879 | 0.043956 | 1.0 | 0.491211 | 0.812121 | 100.0 | 81.21 | OK | 731 | 673 | 58 | 0 | 92.07 | 7.93 | 0.0 | False |
| 0.765027 | 0.789821 | 0.72956 | 0.623656 | 0.878788 | 76.5 | 78.98 | 72.96 | 62.37 | 87.88 | cranio_genero_vgg16_finetune_20260721_104652 | crop_pad_percentile | vgg16_finetune_block5_imagenet_silhueta_256_threshold_multivista | 731 | 2183 | 438 | 110 | 183 | 0.35 | ponderada | 0.777646 | 1.400428 | 1e-05 | block5_conv1,block5_conv2,block5_conv3,block5_pool | 76.22 | 0.075956 | 0.996942 | 0.541838 | 0.743902 | 0.075956 | 0.997706 | 0.530943 | 0.762195 | 99.77 | 76.22 | OK | 731 | 656 | 75 | 0 | 89.74 | 10.26 | 0.0 | False |

## 4. Matching crânio-face V1

- Local: `Matching_cranio_face`.
- A app atual recebe upload de uma face e múltiplos crânios candidatos.
- Ranking combina shape/contorno de 64 pontos, qualidade aproximada e compatibilidade de sexo estimado.
- Útil como baseline/rascunho operacional; ainda não é o two-tower principal do roteiro.

### Scripts do matching V1

| arquivo | o_que_faz | funcoes_classes |
| --- | --- | --- |
| Matching_cranio_face/scripts/app_matching_cranio_face.py | App V1 de ranking por shape e sexo auxiliar. |  |
| Matching_cranio_face/scripts/common_matching.py | Módulo de apoio/experimento do projeto. | read_image, uploaded_to_temp, write_temp_image, detect_face_landmarks, face_shape64, draw_face_shape64, cranio_shape64, cranio_shape64_preview |

## 5. Modelo_novo

- Local: `Modelo_novo`.
- Já contém app, CLI, configs, indexação, pré-processamento, splits, baselines, treinamento, inferência, retrieval, estudos e testes.
- Já gerou metadata, splits, baselines, auditoria visual, simulações gallery-10, runs e estudos leave-one-year-out/multiseed.
- É o melhor lugar para continuar sem quebrar os treinamentos antigos.

### Scripts/módulos do Modelo_novo

| arquivo | o_que_faz | funcoes_classes |
| --- | --- | --- |
| Modelo_novo/src/__init__.py | Módulo de apoio/experimento do projeto. |  |
| Modelo_novo/src/baselines.py | Módulo de apoio/experimento do projeto. | evaluate_random, evaluate_sex_only, _shape_distance, evaluate_shape64, run_baselines |
| Modelo_novo/src/domain_preprocessing.py | Módulo de apoio/experimento do projeto. | _stats, _contact_sheet, audit_domains, _face_landmarks, _letterbox, normalize_face, normalize_skull, normalize_skull_image |
| Modelo_novo/src/experiments.py | Módulo de apoio/experimento do projeto. | provenance, save_provenance |
| Modelo_novo/src/gallery_simulator.py | Módulo de apoio/experimento do projeto. | _ranking_files_for_variant, _metrics, _auc, _best_threshold, _cross_validated_rejection, _sample_gallery, _sample_fast, simulate_galleries |
| Modelo_novo/src/index_builder.py | Módulo de apoio/experimento do projeto. | _read, face_person_id, _sex_label, _existing_face_path, validate_metadata, build_metadata |
| Modelo_novo/src/inference.py | Módulo de apoio/experimento do projeto. | ensemble_checkpoint_paths, _tensor, load_ensemble, rank_uploaded_candidates |
| Modelo_novo/src/model.py | Módulo de apoio/experimento do projeto. | require_torch, symmetric_infonce, symmetric_hard_negative_loss, Encoder, GeometryEncoder, TwoTower, TwoTower, directional |
| Modelo_novo/src/paths.py | Módulo de apoio/experimento do projeto. | repo_path, relative_repo_path, ensure_project_dirs |
| Modelo_novo/src/preprocess.py | Módulo de apoio/experimento do projeto. | _letterbox, preprocess_manifest |
| Modelo_novo/src/retrieval.py | Módulo de apoio/experimento do projeto. | retrieval_metrics, add_sex_penalty, sampled_gallery_metrics |
| Modelo_novo/src/splits.py | Módulo de apoio/experimento do projeto. | _allocate, create_splits, create_leave_one_year_out |
| Modelo_novo/src/studies.py | Módulo de apoio/experimento do projeto. | _append_summary, run_view_ablation, run_loyo_fold, summarize_multiseed, run_multiseed_loyo, average_similarity_matrices, build_multiseed_ensemble, ... |
| Modelo_novo/src/training.py | Módulo de apoio/experimento do projeto. | _stack, _seed, make_dataset_class, _encode, mine_global_hard_neighbors, hard_negative_batches, apply_domain_normalized_paths, apply_shape64_features |

### Metadata de pares

| item | valor |
| --- | --- |
| pares no metadata | 810 |
| coluna disponível: ano | sim |

Exemplo de pares:

| ano | id_pessoa | face_path |
| --- | --- | --- |
| 2017 | 2017_10_AWS | Treinamento_faces/dados_originais/faces/2017/images/2017_10_AWS_FFN.jpg |
| 2017 | 2017_11_BB | Treinamento_faces/dados_originais/faces/2017/images/2017_11_BB_FFN.jpg |
| 2017 | 2017_12_BDSA | Treinamento_faces/dados_originais/faces/2017/images/2017_12_BDSA_FFN.jpg |
| 2017 | 2017_13_BTN | Treinamento_faces/dados_originais/faces/2017/images/2017_13_BTN_FFN.jpg |
| 2017 | 2017_14_CAFM | Treinamento_faces/dados_originais/faces/2017/images/2017_14_CAFM_FFN.jpg |
| 2017 | 2017_15_CBL | Treinamento_faces/dados_originais/faces/2017/images/2017_15_CBL_FFN.jpg |
| 2017 | 2017_16_CFDC | Treinamento_faces/dados_originais/faces/2017/images/2017_16_CFDC_FFN.jpg |
| 2017 | 2017_17_CG | Treinamento_faces/dados_originais/faces/2017/images/2017_17_CG_FFN.jpg |
| 2017 | 2017_18_CHPM | Treinamento_faces/dados_originais/faces/2017/images/2017_18_CHPM_FFN.jpg |
| 2017 | 2017_19_CPPDS | Treinamento_faces/dados_originais/faces/2017/images/2017_19_CPPDS_FFN.jpg |

### Splits do Modelo_novo

| split | ano | qtde |
| --- | --- | --- |
| test | 2017 | 14 |
| test | 2019 | 35 |
| test | 2021 | 14 |
| test | 2022 | 24 |
| test | 2023 | 21 |
| train | 2017 | 64 |
| train | 2019 | 161 |
| train | 2021 | 67 |
| train | 2022 | 114 |
| train | 2023 | 101 |
| val | 2017 | 14 |
| val | 2019 | 35 |
| val | 2021 | 14 |
| val | 2022 | 24 |
| val | 2023 | 21 |

### Baselines do Modelo_novo

| n_queries | n_candidates | top_1 | top_3 | top_5 | recall_at_10 | mrr | median_rank | mean_rank | baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 108 | 108 | 0.0 | 0.0092592592592592 | 0.037037037037037 | 0.0833333333333333 | 0.038227373916024 | 57.5 | 54.25 | aleatorio |
| 108 | 108 | 0.0185185185185185 | 0.0555555555555555 | 0.0925925925925925 | 0.1851851851851851 | 0.0836333673467665 | 27.5 | 30.46296296296296 | apenas_sexo |
| 108 | 108 | 0.0 | 0.037037037037037 | 0.0462962962962962 | 0.1203703703703703 | 0.0462374044971831 | 52.5 | 54.47222222222222 | shape64_atual |

### Leave-one-year-out

| holdout_year | variant | best_epoch | n_queries | n_candidates | top_1 | top_3 | top_5 | recall_at_10 | mrr | median_rank | mean_rank | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2017 | original | 14 | 92 | 92 | 0.0434782608695652 | 0.0869565217391304 | 0.1304347826086956 | 0.2717391304347826 | 0.1088185051056078 | 24.5 | 31.5 | studies/leave_one_year_out/holdout_2017_20260806T160637Z |
| 2019 | original | 7 | 231 | 231 | 0.0043290043290043 | 0.0259740259740259 | 0.0476190476190476 | 0.0952380952380952 | 0.0428981762546622 | 65.0 | 78.46320346320347 | studies/leave_one_year_out/holdout_2019_20260806T162602Z |
| 2019 | domain_normalized | 11 | 231 | 231 | 0.0129870129870129 | 0.0606060606060606 | 0.1125541125541125 | 0.1428571428571428 | 0.0636129939436242 | 62.0 | 74.38095238095238 | studies/leave_one_year_out/holdout_2019_20260806T193500Z |
| 2021 | original | 12 | 95 | 95 | 0.0421052631578947 | 0.1473684210526315 | 0.2105263157894736 | 0.3578947368421052 | 0.1426978104304135 | 19.0 | 22.94736842105263 | studies/leave_one_year_out/holdout_2021_20260806T164728Z |
| 2022 | original | 10 | 162 | 162 | 0.0185185185185185 | 0.0432098765432098 | 0.0802469135802469 | 0.1604938271604938 | 0.0759925633978033 | 32.0 | 41.666666666666664 | studies/leave_one_year_out/holdout_2022_20260806T170715Z |
| 2023 | original | 9 | 143 | 143 | 0.0209790209790209 | 0.0909090909090909 | 0.1258741258741259 | 0.1818181818181818 | 0.0829530248750747 | 40.0 | 46.62237762237762 | studies/leave_one_year_out/holdout_2023_20260806T180706Z |
| 2021 | domain_normalized | 11 | 95 | 95 | 0.031578947368421 | 0.1473684210526315 | 0.2421052631578947 | 0.3894736842105263 | 0.1507366449125903 | 14.0 | 20.96842105263158 | studies/leave_one_year_out/holdout_2021_20260806T194626Z |
| 2017 | domain_normalized | 14 | 92 | 92 | 0.0760869565217391 | 0.1847826086956521 | 0.2391304347826087 | 0.3152173913043478 | 0.1731037800993264 | 18.0 | 24.42391304347826 | studies/leave_one_year_out/holdout_2017_20260806T230621Z |
| 2022 | domain_normalized | 12 | 162 | 162 | 0.0185185185185185 | 0.0987654320987654 | 0.1358024691358024 | 0.2592592592592592 | 0.0975659680451672 | 33.0 | 40.45679012345679 | studies/leave_one_year_out/holdout_2022_20260806T231555Z |
| 2023 | domain_normalized | 10 | 143 | 143 | 0.0349650349650349 | 0.0839160839160839 | 0.1118881118881118 | 0.2027972027972027 | 0.1016126190809352 | 30.0 | 43.64335664335665 | studies/leave_one_year_out/holdout_2023_20260806T232217Z |
| 2017 | domain_normalized_hard_negatives | 18 | 92 | 92 | 0.0326086956521739 | 0.1630434782608695 | 0.2391304347826087 | 0.391304347826087 | 0.1548343607876202 | 15.0 | 20.83695652173913 | studies/leave_one_year_out/holdout_2017_20260806T234055Z |
| 2019 | domain_normalized_hard_negatives | 9 | 231 | 231 | 0.0216450216450216 | 0.0606060606060606 | 0.0779220779220779 | 0.1515151515151515 | 0.0664257940974688 | 60.0 | 76.74458874458874 | studies/leave_one_year_out/holdout_2019_20260806T235217Z |
| 2021 | domain_normalized_hard_negatives | 20 | 95 | 95 | 0.0947368421052631 | 0.2210526315789473 | 0.3263157894736842 | 0.431578947368421 | 0.2104152327541613 | 16.0 | 20.210526315789473 | studies/leave_one_year_out/holdout_2021_20260806T235612Z |
| 2022 | domain_normalized_hard_negatives | 8 | 162 | 162 | 0.037037037037037 | 0.0925925925925925 | 0.1296296296296296 | 0.2160493827160493 | 0.1040958375748832 | 30.0 | 36.60493827160494 | studies/leave_one_year_out/holdout_2022_20260807T001328Z |
| 2023 | domain_normalized_hard_negatives | 20 | 143 | 143 | 0.0629370629370629 | 0.1258741258741259 | 0.1468531468531468 | 0.2657342657342657 | 0.1289705604175821 | 29.0 | 45.37762237762238 | studies/leave_one_year_out/holdout_2023_20260807T001825Z |

### Ablation por vistas

| study_id | combination | views | best_epoch | n_queries | n_candidates | top_1 | top_3 | top_5 | recall_at_10 | mrr | median_rank | mean_rank | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| view_ablation | CF_CE_CD | CF+CE+CD | 9 | 108 | 108 | 0.1203703703703703 | 0.2314814814814814 | 0.3333333333333333 | 0.4907407407407407 | 0.2322513862162338 | 11.0 | 21.60185185185185 | studies/view_ablation/CF_CE_CD_20260806T155053Z |
| view_ablation | CE_CD | CE+CD | 7 | 108 | 108 | 0.1203703703703703 | 0.2222222222222222 | 0.2962962962962963 | 0.4722222222222222 | 0.2257588550381755 | 13.0 | 20.675925925925927 | studies/view_ablation/CE_CD_20260806T151801Z |
| view_ablation | CF_CE | CF+CE | 7 | 108 | 108 | 0.1018518518518518 | 0.2037037037037037 | 0.2777777777777778 | 0.4444444444444444 | 0.2063661123683072 | 14.0 | 21.76851851851852 | studies/view_ablation/CF_CE_20260806T153002Z |
| view_ablation | CF | CF | 15 | 108 | 108 | 0.0925925925925925 | 0.1944444444444444 | 0.2407407407407407 | 0.3611111111111111 | 0.1904190254214122 | 17.0 | 22.953703703703702 | studies/view_ablation/CF_20260806T150835Z |
| view_ablation | CF_CD | CF+CD | 15 | 108 | 108 | 0.074074074074074 | 0.1944444444444444 | 0.2685185185185185 | 0.4074074074074074 | 0.1884100487186061 | 14.0 | 23.03703703703704 | studies/view_ablation/CF_CD_20260806T153928Z |

### Melhores resumos de estudos encontrados

| arquivo | linhas | mrr | mrr_mean |
| --- | --- | --- | --- |
| Modelo_novo/studies/ensembles/ensemble_domain_normalized_hard_negatives/summary.csv | 5 | 0.2543 |  |
| Modelo_novo/studies/validation_multiseed/multiview_faces_global_hard_attention/summary.csv | 2 |  | 0.2353 |
| Modelo_novo/studies/view_ablation/summary.csv | 5 | 0.2323 |  |
| Modelo_novo/studies/validation_multiseed/validation_control_visual/summary.csv | 2 |  | 0.2196 |
| Modelo_novo/studies/ensembles/ensemble_domain_normalized_gallery10_tuned/summary.csv | 5 | 0.2119 |  |
| Modelo_novo/studies/validation_multiseed/multiview_faces_control_ffn/summary.csv | 2 |  | 0.2113 |
| Modelo_novo/studies/leave_one_year_out/summary.csv | 15 | 0.2104 |  |
| Modelo_novo/studies/multiseed_loyo/domain_normalized_hard_negatives/summary.csv | 6 |  | 0.2086 |
| Modelo_novo/studies/validation_tuning/gallery10/summary.csv | 3 |  | 0.1997 |
| Modelo_novo/studies/multiseed_loyo/domain_normalized_gallery10_tuned/summary.csv | 6 |  | 0.1923 |
| Modelo_novo/studies/validation_multiseed/validation_domain_robust_grayscale/summary.csv | 2 |  | 0.1914 |
| Modelo_novo/studies/multiseed_loyo/domain_normalized_global_hard_mean/summary.csv | 6 |  | 0.1694 |
| Modelo_novo/studies/multiseed_loyo/domain_normalized_global_hard_attention/summary.csv | 2 |  | 0.1472 |
| Modelo_novo/studies/multiseed_loyo/domain_normalized_hybrid_shape64/summary.csv | 2 |  | 0.1469 |
| Modelo_novo/studies/multiseed_loyo/multiview_faces_global_hard_attention/summary.csv | 6 |  | 0.1465 |
| Modelo_novo/studies/multiseed_loyo/domain_normalized_attention_only/summary.csv | 2 |  | 0.1454 |

## 6. O que está OK

- Repositório organizado em faces, crânios, matching V1 e modelo novo.
- Dados originais presentes para 2017, 2019, 2021, 2022 e 2023.
- Faces: pipeline completo de landmarks, qualidade, curadoria, FairFace, modelo próprio e features.
- Crânios: pipeline completo de contorno, shape 64, medidas, qualidade, silhuetas, modelo de sexo e features.
- `Modelo_novo`: estrutura boa para pesquisa com configs, splits, baselines, estudos e métricas.
- Há imagens de exemplo, crops, marcações e galerias para revisão visual.

## 7. O que não está OK ou merece atenção

- Caminhos absolutos em CSVs atrapalham uso em dois computadores.
- Resultados versionados junto com código ajudam continuidade, mas podem gerar conflitos quando dois computadores rodam pipelines.
- Matching V1 ainda não é o modelo two-tower principal.
- Confirmar Git LFS para `.keras`, `.pkl`, `.npz` e imagens grandes.
- Caches/ambiente local (`.venv`, `.cache`, `.DS_Store`) devem ficar fora dos commits.
- Para texto científico, preferir “sexo estimado”; manter `genero` nos nomes só por compatibilidade.

## 8. Próximos passos recomendados

1. Usar `Modelo_novo` como trilha principal de pesquisa, mantendo os pipelines antigos como produtores de dados.
2. Normalizar caminhos dos CSVs para relativos à raiz do repo.
3. Consolidar um botão/CLI de “reconstruir índice” que regenere `metadata_pares.csv` sem mexer nos originais.
4. Definir baseline oficial: aleatório, apenas sexo, shape64 V1 e modelo neural.
5. Rodar avaliação por indivíduo e leave-one-year-out com Top-1, Top-3, Top-5, MRR e CMC.
6. Separar resultados finais importantes de intermediários descartáveis para reduzir conflitos no Git.

## 9. Arquivos principais para abrir ou enviar ao ChatGPT/work

- Relatório atual: `docs/relatorio_estado_repositorio_gigi.md`.
- Contact sheets: `docs/assets_relatorio_gigi/`.
- Metadata principal: `Modelo_novo/data/manifests/metadata_pares.csv`.
- Baselines: `Modelo_novo/reports/baselines/metricas_baselines.csv`.
- Métricas face: `Treinamento_faces/resultados/genero_faces/metricas_modelo_genero_faces.csv`.
- Métricas crânio: `Treinamento_cranios/resultados/sexo_cranio/metricas_modelo_sexo_cranio.csv`.
