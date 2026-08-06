# Relatorio Informativo - Treinamento de Sexo Craniano

Este relatorio resume os resultados obtidos no programa que estima o sexo da pessoa a partir de imagens de cranio. O objetivo e reunir, de forma simples e completa, as informacoes observadas no ultimo teste salvo e no historico de runs, para permitir uma analise posterior por outra IA ou por um avaliador humano.

Base analisada:

- Projeto: `/Users/giovanni/Documents/GitHub/Gigi`
- Modulo: `Treinamento_cranios`
- Pasta principal de resultados: `Treinamento_cranios/resultados/sexo_cranio`
- Historico de runs: `Treinamento_cranios/resultados/runs/historico_treinos_cranios.csv`

## 1. Ultimo Teste Salvo

O ultimo teste salvo foi:

```text
run_id: cranio_genero_article_vgg16_20260630_035739
tipo: genero_cranio_artigo
modelo: article_vgg16_imagenet_silhueta_256_aug_flip_rot5_voto_3vistas
timestamp: 2026-06-30T03:57:39.715747
```

Parametros usados:

```json
{
  "backend": "article_vgg16",
  "weights": "imagenet",
  "image_size": 256,
  "batch_size": 8,
  "epochs": 18,
  "learning_rate": 0.000002,
  "augmentation": "flip_horizontal_rotacao_pm5",
  "voto": "CF_CE_CD"
}
```

Descricao operacional:

- O modelo usa VGG16 com pesos `imagenet`.
- A entrada sao silhuetas binarizadas do cranio.
- O tamanho usado foi `256x256`.
- A VGG16 fica congelada, e apenas a cabeca neural aprende.
- O treino usa augmentation com:
  - imagem original;
  - flip horizontal;
  - rotacao -5 graus;
  - rotacao +5 graus.
- A predicao final por individuo usa as tres vistas:
  - `CF`: cranio frontal;
  - `CE`: cranio esquerdo;
  - `CD`: cranio direito.
- A decisao final usa voto pelas tres vistas.
- O threshold usado para masculino/feminino e `0.5`.

## 2. Dataset Usado No Ultimo Teste

Dados principais:

| Indicador | Valor |
|---|---:|
| Individuos usados | 731 |
| Silhuetas usadas no treino/teste | 2.193 |
| Individuos no treino | 548 |
| Individuos no teste | 183 |
| Imagens aumentadas no treino | 6.576 |

Distribuicao das classes por individuo:

| Classe real | Quantidade | Percentual |
|---|---:|---:|
| FEMININO | 468 | 64,02% |
| MASCULINO | 263 | 35,98% |

Distribuicao por ano:

| Ano | Feminino | Masculino | Total |
|---|---:|---:|---:|
| 2017 | 61 | 33 | 94 |
| 2019 | 139 | 95 | 234 |
| 2021 | 65 | 30 | 95 |
| 2022 | 106 | 58 | 164 |
| 2023 | 97 | 47 | 144 |
| Total | 468 | 263 | 731 |

Distribuicao percentual por ano:

| Ano | Feminino | Masculino |
|---|---:|---:|
| 2017 | 64,89% | 35,11% |
| 2019 | 59,40% | 40,60% |
| 2021 | 68,42% | 31,58% |
| 2022 | 64,63% | 35,37% |
| 2023 | 67,36% | 32,64% |

Observacao importante:

- O dataset esta desbalanceado.
- Ha mais casos femininos do que masculinos.
- A proporcao geral e aproximadamente 64% feminino e 36% masculino.
- Esse desbalanceamento pode induzir o modelo a prever feminino com muita frequencia.

## 3. Silhuetas, Tripletas E Qualidade

Arquivos analisados:

- `silhuetas_sexo_cranio.csv`
- `falhas_silhuetas_sexo_cranio.csv`
- `tripletas_cranios.csv`
- `qualidade_cranios.csv`

Resumo das silhuetas:

| Indicador | Valor |
|---|---:|
| Silhuetas com status OK | 2.324 |
| Silhuetas CF | 779 |
| Silhuetas CE | 771 |
| Silhuetas CD | 774 |

Resumo de tripletas:

| Indicador | Valor |
|---|---:|
| Individuos em `tripletas_cranios.csv` | 795 |
| Com tripleta completa | 791 |

Falhas de silhueta:

| Lado | Falhas |
|---|---:|
| CF | 12 |
| CE | 20 |
| CD | 17 |

Qualidade craniana:

| Status | Quantidade |
|---|---:|
| OK | 736 |
| REVISAR | 43 |

Qualidade por ano:

| Ano | OK | REVISAR | Total |
|---|---:|---:|---:|
| 2017 | 94 | 3 | 97 |
| 2019 | 237 | 28 | 265 |
| 2021 | 95 | 5 | 100 |
| 2022 | 164 | 2 | 166 |
| 2023 | 146 | 5 | 151 |
| Total | 736 | 43 | 779 |

Observacao:

- Existem mais silhuetas OK do que individuos efetivamente usados no treino final.
- O dataset final de sexo craniano ficou com 731 individuos.
- Parte dos individuos pode ter sido excluida por falta de rotulo, problemas de associacao com faces ou ausencia/falha em alguma vista necessaria.

## 4. Metricas Do Ultimo Teste

Metricas gerais do teste:

| Metrica | Valor |
|---|---:|
| Accuracy | 66,12% |
| F1 masculino | 27,91% |
| Precision masculino | 60,00% |
| Recall masculino | 18,18% |
| Melhor validation accuracy | 67,33% |

Interpretacao:

- A acuracia geral ficou em 66,12%.
- O F1 masculino ficou baixo: 27,91%.
- A precision masculina foi 60,00%, ou seja, quando o modelo prediz masculino, ele acerta uma parte razoavel.
- O recall masculino foi muito baixo: 18,18%.
- O principal problema e que o modelo quase nao identifica individuos masculinos.

## 5. Matriz De Confusao Do Ultimo Teste

Matriz de confusao:

| Classe real | Predito feminino | Predito masculino |
|---|---:|---:|
| Real feminino | 109 | 8 |
| Real masculino | 54 | 12 |

Leitura da matriz:

- No conjunto de teste havia:
  - 117 femininos;
  - 66 masculinos.
- Dos 117 femininos:
  - 109 foram classificados corretamente como feminino;
  - 8 foram classificados erroneamente como masculino.
- Dos 66 masculinos:
  - 12 foram classificados corretamente como masculino;
  - 54 foram classificados erroneamente como feminino.

Conclusao da matriz:

- O modelo esta fortemente enviesado para predizer feminino.
- O modelo acerta muito bem os femininos.
- O modelo erra a maioria dos masculinos.
- A baixa performance masculina e o principal gargalo do modelo atual.

## 6. Comparacao Face Definida X Cranio Previsto

Arquivo:

- `comparacao_genero_face_cranio.csv`

Resultado geral:

| Resultado | Quantidade | Percentual |
|---|---:|---:|
| ACERTO | 488 | 66,76% |
| DIVERGENCIA | 243 | 33,24% |

Resultado por genero definido na face:

| Genero definido na face | Acerto | Divergencia | Total |
|---|---:|---:|---:|
| FEMININO | 440 | 28 | 468 |
| MASCULINO | 48 | 215 | 263 |
| Total | 488 | 243 | 731 |

Percentual por genero:

| Genero definido na face | Acerto | Divergencia |
|---|---:|---:|
| FEMININO | 94,02% | 5,98% |
| MASCULINO | 18,25% | 81,75% |

Conclusao:

- O modelo acertou 94,02% dos femininos.
- O modelo acertou apenas 18,25% dos masculinos.
- O modelo divergiu em 81,75% dos casos masculinos.
- O problema principal e claramente a deteccao de masculino.

## 7. Distribuicao Das Predicoes

Predicoes finais por individuo:

| Predicao cranio | Quantidade |
|---|---:|
| FEMININO_PROVAVEL | 655 |
| MASCULINO_PROVAVEL | 76 |

Comparacao com a distribuicao real:

| Classe | Real | Predito |
|---|---:|---:|
| FEMININO | 468 | 655 |
| MASCULINO | 263 | 76 |

Interpretacao:

- O modelo esta prevendo feminino em excesso.
- Ele previu 655 femininos, mas o real era 468.
- Ele previu apenas 76 masculinos, mas o real era 263.
- Isso confirma o vies para feminino.

## 8. Confianca Das Predicoes

Distribuicao por confianca:

| Confianca | Quantidade |
|---|---:|
| ALTA | 473 |
| MEDIA | 258 |

Confianca x resultado:

| Confianca | Acerto | Divergencia | Total |
|---|---:|---:|---:|
| ALTA | 360 | 113 | 473 |
| MEDIA | 128 | 130 | 258 |

Percentual por confianca:

| Confianca | Acerto | Divergencia |
|---|---:|---:|
| ALTA | 76,11% | 23,89% |
| MEDIA | 49,61% | 50,39% |

Interpretacao:

- Predicoes de alta confianca sao melhores que as de media confianca.
- Mesmo assim, ha 113 divergencias com confianca alta.
- A confianca alta nao garante que a predicao esteja correta.
- As predicoes de confianca media estao praticamente no nivel de aleatoriedade: 49,61% acerto e 50,39% divergencia.

## 9. Classe Predita X Resultado

| Classe cranio | Acerto | Divergencia | Total |
|---|---:|---:|---:|
| FEMININO_PROVAVEL | 440 | 215 | 655 |
| MASCULINO_PROVAVEL | 48 | 28 | 76 |

Interpretacao:

- Quando o modelo prediz feminino, ele acerta 440 e erra 215.
- Quando prediz masculino, ele acerta 48 e erra 28.
- A predicao masculina e menos frequente, mas nao e inutil.
- O problema e que o modelo prediz masculino poucas vezes.

## 10. Resultados Por Ano

Comparacao por ano:

| Ano | Acerto | Divergencia | Total |
|---|---:|---:|---:|
| 2017 | 61 | 33 | 94 |
| 2019 | 155 | 79 | 234 |
| 2021 | 65 | 30 | 95 |
| 2022 | 115 | 49 | 164 |
| 2023 | 92 | 52 | 144 |
| Total | 488 | 243 | 731 |

Percentual por ano:

| Ano | Acerto | Divergencia |
|---|---:|---:|
| 2017 | 64,89% | 35,11% |
| 2019 | 66,24% | 33,76% |
| 2021 | 68,42% | 31,58% |
| 2022 | 70,12% | 29,88% |
| 2023 | 63,89% | 36,11% |

Interpretacao:

- O desempenho e relativamente parecido entre os anos.
- O melhor ano foi 2022, com 70,12% de acerto.
- O pior ano foi 2023, com 63,89% de acerto.
- A diferenca entre anos existe, mas nao parece ser o principal problema.
- O principal problema continua sendo o vies para feminino.

## 11. Probabilidades Por Resultado

Probabilidade masculina media:

| Resultado | Count | Media | Mediana | Min | Max |
|---|---:|---:|---:|---:|---:|
| ACERTO | 488 | 0,2544 | 0,2332 | 0,0232 | 0,7346 |
| DIVERGENCIA | 243 | 0,3424 | 0,3581 | 0,0717 | 0,6679 |

Probabilidade feminina media:

| Resultado | Count | Media | Mediana | Min | Max |
|---|---:|---:|---:|---:|---:|
| ACERTO | 488 | 0,7456 | 0,7668 | 0,2654 | 0,9768 |
| DIVERGENCIA | 243 | 0,6576 | 0,6419 | 0,3321 | 0,9283 |

Interpretacao:

- Mesmo nas divergencias, a probabilidade feminina media e alta: 0,6576.
- Isso mostra que muitos masculinos estao recebendo probabilidade feminina alta.
- A probabilidade masculina media nas divergencias e 0,3424, maior que nos acertos.
- Isso sugere que um threshold masculino menor que 0,5 pode recuperar parte dos masculinos.
- Testar thresholds como 0,35 ou 0,40 pode ser importante.

## 12. Votos Por Individuo

Como ha tres vistas, o modelo final registra votos masculinos e femininos.

Distribuicao de votos:

| Votos masculinos | Votos femininos | Quantidade |
|---:|---:|---:|
| 0 | 3 | 465 |
| 1 | 2 | 190 |
| 2 | 1 | 68 |
| 3 | 0 | 8 |

Interpretacao:

- Em 465 individuos, as tres vistas votaram feminino.
- Em apenas 8 individuos, as tres vistas votaram masculino.
- Em 68 individuos, duas vistas votaram masculino.
- Em 190 individuos, apenas uma vista votou masculino.
- O sistema de voto tambem esta fortemente inclinado ao feminino.

## 13. Analise Por Vista

Metricas por vista no conjunto de teste:

| Vista | Accuracy | F1 masculino |
|---|---:|---:|
| CD | 64,48% | 38,10% |
| CE | 65,03% | 37,25% |
| CF | 67,76% | 25,32% |

Recall por vista e genero:

| Vista | Genero real | Recall/acerto |
|---|---|---:|
| CD | FEMININO | 83,55% |
| CD | MASCULINO | 32,70% |
| CE | FEMININO | 89,32% |
| CE | MASCULINO | 27,38% |
| CF | FEMININO | 96,37% |
| CF | MASCULINO | 18,25% |

Tabela de predicao por vista:

| Vista | Genero real | Predito feminino | Predito masculino |
|---|---|---:|---:|
| CD | FEMININO | 391 | 77 |
| CD | MASCULINO | 177 | 86 |
| CE | FEMININO | 418 | 50 |
| CE | MASCULINO | 191 | 72 |
| CF | FEMININO | 451 | 17 |
| CF | MASCULINO | 215 | 48 |

Interpretacao por vista:

- A vista CF tem maior accuracy geral, mas pior recall masculino.
- A vista CF acerta 96,37% dos femininos, mas apenas 18,25% dos masculinos.
- As vistas CE e CD reconhecem masculinos um pouco melhor que CF.
- CD tem o melhor F1 masculino entre as vistas: 38,10%.
- CE tambem tem F1 masculino superior ao CF: 37,25%.
- O uso da CF no voto pode estar puxando a decisao final para feminino.
- Testar voto sem CF, ou com peso menor para CF, pode melhorar o equilibrio.

## 14. Probabilidades Por Vista

Probabilidade masculina media por vista e predicao:

| Vista | Predicao vista | Count | Media | Mediana | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| CD | FEMININO | 568 | 0,2638 | 0,2533 | 0,0051 | 0,4991 |
| CD | MASCULINO | 163 | 0,6426 | 0,6323 | 0,5001 | 0,9753 |
| CE | FEMININO | 609 | 0,2305 | 0,2214 | 0,0082 | 0,4978 |
| CE | MASCULINO | 122 | 0,6336 | 0,6134 | 0,5000 | 0,9357 |
| CF | FEMININO | 666 | 0,1612 | 0,1261 | 0,0058 | 0,4978 |
| CF | MASCULINO | 65 | 0,6534 | 0,6578 | 0,5100 | 0,8957 |

Interpretacao:

- CF tem probabilidade masculina muito baixa quando prediz feminino.
- A media masculina em CF/FEMININO e apenas 0,1612.
- Isso mostra que CF esta muito confiante em feminino para muitos casos.
- CD e CE sao menos extremos que CF.
- O modelo pode estar aprendendo padroes de silhueta frontal que favorecem demais feminino.

## 15. Comparativo Com Runs Anteriores

Historico de runs:

| Run | Modelo | Individuos | Silhuetas | Accuracy teste | F1 masc. | Comparacao geral acerto | Divergencia | Indefinido |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-06-28 23:40 | sklearn_logistic_silhueta_64_media_vistas | 733 | 2.199 | 64,67% | 57,52% | 90,86% | 8,19% | 0,95% |
| 2026-06-29 00:21 | vgg16_imagenet_silhueta_128_head_media_vistas | 731 | 2.193 | 74,32% | 69,68% | 77,84% | 16,14% | 6,02% |
| 2026-06-30 03:57 | article_vgg16_imagenet_silhueta_256_aug_flip_rot5_voto_3vistas | 731 | 2.193 | 66,12% | 27,91% | 66,76% | 33,24% | 0,00% |

Observacoes importantes:

- O melhor resultado de teste registrado foi o `vgg16` embeddings:
  - accuracy: 74,32%;
  - F1 masculino: 69,68%.
- O `article_vgg16` foi pior que o `vgg16 embeddings`.
- O `article_vgg16` teve F1 masculino muito baixo.
- O `sklearn` aparece com comparacao geral muito alta, 90,86%, mas a acuracia real de teste registrada foi 64,67%.
- A metrica `comparacao geral acerto` nao deve ser confundida com acuracia limpa de teste.
- Para comparar modelos, usar preferencialmente:
  - accuracy de teste;
  - F1 masculino;
  - precision masculino;
  - recall masculino;
  - matriz de confusao.

## 16. Por Que O Modelo Artigo Foi Pior

Hipoteses baseadas nos resultados:

1. Dataset desbalanceado

- O dataset tem 64,02% feminino.
- O modelo pode obter acuracia razoavel prevendo feminino muitas vezes.
- Isso reduz recall masculino.

2. Ausencia de `class_weight` no `article_vgg16`

- O `sklearn` usa `class_weight="balanced"`.
- O `article_vgg16` atual nao usa peso de classe.
- Isso favorece a classe majoritaria, que e feminino.

3. Learning rate muito baixo

- O `learning_rate` usado foi 0.000002.
- Esse valor e muito conservador.
- Pode ter feito a cabeca neural aprender pouco.

4. VGG16 congelada

- A base VGG16 fica congelada.
- Apenas a cabeca neural aprende.
- Como as imagens sao silhuetas cranianas, os filtros ImageNet podem nao ser ideais.

5. Threshold fixo em 0,5

- O sistema considera masculino apenas quando a probabilidade masculina passa de 0,5.
- Como as probabilidades masculinas ficam frequentemente abaixo disso, muitos masculinos viram feminino.
- Thresholds menores podem melhorar recall masculino.

6. Voto das tres vistas

- CF puxa fortemente para feminino.
- Como CF entra no voto, ela pode reduzir a chance de classificar masculino.
- CE e CD parecem melhores para masculino.

## 17. Recomendacoes De Testes Para Aumentar Acuracia

### 17.1. Voltar ao melhor baseline salvo

O melhor teste salvo ate agora foi:

```text
backend: vgg16
image_size: 128
batch_size: 8
epochs: 8
weights: imagenet
```

Comando sugerido:

```bash
.venv/bin/python Treinamento_cranios/scripts/07_treinar_sexo_cranio.py \
  --root Treinamento_cranios \
  --backend vgg16 \
  --image-size 128 \
  --batch-size 8 \
  --epochs 8 \
  --weights imagenet
```

Esse modelo registrou:

- accuracy teste: 74,32%;
- F1 masculino: 69,68%.

### 17.2. Testar VGG16 embeddings com mais epocas

Comando:

```bash
.venv/bin/python Treinamento_cranios/scripts/07_treinar_sexo_cranio.py \
  --root Treinamento_cranios \
  --backend vgg16 \
  --image-size 128 \
  --batch-size 8 \
  --epochs 16 \
  --weights imagenet
```

Objetivo:

- Ver se mais epocas melhoram a cabeca neural.
- Comparar com o resultado de 8 epocas.

### 17.3. Testar VGG16 embeddings com image_size 160

Comando:

```bash
.venv/bin/python Treinamento_cranios/scripts/07_treinar_sexo_cranio.py \
  --root Treinamento_cranios \
  --backend vgg16 \
  --image-size 160 \
  --batch-size 8 \
  --epochs 12 \
  --weights imagenet
```

Objetivo:

- Aumentar um pouco a resolucao sem chegar ao custo de 256.
- Ver se mais detalhe ajuda sem prejudicar treinamento.

### 17.4. Testar article_vgg16 com learning rate maior

Comando:

```bash
.venv/bin/python Treinamento_cranios/scripts/07_treinar_sexo_cranio.py \
  --root Treinamento_cranios \
  --backend article_vgg16 \
  --image-size 256 \
  --batch-size 8 \
  --epochs 18 \
  --weights imagenet \
  --learning-rate 0.00001
```

Objetivo:

- Ver se o modo artigo estava aprendendo pouco por learning rate baixo.

### 17.5. Testar article_vgg16 mais leve

Comando:

```bash
.venv/bin/python Treinamento_cranios/scripts/07_treinar_sexo_cranio.py \
  --root Treinamento_cranios \
  --backend article_vgg16 \
  --image-size 128 \
  --batch-size 8 \
  --epochs 18 \
  --weights imagenet \
  --learning-rate 0.00001
```

Objetivo:

- Reduzir custo computacional.
- Ver se a resolucao 256 nao esta piorando por overfitting ou dificuldade de treino.

## 18. Melhorias De Codigo Ou Pipeline Recomendadas

Estas melhorias provavelmente sao mais importantes do que apenas mudar epocas ou batch.

### 18.1. Adicionar `class_weight` no modo artigo

Justificativa:

- O dataset e desbalanceado.
- O modelo atual favorece feminino.
- `class_weight` pode penalizar mais erros em masculino.

Resultado esperado:

- Aumentar recall masculino.
- Possivelmente reduzir um pouco acuracia feminina.
- Melhorar F1 masculino e equilibrio geral.

### 18.2. Criar curva de threshold

Atualmente o threshold e 0,5.

Testar:

- 0,30;
- 0,35;
- 0,40;
- 0,45;
- 0,50.

Metricas a registrar para cada threshold:

- accuracy;
- balanced accuracy;
- F1 masculino;
- precision masculino;
- recall masculino;
- matriz de confusao.

Justificativa:

- O modelo tem muitas probabilidades masculinas entre 0,30 e 0,50.
- Um threshold menor pode recuperar masculinos.

### 18.3. Usar balanced accuracy como metrica principal

Accuracy simples pode enganar quando o dataset e desbalanceado.

Metricas recomendadas:

- balanced accuracy;
- F1 macro;
- F1 masculino;
- recall masculino;
- precision masculino.

### 18.4. Testar combinacoes de vistas

Testes sugeridos:

- CF + CE + CD, voto atual.
- CE + CD apenas.
- CF com peso menor.
- Media de probabilidades em vez de voto seco.
- Voto ponderado por performance da vista.

Motivo:

- CF tem maior acuracia geral, mas pior recall masculino.
- CE e CD parecem melhores para reconhecer masculino.

### 18.5. Registrar metricas por vista em todos os modelos

O modo artigo ja salva metricas por vista.

Recomendado salvar tambem para:

- sklearn;
- vgg16 embeddings.

Motivo:

- Permite entender se um modelo depende demais de uma vista.

### 18.6. Separar treino e comparacao geral

Atualmente o historico mistura:

- accuracy de teste;
- comparacao geral contra todos os individuos.

Recomendacao:

- Destacar claramente:
  - `accuracy_teste`;
  - `f1_teste`;
  - `comparacao_total_acerto`;
  - `comparacao_total_divergencia`.

Motivo:

- Evitar interpretar 90,86% do sklearn como acuracia real de teste.

## 19. Diagnostico Principal

O ultimo modelo `article_vgg16` nao falhou de maneira aleatoria.

Ele falhou por um padrao claro:

```text
O modelo classificou cranios como FEMININO em excesso.
```

Evidencias:

- 655 de 731 individuos foram previstos como feminino.
- Apenas 76 de 731 foram previstos como masculino.
- O dataset real tinha 263 masculinos.
- Recall masculino no teste foi 18,18%.
- Acerto em femininos foi 94,02%.
- Acerto em masculinos foi 18,25%.

Portanto, o principal objetivo dos proximos ajustes deve ser:

```text
Aumentar o reconhecimento de MASCULINO sem destruir completamente a boa performance em FEMININO.
```

## 20. Ordem Recomendada De Proximos Experimentos

Ordem sugerida:

1. Repetir o melhor baseline:
   - `backend=vgg16`
   - `image_size=128`
   - `batch_size=8`
   - `epochs=8`
   - `weights=imagenet`

2. Testar `vgg16` com:
   - `image_size=160`
   - `epochs=12`
   - `batch_size=8`

3. Implementar/avaliar thresholds diferentes:
   - 0,35;
   - 0,40;
   - 0,45;
   - 0,50.

4. Implementar `class_weight` no `article_vgg16`.

5. Testar `article_vgg16` com:
   - `learning_rate=0.00001`;
   - `image_size=128`;
   - `image_size=256`;
   - `batch_size=8`.

6. Testar combinacoes de vistas:
   - `CE + CD`;
   - `CF + CE + CD`;
   - media ponderada;
   - voto ponderado.

7. Escolher o melhor modelo nao so por accuracy, mas por:
   - balanced accuracy;
   - F1 masculino;
   - recall masculino;
   - matriz de confusao.

## 21. Conclusao

O ultimo teste com `article_vgg16` teve desempenho geral moderado, mas nao e adequado como melhor modelo atual porque apresenta forte vies para feminino.

Resumo final:

- Accuracy do ultimo teste: 66,12%.
- F1 masculino: 27,91%.
- Recall masculino: 18,18%.
- Acerto geral comparando face x cranio: 66,76%.
- Acerto em feminino: 94,02%.
- Acerto em masculino: 18,25%.

O melhor resultado historico salvo ate agora parece ser o `vgg16 embeddings`, com:

- accuracy teste: 74,32%;
- F1 masculino: 69,68%.

O caminho mais promissor e:

1. Usar o `vgg16 embeddings` como baseline principal.
2. Corrigir o desbalanceamento no modo artigo com `class_weight`.
3. Testar thresholds masculinos menores que 0,5.
4. Reavaliar o papel da vista CF no voto final.
5. Comparar os proximos modelos usando F1 masculino, recall masculino e balanced accuracy, nao apenas accuracy.

