# Entrega e verificacao local

Implementado em 13/10/2026. Pastas de execucao usam horario UTC.

## Estado

- Painel independente em http://127.0.0.1:8503 .
- DINOv2 oficial ViT-S/14 preparado, codigo e pesos registrados por hash.
- Nenhum landmark ou estimativa de sexo e necessario para treinar ou consultar.
- Experimentos E0-E6, projecoes faciais/cranianas, InfoNCE e triplet semi-hard.
- Comparacao de coortes iguais, galerias fixas, tres seeds e bootstrap pareado.
- Jobs de treino, extracao, avaliacao, parada/retomada e historico.
- Ranking visual, consulta conhecida/upload, zoom, oclusao e contribuicao por vista.
- Sobreposicao manual frontal, sorrindo ou lateral; controles nao alteram o ranking.

## Inventario real

5394 arquivos reconhecidos e legiveis, incluindo copias; 798 IDs catalogados.
26 IDs compartilham arquivos com outros IDs e ficaram em quarentena, sem alterar o Banco.
As copias identicas em `_separados` sao aliases da identidade, nao novas pessoas.

| Coorte | Treino | Validacao | Teste reservado |
|---|---:|---:|---:|
| Frontal E1 | 495 | 104 | 106 |
| Todas as vistas E6 | 492 | 103 | 106 |

O inventario registra separadamente 65 exclusoes por split pendente, 26 por hashes entre IDs
e 2 por vista frontal ausente/ambigua. Nenhuma identidade de teste foi usada nos treinos tecnicos.

## Teste real pequeno

Suite `artifacts/suites/20261014T023052178334Z`: 8 pessoas de treino, 4 de validacao,
3 epocas. E0, E1 e E6 completaram extracao/treino/avaliacao; zero falhas nos pares de validacao.
Ranking de 4 candidatos e mapas de oclusao das seis vistas foram produzidos.
Os hashes de todas as imagens usadas permaneceram iguais antes/depois.

E0 e E1 tiveram MRR 0,5417; E6 teve 0,5625. **A amostra e apenas tecnica: estes valores
nao demonstram melhoria nem desempenho util de identificacao.** O treino completo ainda nao foi iniciado.

Uma segunda suite pequena passou pelo worker CLI usado pelo painel:
`artifacts/suites/20261014T024544179771Z`.
O botao Comparar foi acionado no navegador e concluiu em aproximadamente 6 segundos.

## Testes automatizados

15 testes passaram: formatos/EXIF/LFS, proporcao, aliases e escolhas explicitas, quarentena,
mascaras de vistas, cache corrompido, hashes alterados, retomada deterministica, parada,
denominadores com falhas, galeria estavel, equivalencia de inferencia com IDs renomeados,
comparacao pareada, suite com tres seeds, gate do teste congelado, gradientes de triplet,
sobreposicao e navegacao das quatro abas.

## Proximo uso

Em Treinamentos, manter E1 e clicar Novo treinamento + baseline frontal. A extracao necessaria
acontece automaticamente, com reutilizacao do cache. Apos a conclusao no banco, E2-E6 ficam
disponiveis para comparacoes. O baseline frontal sera repetido na coorte de cada experimento.

Este laboratorio treina projecoes sobre DINOv2 congelado. Nao treina o proprio extrator nem
reconstroi anatomia. Sensibilidade e sobreposicao sao ferramentas de inspecao, sem equivalencia
anatomica automatica. A eficacia precisa ser determinada pela validacao e pelo teste reservado.
