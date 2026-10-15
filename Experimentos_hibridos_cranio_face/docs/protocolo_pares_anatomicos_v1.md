# Protocolo de pares cranio-face: v1.0

Data: 2026-09-26. Estado: protocolo operacional proposto, ainda nao ratificado
anatomicamente. Escopo: fotografias faciais frontais e renderizacoes cranianas CF.
Nao e validacao pericial. Confiabilidade so podera ser estimada apos as repeticoes
e a referencia independente descritas abaixo.

## 1. Objetivo e situacao real

Investigar se correspondencias regionais cranio-face auxiliam um ranking
experimental. Os 11 numeros representam relacoes, nao pontos osso-pele identicos.
Os arquivos existentes sao propostas visuais individuais de IA, nao saidas de
um detector treinado nem referencias especialistas. O lote adicional tem 10
pares de desenvolvimento, de 2017 e 2019: nao representa todos os anos.
Nao reclassificar retrospectivamente estes arquivos como anotacoes v1.0
verificadas: conservam seu protocolo original e precisam de nova revisao.

## 2. Contrato da imagem e da anotacao

- Parear somente pelo manifesto e IDs; nunca selecionar o par pela semelhanca.
- Preservar original e hash SHA-256. Registrar dimensoes, modalidade, vista,
  ano, identidade, split, protocolo, autor, ferramenta, data e revisao.
- X cresce para a direita, Y para baixo; origem no canto superior esquerdo.
  Salvar coordenadas na resolucao original, nao na miniatura ou screenshot.
- Registrar crop, escala e transformacao inversa. Nao esticar os eixos
  separadamente. Desfazer a transformacao antes de gravar.
- Lados anatomicos pertencem ao individuo. Se espelhamento nao estiver
  documentado, registrar orientacao desconhecida e nao validar lateralidade.
- CE/CD podem auxiliar a interpretacao; nunca transferir pixels para CF.
- Preservar assimetria. Nao mover um ponto para produzir simetria ou melhorar
  uma sobreposicao. Nao editar anatomia nem reconstruir partes ocultas por IA.
- Para marcar, mostrar UMA modalidade por vez, sem o par correspondente e sem
  ranking. Exibir os pares lado a lado somente depois, para auditoria.
  As propostas atuais foram vistas em pares e nao constituem anotacao cega.

## 3. Dicionario e limites dos 11 numeros

Todas as localizacoes abaixo sao candidatas. Visibilidade frontal insuficiente
obriga abstencao, mesmo quando uma posicao aproximada parece plausivel.

| Numero | Cranio | Face | Regra e limite |
|---|---|---|---|
| 1 | Glabella, g | Glabella cutanea, g' | Regiao mediana entre relevos supraorbitais. A projecao mais anterior nao se confirma em CF isolada. Nao substituir por centro geometrico entre olhos. |
| 2 | Nasospinale, ns | Subnasale, sn | No cranio, mediana ao nivel das margens inferiores da abertura piriforme, nao ponta da espinha nasal. Na face, encontro mediano da columela com o labio superior. Relacao de nivel, nao coincidencia exata. |
| 3 e 4 | Extremos laterais da abertura piriforme | Alare cutaneo direito/esquerdo | Procurar a maior extensao lateral de cada margem, excluindo sombras e fragmentos. Na face, limite externo da asa nasal, nao centro da narina. Confirmar a terminologia ossea com especialista; larguras nao devem ser igualadas. |
| 5 e 6 | Ectoconchion candidato | Exocanthion direito/esquerdo | Na face, comissura lateral das palpebras. No cranio, nao aceitar simplesmente o ponto mais externo da mascara. A definicao do plano transversal deve ser ratificada; ate la, sao propostas orbitais, nao ec confirmado. |
| 7 | Menton projetado | Menton cutaneo projetado | Limite inferior mediano aparente do corpo mandibular/contorno cutaneo do queixo. Excluir suporte, pescoco, sombra e barba. Nao e pogonion nem gnathion. Pose pode impedir localizacao. |
| 8 e 9 | Zygion candidato | Zygion cutaneo candidato | Extremo lateral do arco zigomatico; nao centro malar, orelha, ramo mandibular ou maior largura arbitraria da imagem. Na face, a regiao que recobre o arco pode nao ser localizavel. Nao inferir automaticamente pela altura do olho. |
| 10 e 11 | Margem orbital medial: proxy 2D | Endocanthion direito/esquerdo | Na face, comissura medial das palpebras, nao centro da caruncula. A margem craniana proposta NAO confirma a crista lacrimal anterior. Excluir do conjunto anatomico principal ate ratificacao; testar separadamente como proxy. |

### Congelar a definicao orbital antes de treinar

Para estudos geometricos exploratorios, pode-se propor uma intersecao do
contorno orbital com uma linha transversal documentada. Isso deve receber
nome de proxy e parametros explicitos, nao ser renomeado como ec ou mo.
Nao escolher a altura dessa linha depois de observar qual fornece melhor matching.
Se o especialista nao conseguir fornecer uma regra reproduzivel nas nossas
imagens, retirar essa feature do estudo principal e reportar a limitacao.

## 4. Visibilidade separada de aprovacao

No registro futuro, manter dois campos independentes:

- visibilidade: VISIVEL, OCULTO, FORA_DO_QUADRO, INDETERMINADO, NAO_AVALIADO;
- revisao: PROPOSTA_IA, ANOTACAO_HUMANA, REVISADO, ADJUDICADO, REJEITADO.

Nos CSVs legados, PROPOSTA_IA/PROPOSTA e DUVIDA permanecem como estao.
DUVIDA nao distingue oclusao de artefato: usar observacao e recodificar somente
apos revisar. LOCALIZADO nao significa aprovado. Nenhum destes nomes, sozinho,
autoriza treino. Nao converter ausencia do detector em ausencia anatomica.

Para ponto oculto/indeterminado: X e Y ausentes, motivo obrigatorio.
Para ponto proposto: coordenadas finitas dentro da imagem e evidencia registrada.
Confianca subjetiva nao e probabilidade calibrada. Nunca atribuir 95% por parecer
plausivel. Alterar imagem, ponto, protocolo ou coordenada invalida a aprovacao
anterior para a nova versao.

## 5. Barba, oclusoes e exclusoes

Manter a identidade, por padrao. Barba nao implica exclusao total nem prova
que todos os pontos faciais estejam ocultos. Avaliar cada ponto separadamente.
Se o queixo cutaneo nao for discernivel, omitir o ponto 7. Nao seguir a ponta
da barba, preencher com zero ou gerar um queixo sem barba artificial.

O futuro detector devera ter mascara de visibilidade e perda de coordenadas
apenas nos pontos com referencia valida. A avaliacao deve medir tambem falsas
localizacoes de pontos ocultos. Nao basta medir erro nos pontos que retornou.
Pseudorrotulos com mascara nao constituem validacao independente.

No matching comparar, nos mesmos splits: sem geometria; conjunto fixo sem
queixo; conjunto completo nos casos elegiveis; geometria com mascara.
Normalizar por evidencias comparaveis, reportar quantas regioes foram usadas e
nao favorecer candidatos com poucos pontos. O minimo de regioes necessario
sera escolhido na validacao e congelado antes do teste; ainda nao esta definido.
Se insuficiente, emitir INFORMACAO_INSUFICIENTE, nao um score anatomico seguro.
Reportar separadamente casos com barba/oclusao e perdas de cobertura.

## 6. Repeticao e referencia independente

1. Ratificar definicoes com profissional de anatomia/antropologia, incluindo
   exemplos positivos, negativos e invisiveis para cada numero.
2. Preparar piloto de repetibilidade em desenvolvimento, procurando diversidade
   de anos, pose, renderizacao, resolucao e oclusoes. Proposta: 30 identidades;
   tamanho definitivo depende da precisao desejada, nao e calculo amostral pronto.
3. Dois anotadores humanos independentes, sem propostas IA e sem ver o par,
   marcam a amostra. Pelo menos um deve ter qualificacao anatomica adequada.
4. Repetir apos intervalo predefinido, por exemplo 14 dias, com ordem aleatoria
   e sem acesso a primeira sessao. Registrar o intervalo real.
5. Preservar ambas as versoes antes da discussao. Adjudicar divergencias com
   justificativa; nao trocar silenciosamente as referencias.
6. Comparar IA versus referencia e humano versus humano separadamente.
   Repetir a IA em outra sessao mede estabilidade da IA, nao verdade anatomica.
7. Formar referencia independente no teste congelado. Identidades, vistas,
   duplicatas e aumentos da mesma pessoa nunca atravessam splits.

## 7. Metricas e criterios de avancar

Medir por ponto, modalidade, ano e visibilidade:
- erro euclidiano em pixels originais, mediana, media, percentil 95;
- erro normalizado pela diagonal de uma ROI definida independentemente dos
  pontos previstos e congelada antes da avaliacao; reportar tambem a ROI;
- curva PCK com tolerancias pre-especificadas, por exemplo 1%, 2% e 5% da
  diagonal da ROI como exploracao, NAO limiares de validacao clinica;
- diferencas assinadas X/Y para detectar vies sistematico;
- repetibilidade intra/interanotador, concordancia de visibilidade e abstenções;
- intervalos de confianca com reamostragem por identidade, nao por ponto
  tratado como observacao independente.

Sem calibracao fisica, nao converter pixels em milimetros. Nao usar distancias
ou marcos previstos como denominador do proprio erro. Alta cobertura nao e
acuracia. Nenhum limite numerico de aceitacao anatomica esta validado aqui.
Definir tolerancias justificadas e precisao dos intervalos com orientador e
especialista ANTES de consultar resultados de teste.

## 8. Treinar com rotulos de IA: dois caminhos distintos

A: anotacao assistida por IA + correcao anatomica humana. Referencia de treino
tem origem e revisao documentadas; avaliar em referencia independente.

B: pseudorrotulos de IA sem correcao. Estudo de supervisao fraca/distilacao,
explicitamente exploratorio. Pode aprender vieses e erros da IA. Comparar
com treino em subconjunto humano, IA e IA corrigida, usando MESMO teste
independente e orcamento/arquitetura documentados.

Somente dados do split treino entram na perda, normalizadores, PCA, selecao
de features e ajuste do detector. Validacao seleciona hiperparametros; teste
avalia uma vez o metodo congelado. Concordancia com rotulos da propria IA
mede imitacao, nao precisao anatomica. Este documento nao libera exportacao
ou inicia treinamento: o gate existente permanece bloqueado.

## 9. Rastreabilidade e relato academico

Guardar ferramenta, modelo/versao quando conhecidos, data, prompts, imagens
mostradas, resolucoes de exibicao, coordenadas, transformacoes, scripts e
correcoes humanas. Se a versao exata do modelo nao foi registrada, declarar
NAO_REGISTRADA; nao inventar nem atribuir um modelo retrospectivamente.
O historico de conversa deve ser preservado de forma privada: ainda nao foi
exportado como um pacote reprodutivel por este protocolo.

As coordenadas atuais foram propostas visualmente no Codex e transcritas em
scripts. Executar o script reproduz o ARQUIVO, nao garante que outra sessao de
IA escolheria os mesmos pontos. Nenhuma validacao humana especializada foi
realizada neste trabalho ate este registro.

Texto fiel ao estado atual para metodos:

> Foi realizado um piloto de pre-anotacao de pares cranio-face assistida por
> IA multimodal. As propostas de coordenadas foram registradas na resolucao
> original, com proveniencia e motivos de abstencao. Foram preservadas como
> pseudorrotulos nao validados, separados de qualquer referencia anatomica
> independente. Nesta etapa nao foi treinado nem validado um detector a partir
> dessas propostas.

Nao escrever que houve validacao especializada, anotacao cega ou treinamento
se ainda nao ocorreram. Conferir regras da instituicao/revista e autorizacao
etica/de uso das imagens e de processamento por servicos externos. Nao publicar
faces identificaveis por padrao. A autoria e responsabilidade sao humanas.

## 10. Literatura aplicada e limites da transferencia

- Srisinghasongkram et al., 2026, Int. J. Morphol. 44(1):121-129, Tabela I e
  Figura 1: fonte inicial de nomes e relacoes. Nao fornece coordenadas do nosso
  acervo nem valida os proxies. https://www.scielo.cl/pdf/ijmorphol/v44n1/0717-9502-ijmorphol-44-01-121.pdf
- Estudo de validacao de superimposicao: orienta distinguir correspondencia,
  tecido mole e incerteza. Nao valida nosso esquema de 11 pontos.
  https://link.springer.com/article/10.1007/s00414-022-02929-4
- Landmarking craniano 3D: exemplo de comparacao automatica versus manual;
  erros em mm/3D nao sao transferiveis para nossas fotografias 2D.
  https://www.nature.com/articles/s41598-024-63137-1
- Cranio-ID: motiva detector de pontos e representacao regional; radiografias
  e protocolo diferentes, nao evidenciam eficacia neste acervo.
  https://arxiv.org/html/2511.14411
- ICMJE: transparencia sobre ferramenta e finalidade de uso de IA; consultar
  tambem politica especifica da revista. Nao e certificacao do metodo.
  https://www.icmje.org/recommendations/browse/artificial-intelligence/ai-use-by-authors.html

## Checklist antes de ampliar ou treinar

- [ ] Definicoes orbitais e zigomaticas ratificadas; exemplos e nao exemplos.
- [ ] Lateralidade e qualidade documentadas, sem inferir estruturas ocultas.
- [ ] Auditoria dos pares/duplicatas e splits concluida.
- [ ] Referencia independente e repeticoes coletadas.
- [ ] Tolerancias e plano estatistico congelados.
- [ ] Dados e artefatos com proveniencia completa; lacunas declaradas.
- [ ] Regras eticas/institucionais e revista conferidas.
- [ ] Decisao formal sobre estudo de pseudorrotulos ou rotulos corrigidos.

Nenhum item e automaticamente concluido pela existencia deste documento.
