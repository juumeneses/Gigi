# Scores separados e combinacao

## Iniciar o painel

Na raiz do repositorio Gigi:

```sh
Treinamento_faces/.venv/bin/python -m streamlit run Treinamento_DINOv2_face_cranio/app.py --server.address 127.0.0.1 --server.port 8503
```

Se a porta ja estiver ocupada pelo painel, abra http://127.0.0.1:8503/.
Nao inicie outro servidor na mesma porta. Use 8504 se houver outro aplicativo.

## Treinar

Em Treinamentos, selecione "O que treinar" e "Nova sequencia".
Escolha um modelo individual ou "Treinar os tres modelos" e clique em
"Iniciar treinamento selecionado" para executar:

- frontal: FFN contra CF;
- sorriso: FFS contra CF;
- lateral: FL contra CE e CD (media mascarada das duas vistas cranianas).

Os tres modelos usam as mesmas identidades elegiveis para E6. Cada um aprende
suas proprias projecoes; o DINOv2 continua congelado. Parametros e seed sao os
do formulario. Para comparar seeds, crie uma sequencia nova com outra seed.
Os experimentos E0-E6 anteriores continuam no historico e nos resultados,
mas nao aparecem no seletor de novos treinamentos. A combinacao de scores
nao equivale a agregar vistas dentro de um unico modelo, como nesses experimentos.

Tambem e possivel treinar um componente por vez. Para os seguintes, selecione
a mesma sequencia e preserve os parametros. Componentes concluidos nao sao
recalculados. Retomar usa os checkpoints existentes e verifica compatibilidade.
Alteracoes no codigo ou configuracao podem exigir uma nova sequencia; nunca
substitua checkpoints anteriores para contornar essa verificacao.

## Combinar

Depois de terminar os tres componentes, selecione a sequencia no mesmo painel,
abra "Combinar modelos ja treinados",
defina pesos nao negativos e clique em "Avaliar combinacao dos scores".
O padrao 1/1/1 e uma referencia experimental, nao pesos otimos aprendidos.

Para cada componente, a media e o desvio sao calculados somente sobre scores
entre pessoas diferentes do treino. O score combinado e a soma ponderada dos
scores padronizados. Nao e cosseno, porcentagem nem probabilidade.
Um componente com peso positivo sem entrada utilizavel impede score combinado;
nao ha substituicao silenciosa de vistas. Todos os componentes sao registrados.

Resultados ficam em Resultados > Visao geral > Scores separados.
Sao exibidos Recall@1/3/5/10, MRR e diferencas descritivas contra frontal.
O relatorio HTML e rankings CSV ficam na pasta artifacts/fusions/<execucao>.
O arquivo results.json tambem preserva Gallery-10 e intervalos individuais.
O teste final nao e executado nem usado para pesos ou normalizacao.

O matching esta na propria area Matching > Modelos por vista.
Em Como comparar, escolha Selecionar modelos, Todos os modelos ou Todas as
7 combinacoes. A ultima opcao calcula os tres componentes uma unica vez e
monta os sete rankings sobre a mesma consulta e galeria, sem novo treinamento.
Ranking da combinacao escolhe o resultado exibido. O resumo mostra o primeiro
colocado e, para consultas conhecidas, a posicao do cranio correto.
Foto da pessoa permite alternar neutra/sorrindo/lateral apenas para inspecao;
Cranio no ranking seleciona o candidato. A tabela de contribuicoes mostra
cosseno, peso e parcela padronizada de cada modelo, cuja soma e o score final.
Nao compare o maior score entre combinacoes como probabilidade de acerto.
No Comparador visual, Mostrar quadrados de sensibilidade liga/desliga os mapas
nas mesmas imagens. Modelo do mapa identifica o componente analisado.
Sobrepor face e cranio usa as vistas selecionadas no comparador, com opacidade,
escala, rotacao e deslocamento manuais. Essas alteracoes nao mudam o ranking.
Quando nao existe mapa para o par/modelo/vista, o painel mostra a imagem
original e informa a ausencia; Analisar vistas e mapa de sensibilidade gera
essa analise. Os blocos sao testes de oclusao posteriores ao treino, nao
landmarks usados como supervisao nem correspondencias anatomicas confirmadas.
Marque Neutra, Sorrindo e/ou Lateral. Os modelos marcados recebem pesos iguais;
os desmarcados nao sao executados. Mudar os checkboxes preserva a amostra,
mas exige nova comparacao antes de exibir resultados da nova selecao.
Cada clique em Sortear amostra muda a face de referencia, exceto com seed fixa.
Sorteie uma consulta conhecida e compare. A tabela mostra cossenos individuais,
contribuicoes ponderadas e ranking combinado, com imagens CF/CE/CD.
Teste conhecido e Enviar imagens usam a mesma inferencia, com os modelos
selecionados. Imagens obrigatorias ausentes geram abstencao, nao substituicao
por outra vista. Os mapas de sensibilidade analisam um componente selecionado,
nao a soma dos scores. Experimentos anteriores continuam disponiveis no seletor.

## Todas as combinacoes

"Treinar os tres + avaliar as 7 combinacoes" executa os tres treinos
uma vez e avalia os sete subconjuntos nao vazios com pesos iguais:
neutra, sorrindo, lateral, neutra+sorrindo, neutra+lateral,
sorrindo+lateral e as tres. Nao sao sete novos treinamentos independentes.
Sao combinacoes tardias dos mesmos componentes e das mesmas identidades.
Se os modelos ja estiverem prontos, use "Avaliar as 7 combinacoes dos modelos
ja treinados". Matrizes sao reutilizadas; a retomada preserva avaliacoes
concluidas. A tabela de Resultados identifica modelos, pesos e sequencia.
Comparar sete alternativas na validacao nao constitui sete testes independentes;
nao escolha pesos ou modelos usando o teste final.

## Mapas visuais

Ao concluir um treinamento pelo CLI/painel, sao gerados ate dois exemplos de
validacao por modelo: primeiro acerto e primeiro erro disponiveis. Resultados
mostra esses mapas em Exemplos visuais, inclusive nos componentes separados.
Para execucoes antigas, use Gerar exemplos visuais desta execucao.
Ao comparar, sao gerados mapas do primeiro candidato para os modelos ativos.
Outros candidatos usam Analisar vistas e mapa de sensibilidade.

Vermelho significa que ocultar o bloco reduziu o score; azul, que aumentou.
A intensidade e relativa a cada imagem. Numeros identificam linha.coluna,
nao pontos anatomicos. O mapa de um componente nao e mapa do score combinado.
Os mapas exigem inferencias adicionais e sao reutilizados quando imagens,
checkpoint, configuracao e algoritmo permanecem iguais. Falhas nessa etapa
nao apagam pesos ou rankings concluidos.

O pre-processamento ja corrige EXIF, converte RGB, preserva proporcoes com
letterbox e normaliza canais com media [0.485,0.456,0.406] e desvio
[0.229,0.224,0.225]. Embeddings sao normalizados para cosseno. Nao ha
alinhamento anatomico automatico nem remocao automatica de fundo.

## Limites de interpretacao

Nao foram executados novos treinos completos durante a implementacao.
Testes sinteticos pequenos verificam o fluxo, nao eficacia anatomica.
Mexer repetidamente nos pesos olhando a validacao pode sobreajustar essa
validacao. Os ganhos sao exploratorios ate confirmar protocolo e seeds e
avaliar uma unica configuracao congelada no teste independente.
CE/CD sao combinadas pelo modelo lateral; nao sao alinhadas anatomicamente
com o lado do rosto. Este experimento mede se essa representacao ajuda.
