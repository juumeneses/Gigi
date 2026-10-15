# Piloto executado em 13/10/2026

Foram importadas 22 imagens, correspondentes aos 11 pares do piloto anterior.
Todas encontraram copia em Banco com hash identico. Nenhuma anotacao foi
transferida apenas por semelhanca de nome. Os splits originais foram preservados:
10 pessoas de treino e 1 de validacao. Nao houve uso de teste final.

## Treinamentos novos

CNN de coordenadas, 11 pontos, entrada proporcional de 128 pixels, Adam 0.001,
lote 8, semente 42, 20 epocas, CPU. Pontos ausentes ficaram fora da perda.

| Modalidade | Epoca selecionada | Pontos avaliados | Mediana / diagonal | P95 / diagonal | Baseline medio / diagonal |
|---|---:|---:|---:|---:|---:|
| Face frontal | 19 | 10 | 1.71% | 3.59% | 1.99% |
| Cranio frontal | 18 | 10 | 1.84% | 3.34% | 2.49% |

Essas distancias sao relativas aos pseudorrotulos de uma unica pessoa de validacao.
Nao demonstram precisao anatomica nem generalizacao. O baseline e a posicao media
dos pontos ajustada apenas nas imagens de treino. Nao foi escolhido um modelo por
resultados no teste final.

Execucoes:
- Face: `artifacts/training/20261013T203514131613Z`
- Cranio: `artifacts/training/20261013T203515091309Z`
- Referencias: `artifacts/references/20261013T203431678431Z`

## Processamento

- Aplicacao dos dois modelos nas 11 imagens de cada modalidade concluida.
- Normalizacao proporcional e regioes das referencias geradas para as 22 imagens.
- MediaPipe executado novamente nas 11 faces, com deteccao em todas.
- Mascaras e contornos candidatos gerados para os 11 cranios.
- Os arquivos legados e originais foram utilizados somente para leitura.

No macOS, o MediaPipe precisou inicializar OpenGL. O teste no ambiente isolado
falhou; a execucao local autorizada concluiu as 11 imagens. Falhas de inicializacao
sao registradas por imagem na versao atual, sem eliminar casos do denominador.

## Verificacao tecnica

Sete testes automatizados passaram: transformacao uniforme reversivel, perda com
ausencias/NaN, boxes nas bordas, retomada de treino identica a execucao continua,
bloqueio de vista/split incorretos e retomada de processamento com arquivos
ilegíveis, ponteiros LFS e saidas corrompidas, e preservacao da origem dos rotulos
sem promover estimativas a referencias oficiais. As quatro abas e as quatro
subabas de Faces e Cranios abriram no Streamlit AppTest sem excecoes.
O botao Avaliar execucao foi exercitado no navegador
e produziu o arquivo de avaliacao e o log persistido.

As laterais podem ser abertas e sobrepostas; nao ha referencias laterais para
treinar esses modelos. Os pontos previstos continuam com visibilidade nao avaliada.
Na inspecao visual do cranio de validacao, a proposta inferior de queixo aparece
proxima da estrutura abaixo da mandibula. A distancia global baixa nao aprova esse
ponto; esse exemplo permanece pendente de revisao. As caixas sao geometrias derivadas
dos grupos de pontos e nao delimitacoes anatomicas confirmadas.
