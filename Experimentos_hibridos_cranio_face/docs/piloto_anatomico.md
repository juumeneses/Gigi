# Piloto de referencias anatomicas

## Escopo

Este lote propoe quatro pontos em dez fotografias CF de desenvolvimento (duas por ano).
Nao usa o teste nem a amostra de aceitacao. Nao altera originais, nao treina,
nao libera matching e nao converte propostas em verdade anatomica.

Referencia: Srisinghasongkram et al. (2026), Tabela I, p. 122, e Figura 1, p. 123.
https://www.scielo.cl/pdf/ijmorphol/v44n1/0717-9502-ijmorphol-44-01-121.pdf

## Contrato do piloto

| Ponto | Meta da revisao | Proposta computacional e limite |
|---|---|---|
| ec direito/esquerdo | Borda lateral da orbita no plano transversal medio | Intersecao do contorno escuro com a linha media da cavidade; sombra pode nao coincidir com rebordo osseo |
| al direito/esquerdo | Extremo lateral da abertura nasal ossea | Extremo do contorno escuro nasal; exige confirmar que nao incluiu dentes ou fundo |

Direita/esquerda sao do individuo: na CF nao espelhada, direita anatomica aparece a esquerda da imagem.
O protocolo pressupoe orientacao nao espelhada; uma fotografia isolada nao a garante.
Centros orbitais do detector antigo NAO sao ec. Centros nasais NAO sao al.
Nao projetamos coordenadas da figura sobre outras fotografias.

O detector exige apoio em pelo menos tres limiares e dispersao <= 2,5% da diagonal
do bbox. Estes sao parametros exploratorios, nao limites clinicos ou do artigo.
Confianca probabilistica nao e fornecida. Ausencia de proposta nao significa estrutura ausente.

Mento, gonios, zigios, nasion, glabella e pontos dentarios ficam FORA deste primeiro
lote: nao ha localizador confiavel ainda. Nao inventar pontos para completar um esquema.

## Revisao

No painel, selecione a execucao terminada em `piloto_anatomico`.
Compare original, contorno amarelo, ponto vermelho e ampliacao da regiao.
Use Correto apenas quando a coordenada estiver na estrutura definida, nao apenas perto.
Use Incorreto se a proposta estiver em sombra, suporte, dente ou estrutura errada.
Use Nao visivel se a estrutura nao puder ser vista; Nao sei quando houver duvida.
As respostas sao triagem visual vinculada a imagem/coordenadas, nao certificacao especializada.

## Antes de treinar

Revisar definicoes e exemplos com especialista. Coordenadas verificadas precisam
ter identidade, hash da imagem, versao, revisor e visibilidade. Rejeicoes nao sao
rotulos de invisibilidade. Manter conjunto de avaliacao independente, com referencia
verificada. So entao preparar exportacao YOLO-Pose e treino piloto.
Nao usar a mesma imagem na aprendizagem e na avaliacao. Nao treinar sobre o proprio
resultado sem verificacao. Nenhum exportador automatico de labels foi habilitado.

## Executar novo lote

Da raiz Gigi:

```sh
Treinamento_faces/.venv/bin/python Experimentos_hibridos_cranio_face/scripts/piloto_anatomico.py --por-ano 2
```

Cada comando cria uma execucao separada, com CSV progressivo e JSON por imagem.
Nao implementa retomada: por ser lote pequeno, uma repeticao cria outro historico.
