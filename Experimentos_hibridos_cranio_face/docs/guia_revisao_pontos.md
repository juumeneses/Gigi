# Guia visual para revisar pontos cranianos

## Referencia visual

![Figura 1 do artigo de 2026: cranio e face](figura_1_landmarks_2026.jpg)

Fonte: Srisinghasongkram et al., International Journal of Morphology 44(1),
121-129, 2026. Figura 1, pagina impressa 123 (pagina 3 do PDF).
Figura original extraida do PDF fornecido pelo usuario, sem alterar pontos.
Uso como referencia de leitura local; nao e anotacao de nenhuma imagem do acervo.

## Antes de avaliar

A figura mostra **onde procurar estruturas**, nao coordenadas que podem ser
copiadas para todos os cranios. Posicao, escala, perspectiva, conservacao do
osso e articulacao da mandibula mudam entre imagens. Direita/esquerda devem
se referir ao individuo: numa frontal nao espelhada, sua direita aparece
a esquerda de quem observa. Se o espelhamento for desconhecido, registrar
essa incerteza.

## Pontos da figura: leitura em portugues

| Sigla | Nome | Onde procurar | Cuidado na frontal |
|---|---|---|---|
| g | Glabella | Regiao mediana entre os relevos supraorbitais | Relevo/projecao nao se determina apenas pela altura em pixels |
| n | Nasion | Encontro mediano da sutura frontonasal | Se a sutura nao aparece, nao tratar a estimativa como confirmada |
| ns | Nasospinale | Mediana no nivel das margens inferiores da abertura nasal | Nao e o centro da cavidade nasal |
| al | Alare osseo | Extremidades laterais da abertura nasal representadas na figura | A Tabela I usa linguagem de asa nasal; fixar a definicao ossea no protocolo antes de automatizar |
| pr | Prosthion | Alveolo superior na linha media entre os incisivos centrais | Nao e a ponta/coroa do dente |
| id | Infradentale | Ponto mediano superior do septo alveolar entre incisivos inferiores | Dentes ausentes ou osso danificado podem impedir a avaliacao |
| zy | Zygion | Ponto lateral do arco zigomatico | Nao e qualquer extremo lateral da mascara |
| ec | Ectoconchion | Borda lateral orbital na referencia transversal definida pelo artigo | Nao e o centro escuro da orbita |
| mo | Medial orbit | Crista lacrimal anterior, no plano transversal de ec no protocolo do artigo | Nao equivale automaticamente ao canto interno do olho |
| go | Gonion | Angulo mandibular, na transicao entre ramo e corpo | Vista lateral ajuda; nao usar uma linha fixa de altura |
| pg | Pogonion | Ponto anterior mediano da eminencia mentoniana | A anterioridade exige informacao de perfil/profundidade |
| gn | Gnathion | Ponto mediano entre pogonion e menton, conforme Tabela I deste artigo | Nao confundir com o ponto chamado mento no codigo |

Definicoes resumidas da Tabela I, p. 122, interpretadas junto com a Figura 1.
A figura marca alguns nomes apenas de um lado para ilustrar estruturas bilaterais.

## O que nosso detector marca hoje

| Campo no codigo | O que avaliar agora | O que ele NAO comprova |
|---|---|---|
| orbita_direita / orbita_esquerda | Proposta de centro da cavidade orbital correspondente, sem cair na testa ou na borda externa | Nao sao ec, mo, orbitale nem posicao validada do globo ocular |
| nasal | Proposta de centro da abertura nasal, nao dentes, orbita ou fundo | Nao e nasion, nasospinale ou pronasale |
| mento | Alvo operacional: borda inferior mediana da mandibula visivel | Nao e base da imagem, pescoco, vertebra ou suporte; nao chamar automaticamente de gnathion |
| gonio_direito / gonio_esquerdo | Aproximacoes do angulo mandibular | Pontos a altura fixa do V3 nao sao landmarks validados |
| zigio_direito / zigio_esquerdo | Aproximacoes da regiao lateral zigomatica | Extremos de uma faixa da mascara podem cair em outra estrutura |

Os pontos cinza do piloto V4 sao herdados do V3 e nao foram validados.
O ponto verde e apenas uma proposta de mento. Verde nao significa correto.
O valor de confianca do detector e um escore interno, nao probabilidade de
acerto anatomico. No piloto de 10 imagens, houve nove abstencoes e a unica
proposta de mento ficou no suporte: esse experimento nao foi aprovado.

## Como responder na revisao

- **Correto:** estrutura visivel e proposta compativel com a definicao do campo.
- **Incorreto:** a proposta cai em outra estrutura ou fora do alvo visivel.
- **Nao visivel:** o alvo nao pode ser observado por corte, ausencia, oclusao
  ou qualidade insuficiente. Detector ausente com estrutura visivel nao e
  automaticamente um caso de alvo nao visivel.
- **Nao sei:** falta evidencia ou seguranca para julgar; nao conta como aprovacao.

Avaliar a imagem original e ampliar a regiao. Nao usar apenas simetria ou
proximidade com a figura como criterio. Nao ha neste projeto uma tolerancia
em pixels validada que permita certificar localizacao anatomica precisa.

## O que o handbook acrescenta

Handbook on Craniofacial Superimposition: The MEPROCS Project, Damas,
Cordon e Ibanez. Tabelas 8.1-8.2, p. 145 (pagina 181 do PDF): distinguem
pares de landmarks para orientar a sobreposicao. Por exemplo, apresentam
gnathion craniano / menton facial e crista lacrimal anterior / canto medial.
Isso nao significa que tecido mole e osso devam ocupar o mesmo pixel.

Tabelas 8.5-8.7, pp. 148-149 (paginas 184-185 do PDF): distinguem criterios
por vista e por utilidade para avaliar correspondencia. Localizar um ponto
corretamente e demonstrar que ele discrimina identidades sao tarefas diferentes.
O handbook inclui criterios nao recomendados por subjetividade/baixo poder
discriminativo; nao devemos transformar todos os pontos em pesos iguais.

## Uso no modelo: limites e proximo passo

A figura pode orientar um dicionario de alvos, propostas automaticas e uma
revisao assistida. Ela nao substitui exemplos anotados de cada imagem nem
permite medir erro real de localizacao sozinha. Um modelo pode conferir
plausibilidade, mas nao deve aprovar suas proprias saidas como referencia real.

Antes de treinar um detector supervisionado, fixar os nomes e definicoes,
separar pontos geometricos auxiliares de landmarks anatomicos e obter
referencias independentes. Neste momento o guia serve para aprovar/reprovar
propostas sem exigir que o usuario desenhe pontos. Nao ha novos treinos nem
mudancas de coordenadas causados pela inclusao deste guia.

O artigo de 2026 estudou 70 pares (30 masculinos e 40 femininos), com landmarks
definidos por especialista e comparacoes de quatro candidatos. Seus resultados
com nove pontos superaram os de dezessete naquele experimento; isso nao
valida nosso detector nem pode ser comparado diretamente com uma galeria de 109.
