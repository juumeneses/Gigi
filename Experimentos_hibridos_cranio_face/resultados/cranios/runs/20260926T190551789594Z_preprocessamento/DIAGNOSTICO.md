# Teste de preprocessamento

10 imagens de desenvolvimento, 5 variantes, 4 pontos por imagem. Nenhum treino.
Originais conferidos por SHA-256 antes/depois; nenhuma alteracao encontrada.

| Variante | Propostas / 40 |
|---|---:|
| Original | 5 |
| CLAHE | 7 |
| Bilateral | 5 |
| Brilho x1.10 | 5 |
| Escala x0.85 | 6 |

Apenas 2 pontos tiveram proposta nas cinco variantes e desvio maximo de ate
2% da diagonal original. Estabilidade nao mede acuracia anatomica.
O comparativo de 2021_94_RMA foi examinado visualmente: o ponto orbital desaparece
com filtro bilateral e as bordas nasais mudam com contraste. Nao promover
CLAHE a padrao apenas por gerar mais propostas; faltam coordenadas verificadas.

Coordenadas de escala foram convertidas para o original usando fatores efetivos
de largura/altura, incluindo arredondamento. Transformacoes e propostas por variante
estao em variantes/. O painel mostra contact sheets em Original e variantes de
processamento. As revisoes de pontos continuam sobre o original; nenhuma variante
substitui automaticamente a anotacao de referencia.

Conclusao: preprocessamento isolado nao resolveu localizacao orbital. Proxima etapa
continua sendo busca local de rebordos ou detector supervisionado com referencia
verificada, nao treinamento em massa sobre estas propostas.
