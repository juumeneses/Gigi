# Referencia craniana: lote inicial de 30 imagens

Base: Srisinghasongkram et al., Int. J. Morphol. 44(1):121-129 (2026),
Tabela I p.122 e Figura 1 p.123. A figura nao fornece coordenadas para o acervo.

## Conjuntos

- Nove pontos: g, n, ns, pr, id, zy direito/esquerdo, al direito/esquerdo.
- Dezessete: os nove anteriores + pg, gn, ec direito/esquerdo, mo direito/esquerdo,
  go direito/esquerdo.
- Comecar pelo conjunto de nove, registrando invisibilidade e duvida. Os adicionais
  nao sao obrigatorios para concluir a triagem inicial e nao devem ser inventados.
- O artigo nao garante que esses pontos sejam suficientes para nosso matching.

## Convencoes a ratificar antes do treino

X cresce para direita da imagem e Y para baixo, na resolucao original CF.
Direita/esquerda anatomicas sao do individuo, nao do observador. Confirmar se as
imagens nao foram espelhadas; se desconhecido, registrar em observacao e nao validar.
CE/CD servem para entender a anatomia, nunca para copiar coordenadas para CF.
Para al, a interpretacao ossea da Figura 1 precisa ser ratificada, pois a Tabela I
usa terminologia de asa nasal. Para ec/mo, fixar o plano transversal em protocolo
especializado. Pg/gn podem exigir perfil: nao inferir profundidade da CF.

## Anotar e verificar

LOCALIZADO exige coordenadas visiveis na CF. NAO_VISIVEL e DUVIDA ficam sem X/Y.
PENDENTE significa nao avaliado, nao ausencia. Um detector que nao encontra um
ponto nao permite rotular esse ponto como invisivel.
Identificar o responsavel. Todas as entradas sao rascunhos, mesmo se completas.
A revisao especializada deve confirmar nome, lado, visibilidade e coordenada,
com identidade do revisor e versao. Nenhuma caixa de confirmacao substitui isso.
Cada salvamento cria um arquivo com data; nao apaga as versoes anteriores.
Os originais sao somente leitura e conferidos por hash ao salvar.

## Selecao e avaliacao

Seis identidades por ano (2017, 2019, 2021, 2022, 2023), apenas no grupo de
desenvolvimento definido anteriormente. Selecao espacada pelo score de qualidade
existente, que nao e precisao anatomica. Nao substitui amostragem de tipos de falha.
Teste e amostra de aceitacao permanecem fora. Este lote nao autoriza misturar
identidades de validacao no futuro treino: respeitar a coluna split.
Uma avaliacao futura exige anotacoes independentes nas imagens reservadas.
Novas fotos da mesma identidade nao podem atravessar os splits.

## Bloqueios

Sem referencia verificada, nao exportar YOLO, nao iniciar treinamento e nao liberar
matching. Nao importar centros de orbitas como ec/mo, nem mento legado como gn.
Primeiro ratificar definicoes e revisar o lote; depois dimensionar anotacoes
adicionais, fixar protocolo de erro por ponto e executar treino piloto.
