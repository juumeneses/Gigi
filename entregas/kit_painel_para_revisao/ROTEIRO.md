# Roteiro de trabalho

## Objetivo

Manter um painel simples, com navegação de site, para consultar resultados existentes de FairFace e modelo próprio e usar informações opcionais fornecidas pela usuária no matching. Este roteiro não inclui construir classificadores automáticos de sexo/gênero por imagens.

## Interface

1. Manter as áreas Início, Banco, Treinamentos, Resultados, Matching e Sexo registrado.
2. Em Sexo registrado, separar Faces e Crânios. Em Faces, permitir consultar FairFace, modelo próprio ou ambos. Reaproveitar registered_ui.results('face').
3. Mostrar referência curada, origem do rótulo e resultado de cada modelo em colunas distintas. Não chamar concordância com curadoria de acurácia biológica.
4. Antes de Comparar, permitir informar o sexo da consulta ou deixar não informado. A origem das informações dos candidatos deve aparecer claramente.
5. Manter a opção de considerar essa informação desligada por padrão. Sem referência para a consulta, preservar o ranking visual e explicar o motivo.
6. Ao filtrar, preservar os scores e o ranking original, manter candidatos sem informação e permitir consultar excluídos e posições originais.
7. Recolher probabilidades, procedência completa, ajustes e logs em áreas opcionais. Evitar botões e tabelas duplicados.
8. Downloads devem corresponder ao filtro atualmente mostrado.

## Integração e preservação

- Trabalhar inicialmente nesta cópia separada.
- Não substituir o app.py original inteiro pela versão adaptada do kit: ele contém outras funcionalidades que não fazem parte deste recorte.
- Usar identificadores explícitos nas uniões, verificar duplicados e ausências.
- Não alterar originais, splits, checkpoints ou resultados anteriores.
- Antes da integração real, devolver o código para revisão e combinar os trechos aprovados com a versão atual.

## Conferências a entregar

- Navegação sem erros, inclusive sem dados.
- Consulta separada de FairFace e modelo próprio.
- Referências ausentes e registros duplicados tratados explicitamente.
- Filtro desligado reproduzindo o ranking original.
- Filtro ligado mantendo desconhecidos e scores visuais.
- Consulta do ranking completo e dos candidatos excluídos.
- CSV acompanhando os filtros da tela.
- Lista dos arquivos alterados, dependências e instruções de execução.

## Devolução à proprietária

Enviar um ZIP somente com os arquivos alterados, preservando os caminhos relativos a Gigi/, mais ALTERACOES.md. A pasta de entrega pode ser colocada em Gigi/entregas/retorno_revisao/. Isso não instala nada automaticamente. Após revisão e backup, os trechos aprovados serão integrados aos arquivos correspondentes.
