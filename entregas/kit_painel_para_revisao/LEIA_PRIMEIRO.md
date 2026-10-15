# Kit de referência do painel DINO

Este pacote permite conhecer a estrutura do painel sem acesso ao computador da proprietária.
É uma cópia parcial e adaptada para revisão de interface, consulta de resultados históricos e filtro opcional por informações fornecidas. Não é um backup completo nem deve substituir o projeto original.

O pacote não contém imagens, nomes/IDs do banco, CSVs pessoais, checkpoints, ambientes virtuais, caches ou módulos de treinamento/inferência automática de sexo/gênero. FairFace e modelo próprio aparecem somente no código de consulta dos resultados existentes. Não há resultados pessoais incluídos.

## Começar

1. Extraia o ZIP em uma pasta de trabalho separada no seu computador.
2. Leia ROTEIRO.md e MAPA_ARQUIVOS.md.
3. O código está em codigo_referencia/. A estrutura relativa corresponde à pasta Gigi do projeto original.
4. Sem Banco, splits, inventários e resultados, não é possível reproduzir os treinos ou comparar o desempenho real. A interface pode ser estudada e aberta vazia após instalar as dependências.
5. Ao concluir, devolva apenas os arquivos alterados e a lista das mudanças. A proprietária fará backup e revisão antes de integrar ao projeto real.

Para uma prévia vazia em macOS/Linux, em codigo_referencia/, com um ambiente Python e as dependências instaladas:

    python -m streamlit run Treinamento_DINOv2_face_cranio/app.py

O código herdado usa fcntl e requer adaptação para executar no Windows. Nenhuma dependência foi instalada por este pacote.

## Arquivos adaptados nesta cópia

- app.py: seção de sexo limitada à consulta de resultados históricos.
- sex_controls.py: somente leitura de referências, consulta de resultados já registrados e filtro opcional do ranking; sem treino ou inferência.
- cli.py: removido o encaminhamento de tarefas de inferência/treino de sexo.
- registered.py: leitor de referência independente, sem importar o módulo de inferência.
- references_io.py: novo leitor de referências com verificação de integridade.

Os arquivos originais do computador da proprietária não foram alterados ao gerar este kit.
