# Piloto de pontos, contornos e regioes

Auditoria exploratoria dos 11 pares previamente examinados. Os IDs sao conferidos
no manifesto e nos splits; identidades de teste sao rejeitadas. Nenhum treinamento
ou matching e executado. Originais, CSVs e resultados legados nao sao alterados.

## Executar

Da raiz Gigi, usando o ambiente existente:

```bash
Treinamento_faces/.venv/bin/python Experimentos_hibridos_cranio_face/scripts/piloto_regioes.py --run-dir Experimentos_hibridos_cranio_face/resultados/piloto_regioes/minha_execucao
```

Acrescente `--resume` para retomar a mesma execucao. O hash dos originais, codigo,
configuracao, JSON e arquivos derivados deve conferir. `--force` cria outra pasta,
sem apagar a anterior. `--limit 2` permite interromper de forma controlada apos dois
itens/etapas. `--stage extracao|regioes|estabilidade|relatorio` executa apenas a etapa
indicada; etapas posteriores exigem as anteriores integras. Ao continuar uma pasta
existente, use `--resume`. Alteracao do codigo exige nova execucao.

No macOS, o FaceMesh pode precisar do contexto grafico mesmo usando o grafo CPU.
Uma falha desse ambiente nao significa ausencia de face. O painel deve ser aberto
com a mesma venv; nao com o Python do sistema.

## Saidas e interpretacao

- `config.json`, `entradas.json`: versoes, parametros, manifesto e propostas congeladas.
- Cada imagem: JSON de extracao, regioes e estabilidade, mascaras e recortes.
- `pontos.csv`, `boxes.csv`, `discordancia.csv`, `estabilidade.csv`, `resumo.csv`.
- `relatorio.html`: imagens embutidas, contagens e recomendacoes regionais.
- `execucao.log`, `progresso.json`: andamento e duracao.
- `revisoes/`: registros imutaveis de visibilidade por autor, sessao e hash da saida.

`OK_TECNICO` indica processamento, nao anatomia aprovada. `ATENCAO` inclui
ausencia de deteccao, recorte e segmentacao nao revisada. `ERRO` indica falha.
Coordenadas finitas nao demonstram visibilidade. Nao ha probabilidades de acerto
por ponto nem medidas globais cranianas liberadas.

MediaPipe: static_image_mode=True, refine_landmarks=True, max_num_faces=1,
min_detection_confidence=0.5. Quantidade de pontos registrada por imagem.
Indices candidatos documentados em configs/landmarks_face.yaml; nao representam
equivalencias anatomicas comprovadas. Distancias usam pixels originais e diagonal.

Boxes: retangulo dos pontos necessarios, margem de 20% por eixo (minimo 4px),
recorte letterbox 256, sem imputar ausencias. Boxes cranianas dependem das propostas
anteriores. Os dois pontos orbitais cranianos delimitam uma faixa regional, nao
segmentam toda a orbita. Mandibula/queixo inclui a extensao zigomatica proposta;
nao equivale ao contorno osseo da mandibula.

Otsu usa polaridade clara como primeiro plano e fechamento 3x3. Todos os contornos
externos sao candidatos; fundo claro, suporte e artefatos podem ser incluidos.
Contato com borda e somente um alerta. A mascara precisa de revisao visual.

Estabilidade facial usa original e seis perturbacoes, com transformacao inversa.
Regioes cortadas nao recebem IoU; variantes com corte regional nao recebem
deslocamento agregado. Falhas permanecem no denominador de sete. No cranio mede-se
somente estabilidade da segmentacao; nao foi criado um novo detector de pontos.

## Revisao

Em Piloto de regioes, escolha Revisao cega, modalidade e sessao A/B. O painel
oculta identificadores e resultados da outra sessao; a ordem muda por sessao.
Escolha a regiao, visibilidade, motivo, autor e alertas. A cegueira e de interface,
nao impede acesso aos arquivos locais. Memoria do revisor nao e eliminada; separar
as sessoes no tempo. Compare sessoes somente depois de concluir ambas.

Barba: marque OCULTO/INDETERMINADO quando nao puder reconhecer o limite cutaneo.
Nao remova a identidade nem aceite o queixo so porque MediaPipe devolveu coordenada.
Craneo cortado: registre CORTADO apenas na regiao afetada; mantenha as demais.

Este conjunto pequeno tem pseudorrotulos, selecao previa e nenhuma referencia
anatomica independente. Repetibilidade ou concordancia de revisao nao demonstram
acuracia anatomica. Nao liberar treino ou matching a partir deste piloto.
