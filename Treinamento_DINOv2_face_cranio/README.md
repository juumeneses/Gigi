# Laboratorio DINOv2 / ConvNeXt: Face-Cranio

Matching experimental sem landmarks obrigatorios. Os extratores ficam congelados
e duas projecoes aprendem os pares por identidade. Similaridade nao e probabilidade
de identidade. O teste final permanece reservado; nao ha validacao forense independente.

## 1. Requisitos

- Acesso ao repositorio privado `juumeneses/Gigi`.
- Git, Git LFS e Python **3.11**, 64 bits, recomendado.
- Rede para preparar os extratores pela primeira vez e espaco para dados,
  pesos, ambiente e novas execucoes. Reserve pelo menos 25 GB livres para
  este conjunto; novos treinos aumentam esse uso.
- CPU funciona. GPU NVIDIA/CUDA ou Apple/MPS podem acelerar a extracao;
  as projecoes pequenas treinam na CPU para retomada deterministica.

Nao copie ambientes `.venv` de um sistema operacional para outro.

## 2. Baixar o repositorio e os dados

```bash
git lfs install
git clone https://github.com/juumeneses/Gigi.git
cd Gigi
git lfs pull
git lfs fsck
```

Se a pasta ja existe, use `git pull --rebase` e `git lfs pull`.
Se aparecer `version https://git-lfs.github.com/spec/v1` no lugar de imagem,
o LFS nao baixou os objetos. Verifique autenticacao e cotas; nao treine com ponteiros.

## 3. Windows (PowerShell)

Na raiz `Gigi`, sem precisar ativar scripts do PowerShell:

```powershell
py -3.11 -m venv .venv-dino
.\.venv-dino\Scripts\python.exe -m pip install --upgrade pip
.\.venv-dino\Scripts\python.exe -m pip install -r Treinamento_DINOv2_face_cranio/requirements.txt
.\.venv-dino\Scripts\python.exe rodar_dino.py --check
.\.venv-dino\Scripts\python.exe rodar_dino.py
```

## 4. macOS / Linux

```bash
python3.11 -m venv .venv-dino
.venv-dino/bin/python -m pip install --upgrade pip
.venv-dino/bin/python -m pip install -r Treinamento_DINOv2_face_cranio/requirements.txt
.venv-dino/bin/python rodar_dino.py --check
.venv-dino/bin/python rodar_dino.py
```

Painel: **http://127.0.0.1:8503**. Se a porta estiver ocupada, acrescente
`--port 8504` ao comando do launcher. O acesso padrao fica restrito ao proprio PC.

`requirements-tested.txt` registra o ambiente de teste do Mac. Para reproduzir
os mesmos pesos, prefira essas versoes quando houver wheels para o destino.
`requirements.txt` permite uma instalacao mais ampla; versoes diferentes devem
ser verificadas antes de comparar resultados. Instale PyTorch/CUDA adequado
pela documentacao oficial: https://pytorch.org/get-started/locally/ .

## 5. Estrutura e caminhos

```text
Gigi/
  Banco/
  rodar_dino.py
  Treinamento_DINOv2_face_cranio/
    portable_roots.json
    artifacts/
  Laboratorio_visual_face_cranio/
  Experimentos_hibridos_cranio_face/dataset/splits/split_identidades.csv
  Treinamento_faces/resultados/
```

Clone o **repositorio completo**, nao so a pasta DINO. Banco e splits sao
encontrados a partir da localizacao do repositorio. `portable_roots.json`
resolve os caminhos antigos das execucoes no novo PC, inclusive entre Mac e
Windows, **sem reescrever os JSONs assinados ou checkpoints**. O launcher registra
a nova raiz para que os proximos resultados possam voltar ao outro computador.

Preserve nomes, IDs e splits. Vistas do matching: FFN neutra, FFS sorrindo,
FL lateral facial, CF frontal craniana e CD lateral direita craniana.
CE e legado e nao entra no matching atual.

## 6. Usar os modelos existentes

1. Abrir Banco e conferir os originais. Inventarios antigos continuam usaveis
   se os arquivos e hashes forem os mesmos.
2. Para fotos novas, clicar Atualizar inventario. Nao redistribuir os splits.
3. Em Treinamentos, selecionar o extrator do modelo salvo e clicar **Preparar
   modelo**. Isso baixa pesos oficiais; **nao treina novamente**.
4. Em Matching, selecionar a execucao existente, sortear ou enviar imagens e Comparar.
5. ConvNeXt exige a mesma versao-base de torchvision registrada no treinamento.

Pesos externos sao verificados por hash. Copiar apenas checkpoints proprios
nao substitui preparar o extrator. A preparacao nao se repete a cada consulta.

## 7. Novo pipeline Top-10

Treinamentos > **Pipeline Top-10: DINOv2 x ConvNeXt e parametros** >
**Iniciar pipeline Top-10**.

Padroes: triplet, pesos 50/30/20, tres seeds, 224 pixels, batch 32, ate 100
epocas e paciencia 15. Compara extratores, testa learning rate/margem e
confirma a configuracao escolhida. Os extratores podem ser preparados automaticamente.

Resultados > Pipeline Top-10 mostra Recall@1/3/5/10, parametros, ganhos,
triagem de uma seed versus confirmacao com tres e exemplos visuais.
**Usar recomendado Top-10 no Matching** seleciona o ensemble dessa rodada.
Detalhes: [PIPELINE_TOP10.md](PIPELINE_TOP10.md).

## 8. O que e transferido

Git/LFS: imagens, codigo, splits, checkpoints proprios, manifests, metricas,
rankings e relatorios salvos. Nao transferidos: `.venv`, pesos externos em
`artifacts/models`, cache `artifacts/features`, jobs/PIDs/locks e uploads
temporarios em `artifacts/uploads`. Historicos feitos com upload temporario
podem ficar sem as fotos no outro PC: preserve esses originais em copia privada.

Antes de sincronizar, espere o treino terminar. Se nao abriu pelo launcher,
execute `python -m Treinamento_DINOv2_face_cranio.portable` para registrar a
raiz desta maquina. Depois `git add`, `commit` e `push`. No outro PC: `git pull
--rebase` e `git lfs pull`. Nao edite a mesma curadoria nas duas maquinas ao mesmo tempo.

GitHub pode impor cotas de armazenamento/banda LFS. Falha de transferencia
nao e motivo para ignorar hashes. Atualizar codigo pode impedir retomada
de treinos antigos: preserve-os e inicie outra execucao, sem editar hashes.

## 9. Diagnostico e testes

Com o Python do novo ambiente:

```bash
python rodar_dino.py --check
python -m unittest discover -s Treinamento_DINOv2_face_cranio/tests
python -m Treinamento_DINOv2_face_cranio.cli --help
```

Os testes funcionais usam dados sinteticos; nao medem qualidade cientifica.
O bloqueio de jobs possui implementacoes Unix e Windows. Os testes locais
foram feitos no Mac; confira o Windows no computador de destino.
