# Gigi

Repositorio privado para os projetos de treinamento de faces, cranios e matching.

As imagens originais e as marcacoes sao armazenadas com Git LFS. Ambientes virtuais, modelos externos, logs, crops e arquivos intermediarios nao sao enviados ao GitHub.

## Estrutura

```text
Gigi/
  Treinamento_faces/
  Treinamento_cranios/  # futuro
  Matching/             # futuro
```

## Primeiro uso em um computador

Instale Git e Git LFS. Depois clone o repositorio e baixe as imagens:

```bash
git clone URL_DO_REPOSITORIO
cd Gigi
git lfs install
git lfs pull
```

### Faces no macOS ou Linux

```bash
cd Treinamento_faces
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run scripts/app_revisao_faces.py
```

### Faces no Windows PowerShell

```powershell
cd Treinamento_faces
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run scripts/app_revisao_faces.py
```

## Trabalhar em dois computadores

Antes de iniciar o trabalho:

```bash
git pull --rebase
git lfs pull
```

Depois de alterar codigo, imagens ou decisoes de curadoria:

```bash
git status
git add .
git commit -m "Descreva a alteracao"
git push
```

Evite editar os mesmos CSVs de revisao simultaneamente em dois computadores. Finalize o trabalho com `push` e execute `pull --rebase` no outro computador antes de continuar.

## O que e sincronizado

- Codigo, configuracoes, documentacao e requirements.
- Imagens originais e imagens de marcacao/previews.
- CSVs de revisao, curadoria, metricas e comparacoes.
- Modelos proprios treinados e datasets finais.

## O que permanece local

- `.venv`, caches e dependencias instaladas.
- Logs e relatorios temporarios.
- Modelo FairFace externo, baixado novamente quando necessario.
- Crops, correcoes, backups e outros derivados regeneraveis.

Este repositorio deve permanecer privado porque contem fotografias de pessoas.
