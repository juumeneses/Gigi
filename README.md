# Gigi

Repositorio privado para os projetos de treinamento de faces, cranios e matching.

As imagens originais e as marcacoes sao armazenadas com Git LFS. Ambientes virtuais, modelos externos, logs, crops e arquivos intermediarios nao sao enviados ao GitHub.

## Estrutura

```text
Gigi/
  Painel_Geral/
  Treinamento_faces/
  Treinamento_cranios/
  Matching_cranio_face/
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

### Painel geral

Na raiz do projeto, abre um painel com Face, Crânio e Matching no topo:

```bash
cd /Users/giovanni/Documents/GitHub/Gigi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-compile -r requirements.txt
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8500
```

Depois que a `.venv` da raiz ja existir, basta:

```bash
cd /Users/giovanni/Documents/GitHub/Gigi
.venv/bin/python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8500
```

### Crânios no macOS ou Linux

```bash
cd Treinamento_cranios
python -m pip install -r requirements_cranios.txt
python -m streamlit run scripts/app_revisao_cranios.py
```

Pipeline de crânios:

```bash
cd Treinamento_cranios
python scripts/run_pipeline_cranios.py --root .
```

Treino de gênero/sexo craniano separado:

```bash
cd Treinamento_cranios
python scripts/05_preparar_silhuetas_sexo_cranio.py --root .
python scripts/06_juntar_genero_faces_cranios.py --root . --faces-root ../Treinamento_faces
python scripts/07_treinar_sexo_cranio.py --root .
python scripts/08_exportar_features_cranios_final.py --root .
```

### Matching crânio-face

```bash
cd Matching_cranio_face
python -m pip install -r requirements_matching.txt
python -m streamlit run scripts/app_matching_cranio_face.py
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
