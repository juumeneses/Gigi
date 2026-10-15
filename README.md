# Gigi

Repositorio **privado** de laboratorios de faces, cranios e matching.
Contem fotografias e dados de pesquisa: nao tornar publico.

## Laboratorio DINOv2 / ConvNeXt

**[Instalacao no Windows, Mac ou Linux](Treinamento_DINOv2_face_cranio/README.md)**

**[Pipeline Top-10](Treinamento_DINOv2_face_cranio/PIPELINE_TOP10.md)**

Instale Git, Git LFS e Python 3.11. Depois:

```bash
git lfs install
git clone https://github.com/juumeneses/Gigi.git
cd Gigi
git lfs pull
```

Crie um ambiente novo; nao copie `.venv` entre sistemas. O README do laboratorio
tem os comandos de instalacao e o launcher `rodar_dino.py`.

## Sincronizacao

Antes de trabalhar: `git pull --rebase` e `git lfs pull`.
Finalize treinamentos antes de sincronizar checkpoints. Depois:

```bash
python -m Treinamento_DINOv2_face_cranio.portable
git status
git add .
git commit -m "Atualiza laboratorio e resultados"
git push origin main
```

O registro `portable_roots.json` permite resolver caminhos historicos em
outro computador sem editar os JSONs assinados ou checkpoints.
Evite editar a mesma curadoria ou execucao simultaneamente em duas maquinas.

Codigo, Banco, splits e resultados/checkpoints proprios usam Git/Git LFS.
Pesos externos, caches, ambientes, jobs/PIDs e uploads temporarios permanecem
locais. Os pesos podem ser preparados novamente pelo painel, sem novo treino.

Os projetos legados de faces, cranios, Modelo_novo, Matching e laboratorios
permanecem no repositorio; o launcher novo abre especificamente o laboratorio DINO.
