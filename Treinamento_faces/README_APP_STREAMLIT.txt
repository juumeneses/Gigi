APP STREAMLIT — REVISÃO E PIPELINE DAS FACES

Arquivos incluídos:
  scripts/app_revisao_faces.py
  requirements_app.txt

Como instalar dentro do seu ambiente .venv:

  python -m pip install -r requirements_app.txt

Como abrir o app:

  streamlit run scripts/app_revisao_faces.py

Se o comando streamlit não existir:

  python -m streamlit run scripts/app_revisao_faces.py

O app permite:
  - escolher a pasta raiz do projeto;
  - rodar pipeline completo;
  - rodar tudo com --force;
  - rodar etapas separadas:
      checagem,
      landmarks,
      medidas,
      análise,
      HTML;
  - escolher ano específico ou todos os anos;
  - acompanhar logs no navegador;
  - ver tabelas de checks, falhas, análises e correções;
  - revisar imagens REVISAR/FALHA;
  - tentar corrigir uma imagem;
  - comparar antes/depois;
  - ver métricas antes/depois;
  - salvar a versão corrigida, mantendo backup da original.

Fluxo recomendado:
  1. Abra o app.
  2. Vá em "Rodar pipeline".
  3. Rode "Pipeline completo".
  4. Vá em "Revisar/corrigir".
  5. Filtre REVISAR/FALHA.
  6. Clique em uma imagem.
  7. Clique em "Tentar corrigir esta imagem".
  8. Compare antes/depois.
  9. Se estiver bom, clique em "Usar esta versão corrigida e salvar backup".
  10. Rode pipeline FORCE do ano ou tudo.

Backups:
  dados_originais/faces/ANO/backup_originais_auto/

Correções sugeridas:
  dados_originais/faces/ANO/images_corrigidas_auto/

Logs do app:
  resultados/logs/

Correções aplicadas:
  resultados/auto_correcao_faces/correcoes_aplicadas_streamlit.csv
