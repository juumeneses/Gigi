TREINAMENTO FACES — APP FINAL COMPLETO

Substitua/copiei os arquivos deste zip por cima da pasta Treinamento_faces.

Instalar dependências dentro do .venv:

  python -m pip install --upgrade pip
  python -m pip install -r requirements_app.txt

Testar MediaPipe:

  python -c "import mediapipe as mp; print(mp.__version__); print(hasattr(mp, 'solutions'))"

O último valor deve ser True.

Rodar só o app final:

  python -m streamlit run scripts/app_revisao_faces.py

O app inclui:
- rodar pipeline completo;
- rodar etapas separadas;
- painel com análise e PCA;
- revisão individual;
- curadoria automática para matching dentro da aba Revisar/corrigir;
- logs ao vivo;
- resultados da curadoria;
- dataset final para matching.

Arquivo final para matching:

  resultados/curadoria_matching/dataset_matching_faces.csv

Fluxo recomendado:
1. Abrir o app.
2. Rodar pipeline completo.
3. Ir em Revisar/corrigir.
4. Rodar curadoria primeiro sem aplicar correções.
5. Conferir antes/depois.
6. Se estiver ok, marcar aplicar correções com backup e rodar novamente.
