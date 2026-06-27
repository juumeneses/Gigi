Treinamento Faces V4 — FairFace + gênero probabilístico

Instalação, com .venv ativo:
  python -m pip install -r requirements.txt

Rodar o app:
  python -m streamlit run scripts/app_revisao_faces.py

Fluxo recomendado:
1. Rodar pipeline completo.
2. Rodar curadoria das faces e aplicar recomendações.
3. Ir em Gênero / padrões.
4. Rodar FairFace nas faces aprovadas.
5. Aplicar alta confiança e revisar indefinidos/divergentes.
6. Gerar features de gênero.
7. Treinar modelo próprio por métricas/landmarks.
8. Comparar FairFace × modelo próprio.
9. Exportar features finais com gênero.

Arquivo final:
  resultados/features_matching_faces/features_matching_faces_com_genero.csv

Observação: a classificação é de gênero/sexo aparente e probabilística; é um atributo auxiliar, não uma determinação absoluta.
