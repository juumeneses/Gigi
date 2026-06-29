TREINAMENTO CRANIOS

Pipeline automatico para cranios com entrada por vistas CF, CE e CD.

Estrutura esperada:

  dados_originais/cranios/ANO/CF/
  dados_originais/cranios/ANO/CE/
  dados_originais/cranios/ANO/CD/

Somente individuos com CF + CE + CD completos entram no pipeline.
Casos incompletos sao reportados em resultados/checks/problemas_cranios.csv
e ignorados para evitar contaminar medidas, matching e sexo craniano.

CF alimenta contorno, shape 64, medidas e matching geometrico.
CF + CE + CD alimentam sexo craniano por silhuetas 256x256 e VGG16.

Rodar pipeline:

  python scripts/run_pipeline_cranios.py --root .

Com sexo craniano:

  python scripts/run_pipeline_cranios.py --root . --with-sexo --faces-root ../Treinamento_faces

Abrir painel:

  python -m streamlit run scripts/app_revisao_cranios.py

O pipeline nao usa landmarks anatomicos manuais obrigatorios. Ele usa contorno,
pseudo-landmarks geometricos e silhuetas, conforme a metodologia definida para
automatizar a analise e manter o matching compativel com faces.
