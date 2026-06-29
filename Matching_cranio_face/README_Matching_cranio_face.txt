MATCHING CRANIO-FACE

Painel Streamlit para comparar uma face com multiplos cranios.

Rodar:

  python -m streamlit run scripts/app_matching_cranio_face.py

V1:
- upload de 1 face
- upload de multiplos cranios
- extracao temporaria de shape 64
- ranking por similaridade geometrica
- genero/sexo como peso auxiliar, nunca filtro absoluto

V2 planejada:
- arquitetura two-tower baseada na monografia
- torre craniana VGG16
- torre facial MLP sobre PCA dos landmarks
- similaridade de cosseno
- treino binario com pares positivos/negativos
