APP FINAL V2 — Faces Landmarks + Curadoria para Matching

Como usar:
1) Copie a pasta scripts por cima da pasta scripts do seu projeto Treinamento_faces.
2) Copie também a pasta .streamlit para dentro de Treinamento_faces.
3) Não precisa instalar requirements de novo se seu ambiente já está funcionando.
4) Rode:

   python -m streamlit run scripts/app_revisao_faces.py

Novidades desta versão:
- box de log com altura fixa e opção "Acompanhar últimas linhas";
- menu novo "Imagens" para ver original, marcação ou comparar original x marcação;
- gráficos no app, sem depender de HTML separado;
- gráficos de qualidade, PCA, distribuições de métricas, scores da curadoria, ações e dataset matching;
- legenda/dicionário das métricas com faixas esperadas e interpretação;
- curadoria dentro da aba Revisar/corrigir;
- aplicar recomendação individual ou aplicar recomendações em todas as imagens;
- recalcular curadoria depois das recomendações;
- gera dataset final para matching em:

  resultados/curadoria_matching/dataset_matching_faces.csv

Arquivos importantes:
- resultados/analises_faces/analise_padroes_faces_todos_anos.csv
- resultados/curadoria_matching/status_matching_faces.csv
- resultados/curadoria_matching/dataset_matching_faces.csv
- resultados/revisao_manual_faces/revisao_manual_faces.csv

