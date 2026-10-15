"""Plain-language metric definitions shared by the panel."""
import streamlit as st


HELP = {
    'Extrator': 'Rede pre-treinada congelada que transforma imagens em vetores. Cada extrator exige novas projecoes treinadas; resultados antigos nao mudam.',
    'MRR': 'De 0 a 1; maior e melhor. Media de 1/posicao do correto: primeiro = 1, segundo = 0,5. Falha = 0. Nao e porcentagem de acerto.',
    'Score': 'Maior significa maior similaridade dentro deste ranking, nao identidade confirmada.',
    'Cosseno': 'De -1 a 1: mais perto de 1 significa vetores mais semelhantes. Nao existe limiar de identidade validado.',
    'Score combinado': 'Soma ponderada dos cossenos padronizados pelo treino. Pode ser negativo ou positivo, sem limite fixo. Zero e a referencia media ponderada dos pares diferentes no treino, nao um limiar de identidade. Maior e melhor apenas dentro da mesma combinacao.',
    'Margem': 'Score do primeiro menos o do segundo. Zero indica empate; maior distancia nao e garantia de acerto.',
    'Contribuicao ao score': 'Peso vezes score padronizado do modelo. Pode ser negativa. A soma das contribuicoes e o score final; nao e porcentagem de certeza.',
    'Peso': 'Participacao escolhida para o modelo; os pesos ativos somam 1. Nao mede importancia anatomica.',
    'Consultas': 'Numero de faces avaliadas, incluindo falhas.',
    'Candidatos': 'Numero de cranios concorrentes por consulta. Galerias maiores costumam ser mais dificeis.',
    'Seed': 'Controla a aleatoriedade. Seeds diferentes testam a estabilidade do treinamento.',
    'Loss': 'Funcao de treinamento: InfoNCE ou triplet. Valores de losses diferentes nao sao diretamente comparaveis.',
    'Protocolo': 'Identifica pessoas, divisao e galeria comparaveis. Nao compare ganhos entre protocolos diferentes.',
    'Data (Sao Paulo)': 'Criacao da execucao ou sequencia no horario de Sao Paulo; retomar preserva a data.',
    'Ganho MRR vs melhor anterior': 'MRR atual menos o melhor de uma execucao anterior compativel. Positivo = aumento; negativo = queda; nao prova ganho estatistico.',
    'Diferenca MRR vs melhor atual': 'Distancia ao maior MRR do mesmo protocolo. Zero = lider ou empate; negativo = abaixo do lider.',
    'Evolucao': 'Comparacao descritiva com o melhor anterior, nao conclusao estatistica.',
    'Posicao': '1 e o primeiro do ranking; numero menor e melhor. Ausente significa que nao foi possivel calcular score.',
    'Pessoa': 'Identificador vinculado ao banco, nao uma caracteristica usada no modelo.',
    'Pessoa de referencia': 'Identificador da face consultada, vinculado ao banco. Nao e usado como entrada visual do modelo.',
    'Referencia conhecida': 'O cranio correspondente a face esta identificado na galeria; permite conferir a posicao correta.',
    'Candidato': 'Identificador do cranio concorrente.', 'Vistas': 'Imagens disponiveis para este candidato no modelo selecionado.',
    'Vista': 'FFN: face neutra; FFS: sorrindo; FL: lateral; CF: cranio frontal; CE/CD: laterais cranianas.',
    'Ano': 'Ano do conjunto de origem.', 'Divisao': 'Treino ajusta pesos; validacao escolhe configuracoes; teste final fica reservado.',
    'Estado': 'Situacao tecnica do arquivo, nao validacao anatomica.', 'Problema': 'Motivo de falha ou exclusao.',
    'Usado': 'Indica se este modelo participa do score selecionado.',
    'Combinacao': 'Modelos cujos scores foram combinados para formar este ranking.',
    'Primeiro colocado': 'Candidato com maior score nesta combinacao.',
    'Posicao do correto': 'Posicao do cranio conhecido. Ausente para uploads sem referencia ou quando nao ha score valido.',
    'Score do primeiro': 'Maior score da combinacao; sem limite fixo ou probabilidade calibrada. Nao comparar escalas entre combinacoes.',
    'Candidatos com score': 'Quantidade processada com entradas suficientes para esta combinacao.',
    'Learning rate': 'Tamanho dos passos de ajuste dos pesos.', 'Batch': 'Pessoas usadas em cada passo de treinamento.',
    'Epocas limite': 'Maximo de passagens pelo treino.', 'Epocas executadas': 'Passagens realizadas antes de parar.',
    'Melhor epoca (MRR)': 'Epoca com o maior MRR de validacao.', 'Paciencia': 'Epocas sem melhora antes de parar.',
    'Weight decay': 'Regularizacao dos pesos; nao e uma medida de acerto.', 'Resolucao': 'Lado da entrada em pixels, com proporcoes preservadas.',
    'Temperatura': 'Escala dos logits da InfoNCE; nao e usada pela triplet.', 'Margem triplet': 'Separacao alvo da triplet; nao e usada pela InfoNCE.',
}


def column_help(name):
    name = str(name)
    if name.startswith('Top-'):
        return 'Sim: correto entre os K primeiros nesta consulta. Nao: fora do Top-K ou sem score. Nao avaliavel: sem identidade conhecida. Nao e Recall agregado.'
    if name.isdigit():
        return 'Coluna de blocos do mapa; valor = score original menos score com o bloco ocultado. Nao e probabilidade.'
    if name.startswith('Recall@'):
        return 'De 0% a 100%: consultas com o cranio correto entre os K primeiros. Maior e melhor; os grupos se sobrepoem.'
    if 'Recall@' in name and 'p.p.' in name:
        return 'Diferenca em pontos percentuais contra a referencia indicada. +5 p.p. significa, por exemplo, de 10% para 15%.'
    return HELP.get(name, {'Execucao':'Identificador da execucao salva.', 'Modelo':'Vistas e modelo usados.',
        'Tipo':'Modelo individual ou combinacao de scores.', 'Melhor atual (MRR)':'Execucao lider por MRR no mesmo protocolo.',
        'Melhor anterior':'Lider compativel antes desta execucao.'}.get(name, name.replace('_', ' ')))


def table(data, **kwargs):
    columns = list(data.columns) if hasattr(data, 'columns') else list(data[0]) if len(data) else []
    st.dataframe(data, hide_index=True, width='stretch', height=min(360, 40+35*len(data)),
                 column_config={str(c): st.column_config.Column(str(c), help=column_help(c)) for c in columns}, **kwargs)


def metrics_help():
    with st.popover('Entender os numeros', icon=':material/help:'):
        for label in ('Cosseno', 'Score combinado', 'Margem', 'MRR', 'Contribuicao ao score'):
            st.markdown('**'+label+'**: '+HELP[label])
        st.markdown('**Recall@K:** percentual de consultas em que o correto aparece entre os K primeiros; vai de 0% a 100%.')
        st.caption('Score de um par e desempenho do banco sao coisas diferentes. Nenhum score aqui e probabilidade de identidade.')
