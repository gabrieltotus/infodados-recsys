import html
import pickle
import re
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib as jb
from sklearn.metrics import euclidean_distances

# ============================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Recomendador de FIlmes",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. CONFIGURAÇÕES DOS ARQUIVOS
# ============================================================

#PASTA_OBJETOS = Path("objetos")
PASTA_OBJETOS = Path("")

ARQUIVO_OBJETOS = PASTA_OBJETOS / "modelo.pkl"

# ============================================================
# 3. ESTILO VISUAL
# ============================================================

st.markdown(
    """
    <style>
        /* Página */
        .stApp {
            background:
                radial-gradient(
                    circle at top right,
                    rgba(229, 9, 20, 0.13),
                    transparent 28%
                ),
                #080808;
            color: #ffffff;
        }

        /* Remove elementos padrão */
        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        header[data-testid="stHeader"] {
            background-color: transparent;
        }

        /* Limita e centraliza o conteúdo */
        .block-container {
            max-width: 1450px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        /* Cabeçalho */
        .app-header {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 2rem;
        }

        .app-logo {
            color: #00E676;
            font-size: 2.2rem;
            font-weight: 900;
            letter-spacing: -2px;
            line-height: 1;
        }

        .app-description {
            color: #a8a8a8;
            font-size: 0.95rem;
            border-left: 1px solid #383838;
            padding-left: 14px;
        }

        /* Hero */
        .hero-container {
            padding: 2rem 0 1.6rem 0;
        }

        .hero-label {
            color: #00E676;
            font-weight: 700;
            font-size: 0.8rem;
            letter-spacing: 0.16rem;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        }

        .hero-title {
            font-size: clamp(2rem, 5vw, 4.5rem);
            max-width: 850px;
            font-weight: 900;
            line-height: 0.98;
            letter-spacing: -0.15rem;
            margin-bottom: 1rem;
        }

        .hero-text {
            color: #b5b5b5;
            font-size: 1.05rem;
            max-width: 720px;
            line-height: 1.6;
        }

        /* Selectbox */
        div[data-baseweb="select"] > div {
            background-color: #181818;
            border: 1px solid #454545;
            color: white;
            min-height: 48px;
        }

        div[data-baseweb="select"] > div:focus-within {
            border-color: #00E676;
            box-shadow: 0 0 0 1px #00E676;
        }

        div[data-baseweb="popover"] {
            color: white;
        }

        /* Botão */
        div.stButton > button {
            background-color: #00E676;
            color: #080808;
            border: none;
            border-radius: 5px;
            min-height: 48px;
            padding: 0 1.8rem;
            font-weight: 700;
            transition: 0.2s ease;
        }

        div.stButton > button:hover {
            background-color: #22C55E;
            color: #080808;
            border: none;
            transform: scale(1.02);
        }

        /* Títulos das seções */
        .section-title {
            color: #ffffff;
            font-size: 1.45rem;
            font-weight: 800;
            margin-top: 2rem;
            margin-bottom: 0.35rem;
        }

        .section-subtitle {
            color: #8f8f8f;
            font-size: 0.9rem;
            margin-bottom: 1.2rem;
        }

        /* Linha horizontal de filmes */
        .movie-row {
            display: flex;
            gap: 16px;
            overflow-x: auto;
            overflow-y: hidden;
            padding: 10px 4px 24px 4px;
            scroll-snap-type: x mandatory;
        }

        .movie-row::-webkit-scrollbar {
            height: 8px;
        }

        .movie-row::-webkit-scrollbar-track {
            background: #151515;
            border-radius: 10px;
        }

        .movie-row::-webkit-scrollbar-thumb {
            background: #4a4a4a;
            border-radius: 10px;
        }

        .movie-row::-webkit-scrollbar-thumb:hover {
            background: #00E676;
        }

        /* Card do filme */
        .movie-card {
            position: relative;
            flex: 0 0 212px;
            min-height: 245px;
            background:
                linear-gradient(
                    155deg,
                    rgba(229, 9, 20, 0.20),
                    rgba(20, 20, 20, 0.92) 48%
                ),
                #32D583;
            border: 1px solid #333333;
            border-radius: 9px;
            padding: 18px;
            scroll-snap-align: start;
            overflow: hidden;
            transition:
                transform 0.22s ease,
                border-color 0.22s ease,
                box-shadow 0.22s ease;
        }

        .movie-card:hover {
            transform: translateY(-5px) scale(1.015);
            border-color: rgba(229, 9, 20, 0.75);
            box-shadow: 0 13px 32px rgba(0, 0, 0, 0.48);
        }

        .movie-number {
            width: 31px;
            height: 31px;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #00E676;
            color: black;
            border-radius: 50%;
            font-size: 0.77rem;
            font-weight: 800;
            margin-bottom: 42px;
        }

        .movie-badge {
            display: inline-block;
            color: #e2e2e2;
            border: 1px solid #555555;
            border-radius: 3px;
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.05rem;
            padding: 3px 6px;
            margin-bottom: 9px;
        }

        .movie-title {
            color: #ffffff;
            font-size: 1.08rem;
            font-weight: 800;
            line-height: 1.25;
            margin-bottom: 8px;
            overflow-wrap: anywhere;
        }

        .movie-year {
            color: #b7b7b7;
            font-size: 0.82rem;
            margin-bottom: 13px;
        }

        .movie-genres {
            color: #909090;
            font-size: 0.77rem;
            line-height: 1.45;
        }

        .movie-indicator {
            position: absolute;
            width: 52px;
            height: 4px;
            left: 18px;
            bottom: 17px;
            background-color: #00E676;
            border-radius: 8px;
        }

        /* Informações */
        .selected-movie {
            border-left: 4px solid #00E676;
            background-color: #151515;
            padding: 13px 16px;
            border-radius: 0 6px 6px 0;
            color: #d7d7d7;
            margin-top: 0.8rem;
        }

        .empty-state {
            color: #b0b0b0;
            background-color: #151515;
            border: 1px dashed #3b3b3b;
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
        }

        /* Rodapé */
        .app-footer {
            border-top: 1px solid #242424;
            color: #656565;
            font-size: 0.75rem;
            margin-top: 3.5rem;
            padding-top: 1.3rem;
            text-align: center;
        }
        .app-logo {
            color: #00E676;
            font-size: 2rem;
            font-weight: 900;
            letter-spacing: -1px;
        }

        .app-description-header {
            color: #A8A8A8;
            font-size: 0.9rem;
            margin-top: 3px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. CARREGAMENTO DOS OBJETOS
# ============================================================

dict_modelo     = jb.load(ARQUIVO_OBJETOS)
df_filmes       = dict_modelo['BASE_DADOS']
variaves_modelo = dict_modelo['VARIAVEIS']
modelo_kmeans   = dict_modelo['MODELO']

COLUNA_TITULO = "TITLE"
COLUNA_GENERO = "GENRES"


# ============================================================
# 6. FUNÇÕES DE TRATAMENTO DOS RESULTADOS
# ============================================================

def separar_titulo_ano(titulo):
    """
    Separa títulos no formato:
    Toy Story (1995)

    Retorno:
    ("Toy Story", "1995")
    """

    titulo = str(titulo).strip()

    resultado = re.search(
        r"\((\d{4})\)\s*$",
        titulo
    )

    if resultado:
        ano = resultado.group(1)

        nome = re.sub(
            r"\s*\(\d{4}\)\s*$",
            "",
            titulo
        ).strip()

        return nome, ano

    return titulo, "Ano não informado"


def preparar_recomendacoes(resultado):
    """
    Normaliza o DataFrame retornado por get_recomendação().

    A função aceita title ou TITLE, genres ou GENRES e
    year ou YEAR.
    """

    if resultado is None:
        return pd.DataFrame()

    if isinstance(resultado, list):
        resultado = pd.DataFrame(resultado)

    if not isinstance(resultado, pd.DataFrame):
        raise TypeError(
            "get_recomendação() deve retornar um DataFrame "
            "ou uma lista de dicionários."
        )

    resultado = resultado.copy()

    if "TITLE" not in resultado.columns:
        raise ValueError(
            "O resultado de get_recomendação() precisa "
            "possuir a coluna 'TITLE'."
        )

    if "GENRES" not in resultado.columns:
        resultado["genres"] = "Gênero não informado"
    else:
        resultado["genres"] = resultado["GENRES"]

    nomes = resultado["TITLE"].apply(separar_titulo_ano)

    resultado["nome_exibicao"] = nomes.str[0]

    if "year" not in resultado.columns:
        resultado["year"] = nomes.str[1]
    else:
        resultado["year"] = (
            resultado["year"]
            .fillna(nomes.str[1])
            .astype(str)
        )

    return resultado


def criar_card_filme(filme, posicao):
    """
    Cria o HTML de um card.

    html.escape evita que textos dos dados sejam
    interpretados como marcação HTML.
    """

    titulo = html.escape( str(filme.get("nome_exibicao", "Título não informado")) )

    ano = html.escape( str(filme.get("year", "Ano não informado")) )

    generos = str( filme.get("genres", "Gênero não informado")  )

    generos = generos.replace("|", " • ")

    generos = html.escape(generos)

    card = (    
        f'<article class="movie-card">'
        f'<div class="movie-number">{posicao:02d}</div>'
        f'<div class="movie-badge"> RECOMENDADO </div>'
        f'<div class="movie-title"> {titulo} </div>'
        f'<div class="movie-year"> {ano} </div>'
        f'<div class="movie-genres"> {generos} </div>'
        f'<div class="movie-indicator"></div>'
        f'</article>'
    )
    return card

def exibir_linha_recomendacoes(recomendacoes):
    """
    Exibe todos os filmes em uma única faixa horizontal.
    """

    cards = []
    #cards = ""
    for posicao, (_, filme) in enumerate(recomendacoes.iterrows(),start=1):
        cards.append( criar_card_filme( filme = filme, posicao=posicao) )
        

    linha_html = (
        '<section class="movie-row">' + "".join(cards) + "</section>"
    )
    
    st.markdown(
        linha_html,
        unsafe_allow_html=True
    )


# ============================================================
# 7. FUNÇÃO DO RECOMENDADOR
# ============================================================
def get_recomendacao(filme_selec, qtd_recomendacao = 10 ):
    """
    FUNÇÃO COM O PROCEDIMENTO PARA GERAR AS RECOMENDAÇÕES
    """

    # PREPARANDO NOVO FILME
    #filme_selecionado = df_filmes.query(f"MOVIEID == {id_filme_selec} ")
    filme_selecionado = df_filmes.query(f"TITLE == '{filme_selec}' ")

    ## RETORNANDO O CLUSTER QUE O FILME PERTENCE
    cluster_filme_selec  = filme_selecionado.CLUSTER.values[0]

    ## FILTRANDO FILMES DO CLUSTER INDICADO
    df_recomendacao = df_filmes.query(f"CLUSTER == {cluster_filme_selec} and TITLE != '{filme_selec}' ")

    ## ESTRUTURANDO OS GENEROS DO FILME SELECIONADO NA MESMA ESTRUTURA
    # DA TABELA DE TREINAMENTO DO ALGORITMO
    genero_filme_selec = filme_selecionado.GENRES.str.get_dummies(sep="|")
    
    ## IGUALANDO A ESTUTURA COM A BASE DE TREINAMENTO
    genero_filme_selec = genero_filme_selec.reindex(columns=variaves_modelo, fill_value=0)
    
    ## LISTA PARA ARMAZENAR OS VALORES DE SIMILARIDADE
    lista_similaridade = []

    ## FOR PARA PERCORRER TODOS OS FILMES DO CLUSTER SELECIONADO
    print("** CALCULANDO SIMILARIDADE DOS FILMES **")
    for id in df_recomendacao.MOVIEID[:]:
        resultado = df_recomendacao.query(f"MOVIEID == {id}")

        ## ESTRUTURANDO GENERADO NO FORMATO CORRETO
        genero_filme_result = resultado.GENRES.str.get_dummies(sep="|")
        genero_filme_result = genero_filme_result.reindex(columns=variaves_modelo, fill_value=0)
        
        ## CALCULANDO A SIMILARIDADE
        similaridade = euclidean_distances(genero_filme_selec, genero_filme_result)[0][0]
        lista_similaridade.append(similaridade)

    ## CRIANDO COLUNA SIMILARIDADE PARA ORDENAR OS TOP N
    df_recomendacao['SIMILARIDADE'] = lista_similaridade
    df_recomendacao = df_recomendacao.sort_values('SIMILARIDADE').head(qtd_recomendacao)
    
    return df_recomendacao

# ============================================================
# 8. CABEÇALHO
# ============================================================
col_header, col_logo = st.columns( [5, 1], vertical_alignment="center")

with col_header:
    st.markdown(
        """
        <div class="app-header">
        <div class="app-logo">Recomendador de Filmes</div>
        <div class="app-description">Recomendação de filmes com Machine Learning</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <section class="hero-container">
            <div class="hero-label">Dos Dados à Decisção</div>
            <div class="hero-title">
                Descubra sua próxima história.
            </div>
            <div class="hero-text">Escolha um filme que você assistiu e encontre outras produções com características semelhantes.</div>
        </section>
        """,
        unsafe_allow_html=True
    )
with col_logo:
    st.image("LOGO_INFODADOS_branco.png",width=350)


# ============================================================
# 9. SELEÇÃO DO FILME
# ============================================================

lista_filmes = sorted(
    df_filmes[COLUNA_TITULO]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

coluna_filme, coluna_quantidade, coluna_botao = st.columns([5, 1.3, 1.5], vertical_alignment="bottom")

with coluna_filme:
    filme_selecionado = st.selectbox(
        label="Qual filme você assistiu?",
        options=lista_filmes,
        index=None,
        placeholder="Digite ou selecione um filme...",
        key="filme_selecionado"
    )

with coluna_quantidade:
    quantidade = st.selectbox(
        label="Quantidade",
        options=list(range(5, 11)),
        index=0,
        key="quantidade_recomendacoes"
    )

with coluna_botao:
    recomendar = st.button(
        "Recomendar",
        type="primary",
        use_container_width=True
    )


# ============================================================
# 10. EXECUÇÃO DA RECOMENDAÇÃO
# ============================================================

if recomendar:

    if filme_selecionado is None:
        st.warning("Selecione um filme antes de solicitar as recomendações.")

    else:
        st.markdown(
            f"""
            <div class="selected-movie"> Porque você assistiu a <strong>{html.escape(filme_selecionado)}</strong></div>
            """, 
            unsafe_allow_html=True
        )

        try:
            with st.spinner("Procurando filmes semelhantes..."):
                resultado     = get_recomendacao(filme_selecionado, quantidade)
                recomendacoes = preparar_recomendacoes(resultado)

            if recomendacoes.empty:
                st.markdown(
                    """
                    <div class="empty-state"> Não encontramos recomendações para esse filme. </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    """
                    <div class="section-title"> Filmes recomendados para você </div>

                    <div class="section-subtitle"> Arraste horizontalmente para conhecer todas as recomendações. </div>
                    """,
                    unsafe_allow_html=True
                )

                exibir_linha_recomendacoes( recomendacoes )

        except Exception as erro:
            st.error( "Não foi possível gerar as recomendações.")

else:
    st.markdown(
        """
        <div class="empty-state">
            Selecione um filme e clique em
            <strong>Recomendar</strong> para começar.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 11. RODAPÉ
# ============================================================

st.markdown(
    """
    <div class="app-footer">
        Treinamento realizado por: GDG Brasília e INFO DADOS
        <br>
        Responsável: Gabriel Gomes
    </div>
    """,
    unsafe_allow_html=True
)