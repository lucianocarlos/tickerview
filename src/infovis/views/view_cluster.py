import streamlit as st

@st.fragment
def render_cluster(df_mestre):
    st.subheader("Análise Exploratória (Aba A)")
    st.info("Aqui entrará o gráfico de dispersão (Scatter Plot) das ações com filtros de Setor/Indústria.")
