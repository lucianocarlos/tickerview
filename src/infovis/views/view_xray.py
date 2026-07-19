import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
from utils.data_loader import load_xai_metadata


@st.dialog("Detalhes do Experimento", width="large")
def render_xray(df_report, exp_id_selecionado):
    if exp_id_selecionado not in df_report["exp_id"].values:
        return

    row = df_report[df_report["exp_id"] == exp_id_selecionado].iloc[0]

    # Encolhendo o título para o mesmo tamanho dos números (1.2em) e cor forte
    st.markdown(
        f"<div style='font-size:1.2em; font-weight:bold; margin-bottom: 10px;'>Detalhes de {exp_id_selecionado}</div>",
        unsafe_allow_html=True,
    )

    m_html = f"""
    <div style='display: flex; gap: 40px; margin-bottom: 20px;'>
        <div><div style='font-size:0.8em; color:gray;'>Val F1</div><div style='font-size:1.2em; font-weight:bold;'>{row["val_f1_score_macro"]:.4f}</div></div>
        <div><div style='font-size:0.8em; color:gray;'>Test F1</div><div style='font-size:1.2em; font-weight:bold;'>{row["test_f1_score_macro"]:.4f}</div></div>
        <div><div style='font-size:0.8em; color:gray;'>Test Prec</div><div style='font-size:1.2em; font-weight:bold;'>{row["test_precision_macro"]:.4f}</div></div>
        <div><div style='font-size:0.8em; color:gray;'>Test Rec</div><div style='font-size:1.2em; font-weight:bold;'>{row["test_recall_macro"]:.4f}</div></div>
    </div>
    """
    st.markdown(m_html, unsafe_allow_html=True)

    col_spider, col_cm = st.columns(2)

    with col_spider:
        st.markdown("**Comparativo Geral**")

        categories = ["V-Acc", "V-F1", "T-Pre", "T-Acc", "T-F1", "T-Rec"]
        valores = [
            row["val_accuracy"],
            row["val_f1_score_macro"],
            row["test_precision_macro"],
            row["test_accuracy"],
            row["test_f1_score_macro"],
            row["test_recall_macro"],
        ]

        fig_spider = go.Figure(
            data=go.Scatterpolar(
                r=valores + [valores[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=exp_id_selecionado,
            )
        )
        fig_spider.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=8, color="rgba(100,100,100,0.7)"),
                    nticks=4,
                    angle=0,
                ),
                angularaxis=dict(
                    showticklabels=True,
                    tickfont=dict(size=8, color="gray"),
                    ticks="",
                    direction="clockwise",
                ),
            ),
            showlegend=False,
            margin=dict(t=20, b=20, l=30, r=30),
            height=250,
        )
        st.plotly_chart(fig_spider, use_container_width=True)

    with col_cm:
        st.markdown("**Matriz de Confusão**")
        try:
            cm = json.loads(row["confusion_matrix"])
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Valores Preditos", y="Valores Reais"),
            )
            fig_cm.update_layout(
                margin=dict(t=20, b=20, l=30, r=30),
                height=250,
                xaxis_title="Predito",
                yaxis_title="Real",
                xaxis=dict(tickmode="linear"),
                yaxis=dict(tickmode="linear"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        except Exception as e:
            st.warning("Matriz de Confusão indisponível.")

    with st.expander("Ver Parâmetros Brutos e Tabela"):
        col_params, col_table = st.columns([3, 7])
        with col_params:
            st.write("**Parâmetros do Modelo**")
            st.json(row["parameters"])
        with col_table:
            st.write("**Tabela de Dados Brutos**")
            st.dataframe(pd.DataFrame([row]), use_container_width=True, height=100)

    # Nova seção de XAI (Explainable AI) extraída do Datalake
    st.markdown("---")
    st.markdown("**Explainable AI (Top 10 Feature Importances)**")

    xai_dict = load_xai_metadata(row["exp_id"], row["dataset_version"])

    if xai_dict:
        # Pega as top 10 features ordenadas
        top_features = sorted(xai_dict.items(), key=lambda item: item[1], reverse=True)[
            :10
        ]
        # Inverte para o Plotly Bar H (o maior fica em cima)
        top_features.reverse()

        names = [f[0] for f in top_features]
        values = [f[1] * 100 for f in top_features]  # Em porcentagem

        fig_xai = px.bar(
            x=values,
            y=names,
            orientation="h",
            labels={"x": "Importância (%)", "y": ""},
            color=values,
            color_continuous_scale="teal",
        )
        fig_xai.update_layout(
            margin=dict(t=10, b=20, l=10, r=10), height=300, coloraxis_showscale=False
        )
        st.plotly_chart(fig_xai, use_container_width=True)
    else:
        st.info(
            "Nenhuma matriz de explicabilidade (XAI) nativa foi encontrada no Datalake para este modelo. Isto ocorre em modelos caixa-preta (ex: Redes Neurais, KNN) ou não-particionados (Logística, Ensembles)"
        )
