import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from views.view_xray import render_xray
from utils.data_loader import load_xai_metadata, load_bulk_xai_metadata
import pandas as pd


def render_summary_card(df_filtered, selected_exp, sort_metric, panel_id, summary_view, ribbon_mode, current_dataset, df_filtered_full=None):
    #    st.markdown(
    #        f"<div style='font-size: 0.9em; font-weight: bold; text-align: center;'>{summary_view}</div>",
    #        unsafe_allow_html=True,
    #    )

    if df_filtered.empty:
        st.warning("Sem dados.")
        return

    val_selected = None
    row_selected = None
    has_selection = False
    if selected_exp is not None and selected_exp in df_filtered["exp_id"].values:
        row_selected = df_filtered[df_filtered["exp_id"] == selected_exp].iloc[0]
        val_selected = row_selected[sort_metric]
        has_selection = True

    fig = go.Figure()

    if summary_view == "Densidade (Features)":
        xai_bulk = load_bulk_xai_metadata(current_dataset)
        if not xai_bulk.empty:
            df_for_features = df_filtered_full if df_filtered_full is not None else df_filtered
            xai_filtered = xai_bulk[xai_bulk["model_id"].isin(df_for_features["exp_id"])].copy()
            xai_filtered["abs_imp"] = xai_filtered["importance_value"].abs()
            
            # Ponderação
            grouped = xai_filtered.groupby(['feature_name', 'model_name']).agg(
                ocorrencias=('model_id', 'count'),
                media=('abs_imp', 'mean')
            ).reset_index()
            grouped['ocorrencia_ponderada'] = grouped['ocorrencias'] * grouped['media']
            
            # Obter top 10 features globalmente (soma da ocorrência ponderada)
            top_10 = grouped.groupby('feature_name')['ocorrencia_ponderada'].sum().nlargest(10).index
            plot_data = grouped[grouped['feature_name'].isin(top_10)].copy()
            
            # Gráfico de Barras Empilhadas
            fig_px = px.bar(
                plot_data, 
                x="ocorrencia_ponderada", 
                y="feature_name", 
                color="model_name", 
                orientation="h",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            fig_px.update_layout(
                yaxis=dict(categoryorder="total ascending", tickfont=dict(size=8, color="gray"), title=""),
                xaxis=dict(title=dict(text="Ocorrência Ponderada", font=dict(size=8)), tickfont=dict(size=8, color="gray")),
                showlegend=True,
                legend=dict(
                    title="",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=8)
                ),
                margin=dict(t=5, b=0, l=5, r=5),
                height=270,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig = fig_px
        else:
            fig.add_annotation(text="Sem dados de Features", showarrow=False)

    elif summary_view == "SHAP Summary":
        xai_bulk = load_bulk_xai_metadata(current_dataset)
        if not xai_bulk.empty:
            df_for_features = df_filtered_full if df_filtered_full is not None else df_filtered
            xai_filtered = xai_bulk[xai_bulk["model_id"].isin(df_for_features["exp_id"])].copy()
            xai_filtered["abs_imp"] = xai_filtered["importance_value"].abs()
            
            # Ponderação para definir a ordem do Eixo Y
            grouped_ord = xai_filtered.groupby(['feature_name']).agg(
                ocorrencias=('model_id', 'count'),
                media=('abs_imp', 'mean')
            )
            grouped_ord['ocorrencia_ponderada'] = grouped_ord['ocorrencias'] * grouped_ord['media']
            top_global = grouped_ord.sort_values(by='ocorrencia_ponderada', ascending=False).head(10).index
            
            xai_plot = xai_filtered[xai_filtered["feature_name"].isin(top_global)]
            
            # SHAP-like Beeswarm Plot
            fig.add_trace(go.Box(
                x=xai_plot["importance_value"],
                y=xai_plot["feature_name"],
                orientation='h',
                boxpoints='all',
                jitter=0.5,
                pointpos=0, # centers the points
                fillcolor='rgba(255,255,255,0)',
                line=dict(color='rgba(255,255,255,0)'),
                marker=dict(
                    color="#5A92D8", 
                    size=4,
                    line=dict(width=0)
                ),
                showlegend=False
            ))
            # Ocultando medianas e quartis
            fig.update_traces(whiskerwidth=0)
            
            fig.update_layout(
                yaxis=dict(categoryorder='array', categoryarray=top_global[::-1], tickfont=dict(size=8, color="gray")),
                xaxis=dict(title=dict(text="Impacto na Saída", font=dict(size=8)), tickfont=dict(size=8, color="gray"), zeroline=True, zerolinecolor="gray")
            )
        else:
            fig.add_annotation(text="Sem dados de Features", showarrow=False)

    elif summary_view == "Matrix Dinâmica":
        xai_bulk = load_bulk_xai_metadata(current_dataset)
        if not xai_bulk.empty:
            xai_filtered = xai_bulk[xai_bulk["model_id"].isin(df_filtered["exp_id"])]
            xai_filtered["abs_imp"] = xai_filtered["importance_value"].abs()
            top_3_feats = xai_filtered.groupby("feature_name")["abs_imp"].mean().sort_values(ascending=False).head(3).index.tolist()
            
            if len(top_3_feats) == 3:
                # Pivot para ter features como colunas
                pivot_df = xai_filtered.pivot(index="model_id", columns="feature_name", values="importance_value").reset_index()
                # Merge com df_filtered para pegar a métrica de cor
                merged = pd.merge(pivot_df, df_filtered[["exp_id", sort_metric]], left_on="model_id", right_on="exp_id", how="inner")
                
                f1, f2, f3 = top_3_feats[0], top_3_feats[1], top_3_feats[2]
                fig.add_trace(go.Scatter3d(
                    x=merged[f1],
                    y=merged[f2],
                    z=merged[f3],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=merged[sort_metric],
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    text=merged["exp_id"],
                    hoverinfo="text"
                ))
                fig.update_layout(
                    scene=dict(
                        xaxis_title=dict(text=f1[:10], font=dict(size=8)),
                        yaxis_title=dict(text=f2[:10], font=dict(size=8)),
                        zaxis_title=dict(text=f3[:10], font=dict(size=8)),
                    ),
                    margin=dict(l=0, r=0, b=0, t=0)
                )
            else:
                 fig.add_annotation(text="Não há features suficientes", showarrow=False)
        else:
            fig.add_annotation(text="Sem dados de Features", showarrow=False)

    elif summary_view == "Densidade":
        fig.add_trace(
            go.Histogram(
                x=df_filtered[sort_metric],
                marker_color="#5A92D8",
                opacity=0.75,
                xbins=dict(
                    start=0.0,
                    end=1.0,
                    size=0.05,  # Define o tamanho do bin (escala de agrupamento)
                ),
            )
        )
        if val_selected is not None:
            fig.add_vline(
                x=val_selected, line_width=0.5, line_dash="dot", line_color="red"
            )
        #        best_val = df_filtered[sort_metric].max()
        #        fig.add_vline(x=best_val, line_width=0.5, line_dash="dot", line_color="green")

        fig.update_layout(
            xaxis=dict(
                visible=True,
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=8, color="gray"),
                range=[0.0, 1.0],  # Fixa a escala do Eixo X de 0 a 1
            ),
            yaxis=dict(visible=False),
            bargap=0.1,
        )

    elif summary_view == "Overfitting":
        fig.add_trace(
            go.Scatter(
                x=df_filtered["val_f1_score_macro"],
                y=df_filtered["test_f1_score_macro"],
                mode="markers",
                marker=dict(color="#5A92D8", size=3, opacity=0.6),
                text=df_filtered["exp_id"],
                hoverinfo="text",
            )
        )

        if has_selection:
            fig.add_trace(
                go.Scatter(
                    x=[row_selected["val_f1_score_macro"]],
                    y=[row_selected["test_f1_score_macro"]],
                    mode="markers",
                    marker=dict(color="red", size=5, symbol="star"),
                )
            )

        min_val = min(
            df_filtered["val_f1_score_macro"].min(),
            df_filtered["test_f1_score_macro"].min(),
        )
        max_val = max(
            df_filtered["val_f1_score_macro"].max(),
            df_filtered["test_f1_score_macro"].max(),
        )
        fig.add_shape(
            type="line",
            x0=min_val,
            y0=min_val,
            x1=max_val,
            y1=max_val,
            line=dict(color="gray", dash="dot", width=0.5),
        )

        fig.update_layout(
            xaxis=dict(
                title=dict(text="Val F1", font=dict(size=8)),
                tickfont=dict(size=8, color="gray"),
            ),
            yaxis=dict(
                title=dict(text="Test F1", font=dict(size=8)),
                tickfont=dict(size=8, color="gray"),
            ),
        )

    elif summary_view == "Pareto":
        fig.add_trace(
            go.Scatter(
                x=df_filtered["test_precision_macro"],
                y=df_filtered["test_recall_macro"],
                mode="markers",
                marker=dict(color="#5A92D8", size=3, opacity=0.6),
                text=df_filtered["exp_id"],
                hoverinfo="text",
            )
        )

        if has_selection:
            fig.add_trace(
                go.Scatter(
                    x=[row_selected["test_precision_macro"]],
                    y=[row_selected["test_recall_macro"]],
                    mode="markers",
                    marker=dict(color="red", size=5, symbol="star"),
                )
            )

        fig.update_layout(
            xaxis=dict(
                title=dict(text="Precision", font=dict(size=8)),
                tickfont=dict(size=8, color="gray"),
            ),
            yaxis=dict(
                title=dict(text="Recall", font=dict(size=8)),
                tickfont=dict(size=8, color="gray"),
            ),
        )

    elif summary_view == "KPIs":
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=df_filtered[sort_metric].max(),
                title={"text": "Max", "font": {"size": 12, "color": "gray"}},
                number={"font": {"size": 24, "color": "#5A92D8"}, "valueformat": ".4f"},
                domain={"x": [0, 0.33], "y": [0, 1]},
            )
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=df_filtered[sort_metric].mean(),
                title={"text": "Média", "font": {"size": 12, "color": "gray"}},
                number={"font": {"size": 24, "color": "#5A92D8"}, "valueformat": ".4f"},
                domain={"x": [0.33, 0.66], "y": [0, 1]},
            )
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=df_filtered[sort_metric].min(),
                title={"text": "Min", "font": {"size": 12, "color": "gray"}},
                number={"font": {"size": 24, "color": "#5A92D8"}, "valueformat": ".4f"},
                domain={"x": [0.66, 1], "y": [0, 1]},
            )
        )

    # Se não for uma figura px (que já sobreescreveu o layout), aplicamos o fallback.
    # Como fig_px tem layout diferente, só aplicamos isso se o layout não foi totalmente formatado pelo px.
    if summary_view != "Densidade (Features)":
        fig.update_layout(
            margin=dict(t=5, b=0, l=5, r=5),
            height=270,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )

    st.plotly_chart(fig, use_container_width=True, key=f"summary_hist_hz_{panel_id}")


def render_horizontal_ribbon(df_filtered, sort_metric, current_dataset, panel_id, ribbon_mode="Métricas"):
    """Renderiza a fita de experimentos com SCROLL HORIZONTAL NATIVO."""

    if df_filtered.empty:
        st.warning("Nenhum experimento encontrado com os filtros selecionados.")
        return

    df_sorted = df_filtered.reset_index(drop=True)
    limit = 50
    df_limited = df_sorted.head(limit)

    st.markdown(
        "<div class='horizontal-scroll-marker'></div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(len(df_limited))

    for i, row in df_limited.iterrows():
        with cols[i]:
            is_selected = (st.session_state.get("selected_exp") == row["exp_id"]) and (
                st.session_state.get("selected_dataset") == current_dataset
            )

            if is_selected:
                st.markdown(
                    "<div class='selected-card-marker'></div>", unsafe_allow_html=True
                )

            btn_label = (
                f"id:{row['exp_id']} | {row['model_type']} | {row[sort_metric]:.4f}"
            )
            c1, c2 = st.columns([0.85, 0.15])
            with c1:
                if st.button(
                    btn_label,
                    key=f"btn_hz_{panel_id}_{row['exp_id']}_{i}",
                    use_container_width=True,
                    type="tertiary",
                ):
                    if is_selected:
                        st.session_state["selected_exp"] = None
                        st.session_state["selected_dataset"] = None
                        st.rerun()
                    else:
                        st.session_state["selected_exp"] = row["exp_id"]
                        st.session_state["selected_dataset"] = current_dataset
                        st.rerun()
            with c2:
                if st.button(
                    "**:red[[+]]**",
                    key=f"btn_xray_{panel_id}_{row['exp_id']}_{i}",
                    use_container_width=True,
                    type="tertiary",
                ):
                    st.session_state["selected_exp"] = row["exp_id"]
                    st.session_state["selected_dataset"] = current_dataset
                    st.session_state[f"open_dialog_{panel_id}"] = True
                    st.rerun()

            if ribbon_mode == "Métricas":
                categories = ["V-Acc", "V-F1", "T-Pre", "T-Acc", "T-F1", "T-Rec"]
                valores = [
                    row["val_accuracy"],
                    row["val_f1_score_macro"],
                    row["test_precision_macro"],
                    row["test_accuracy"],
                    row["test_f1_score_macro"],
                    row["test_recall_macro"],
                ]
            else:
                xai_data = load_xai_metadata(row["exp_id"], current_dataset)
                if not xai_data:
                    categories = ["Sem Dados"] * 8
                    valores = [0] * 8
                else:
                    sorted_features = sorted(xai_data.items(), key=lambda item: abs(item[1]), reverse=True)[:8]
                    categories = []
                    seen = set()
                    for f in sorted_features:
                        base_name = f[0][:10].strip()
                        name = base_name
                        count = 1
                        while name in seen:
                            name = f"{base_name[:8]}_{count}"
                            count += 1
                        seen.add(name)
                        categories.append(name)
                        
                    raw_valores = [abs(f[1]) for f in sorted_features]
                    max_val = max(raw_valores) if raw_valores and max(raw_valores) > 0 else 1
                    valores = [v/max_val for v in raw_valores]

            fig = go.Figure(
                data=go.Scatterpolar(
                    r=valores + [valores[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    line=dict(width=1),
                )
            )
            fig.update_layout(
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
                margin=dict(t=10, b=10, l=15, r=15),
                height=230,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"spider_hz_{panel_id}_{row['exp_id']}_{i}",
            )

    if st.session_state.get(f"open_dialog_{panel_id}"):
        st.session_state[f"open_dialog_{panel_id}"] = False
        render_xray(df_filtered, st.session_state["selected_exp"])
