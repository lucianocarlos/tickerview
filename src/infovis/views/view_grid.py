import streamlit as st
import plotly.express as px
import pandas as pd

@st.fragment
def render_grid_search(df_report, sort_metric):
    st.subheader("Análise de Hiperparâmetros (Grid Search)")
    
    if df_report.empty:
        st.warning("Nenhum experimento encontrado com os filtros selecionados.")
        return
        
    # Flatten parameters into separate columns
    params_df = df_report["parameters_dict"].apply(pd.Series)
    
    # Filter out columns that have only 1 unique value (constants) or are completely NaN
    varying_params = []
    for col in params_df.columns:
        if params_df[col].nunique(dropna=True) > 1:
            varying_params.append(col)
            
    if not varying_params:
        st.info("Não há variação de hiperparâmetros neste recorte (todos os modelos possuem os mesmos parâmetros exatos). Altere seus filtros para comparar mais rodadas.")
        return

    # Combine with the target metric
    plot_df = params_df[varying_params].copy()
    
    # Convert all varying parameters to string so parallel_categories handles them gracefully
    for col in plot_df.columns:
        plot_df[col] = plot_df[col].astype(str)
        
    plot_df[sort_metric] = df_report[sort_metric]
    
    # Render Parallel Categories chart
    fig = px.parallel_categories(
        plot_df,
        dimensions=varying_params,
        color=sort_metric,
        color_continuous_scale=px.colors.sequential.Inferno,
        title=f"Impacto dos Hiperparâmetros no {sort_metric}"
    )
    
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)
