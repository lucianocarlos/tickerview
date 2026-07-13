import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
from views.view_xray import render_xray

def render_summary_card(df_filtered, selected_exp, sort_metric, panel_id, summary_view):
    st.markdown(f"<div style='font-size: 0.9em; font-weight: bold;'>{summary_view}</div>", unsafe_allow_html=True)
    
    if df_filtered.empty:
        st.warning("Sem dados.")
        return

    val_selected = None
    row_selected = None
    has_selection = False
    if selected_exp is not None and selected_exp in df_filtered['exp_id'].values:
        row_selected = df_filtered[df_filtered['exp_id'] == selected_exp].iloc[0]
        val_selected = row_selected[sort_metric]
        has_selection = True
        
    fig = go.Figure()

    if summary_view == "Densidade":
        fig.add_trace(go.Histogram(
            x=df_filtered[sort_metric],
            marker_color='#5A92D8',
            opacity=0.75
        ))
        if val_selected is not None:
            fig.add_vline(x=val_selected, line_width=2, line_dash="dash", line_color="red")
        
        best_val = df_filtered[sort_metric].max()
        fig.add_vline(x=best_val, line_width=2, line_dash="dot", line_color="gold")
            
        fig.update_layout(
            xaxis=dict(visible=True, showgrid=False, zeroline=False, tickfont=dict(size=8, color="gray")),
            yaxis=dict(visible=False),
            bargap=0.1
        )
        
    elif summary_view == "Overfitting":
        fig.add_trace(go.Scatter(
            x=df_filtered['val_f1_score_macro'],
            y=df_filtered['test_f1_score_macro'],
            mode='markers',
            marker=dict(color='#5A92D8', size=5, opacity=0.6),
            text=df_filtered['exp_id'],
            hoverinfo='text'
        ))
        
        if has_selection:
            fig.add_trace(go.Scatter(
                x=[row_selected['val_f1_score_macro']],
                y=[row_selected['test_f1_score_macro']],
                mode='markers',
                marker=dict(color='red', size=8, symbol='star')
            ))
            
        min_val = min(df_filtered['val_f1_score_macro'].min(), df_filtered['test_f1_score_macro'].min())
        max_val = max(df_filtered['val_f1_score_macro'].max(), df_filtered['test_f1_score_macro'].max())
        fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="gray", dash="dash"))
        
        fig.update_layout(
            xaxis=dict(title=dict(text="Val F1", font=dict(size=8)), tickfont=dict(size=8, color="gray")),
            yaxis=dict(title=dict(text="Test F1", font=dict(size=8)), tickfont=dict(size=8, color="gray"))
        )

    elif summary_view == "Boxplot Modelos":
        for model in df_filtered['model_type'].unique():
            df_model = df_filtered[df_filtered['model_type'] == model]
            fig.add_trace(go.Box(
                x=df_model[sort_metric],
                name=model,
                marker_color='#5A92D8'
            ))
            
        if val_selected is not None:
            fig.add_vline(x=val_selected, line_width=2, line_dash="dash", line_color="red")
            
        fig.update_layout(
            xaxis=dict(visible=True, tickfont=dict(size=8, color="gray")),
            yaxis=dict(visible=True, tickfont=dict(size=8, color="gray"))
        )

    elif summary_view == "Pareto":
        fig.add_trace(go.Scatter(
            x=df_filtered['test_precision_macro'],
            y=df_filtered['test_recall_macro'],
            mode='markers',
            marker=dict(color='#5A92D8', size=5, opacity=0.6),
            text=df_filtered['exp_id'],
            hoverinfo='text'
        ))
        
        if has_selection:
            fig.add_trace(go.Scatter(
                x=[row_selected['test_precision_macro']],
                y=[row_selected['test_recall_macro']],
                mode='markers',
                marker=dict(color='red', size=8, symbol='star')
            ))
            
        fig.update_layout(
            xaxis=dict(title=dict(text="Precision", font=dict(size=8)), tickfont=dict(size=8, color="gray")),
            yaxis=dict(title=dict(text="Recall", font=dict(size=8)), tickfont=dict(size=8, color="gray"))
        )
        
    elif summary_view == "KPIs":
        fig.add_trace(go.Indicator(
            mode = "number",
            value = df_filtered[sort_metric].max(),
            title = {"text": "Max", "font": {"size": 12, "color": "gray"}},
            number = {"font": {"size": 24, "color": "#5A92D8"}, "valueformat": ".4f"},
            domain = {'x': [0, 0.33], 'y': [0, 1]}
        ))
        fig.add_trace(go.Indicator(
            mode = "number",
            value = df_filtered[sort_metric].mean(),
            title = {"text": "Média", "font": {"size": 12, "color": "gray"}},
            number = {"font": {"size": 24, "color": "#5A92D8"}, "valueformat": ".4f"},
            domain = {'x': [0.33, 0.66], 'y': [0, 1]}
        ))
        fig.add_trace(go.Indicator(
            mode = "number",
            value = df_filtered[sort_metric].min(),
            title = {"text": "Min", "font": {"size": 12, "color": "gray"}},
            number = {"font": {"size": 24, "color": "#5A92D8"}, "valueformat": ".4f"},
            domain = {'x': [0.66, 1], 'y': [0, 1]}
        ))

    # Common layout overrides
    fig.update_layout(
        margin=dict(t=5, b=0, l=5, r=5), 
        height=230, 
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"summary_hist_hz_{panel_id}")

def render_horizontal_ribbon(df_filtered, sort_metric, current_dataset, panel_id):
    """Renderiza a fita de experimentos com SCROLL HORIZONTAL NATIVO."""
    
    df_sorted = df_filtered.reset_index(drop=True)
    limit = 50
    df_limited = df_sorted.head(limit)
    
    st.markdown("""
        <style>
        div[data-testid="element-container"]:has(.horizontal-scroll-marker) + div[data-testid="stHorizontalBlock"] {
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
            padding-bottom: 15px !important;
        }
        </style>
        <div class="horizontal-scroll-marker"></div>
    """, unsafe_allow_html=True)
    
    components.html("""
    <script>
    if (window.parent.ribbonInterval) {
        clearInterval(window.parent.ribbonInterval);
    }
    window.parent.ribbonInterval = setInterval(() => {
        const doc = window.parent.document;
        const blocks = doc.querySelectorAll('[data-testid="stHorizontalBlock"]');
        blocks.forEach(b => {
            if(b.children.length > 10) { 
                b.style.setProperty("overflow-x", "auto", "important");
                b.style.setProperty("flex-wrap", "nowrap", "important");
                b.style.setProperty("padding-bottom", "15px", "important");
                
                b.children[0].style.removeProperty("position");
                b.children[0].style.removeProperty("left");
                b.children[0].style.removeProperty("z-index");
                b.children[0].style.removeProperty("background-color");
                
                for(let i=0; i<b.children.length; i++) {
                    let col = b.children[i];
                    col.style.setProperty("min-width", "220px", "important");
                    col.style.setProperty("width", "220px", "important");
                    col.style.setProperty("max-width", "220px", "important");
                    col.style.setProperty("flex", "0 0 220px", "important");
                    
                    if(col.querySelector('.selected-card-marker')) {
                        col.style.setProperty("border", "2px solid #5A92D8", "important");
                        col.style.setProperty("border-radius", "10px", "important");
                        col.style.setProperty("background-color", "rgba(90, 146, 216, 0.05)", "important");
                        col.style.setProperty("padding", "5px", "important");
                    } else {
                        col.style.removeProperty("border");
                        col.style.removeProperty("border-radius");
                        col.style.removeProperty("background-color");
                        col.style.removeProperty("padding");
                    }
                }
            }
        });
    }, 500);
    </script>
    """, height=0, width=0)
    
    cols = st.columns(len(df_limited))
    
    for i, row in df_limited.iterrows():
        with cols[i]:
            is_selected = (st.session_state.get('selected_exp') == row['exp_id']) and (st.session_state.get('selected_dataset') == current_dataset)
            
            if is_selected:
                st.markdown("<div class='selected-card-marker'></div>", unsafe_allow_html=True)
            
            if st.button(f"{row['exp_id']}", key=f"btn_hz_{panel_id}_{row['exp_id']}_{i}", use_container_width=True, type="tertiary"):
                if is_selected:
                    st.session_state['selected_exp'] = None
                    st.session_state['selected_dataset'] = None
                    st.rerun(scope="fragment")
                else:
                    st.session_state['selected_exp'] = row['exp_id']
                    st.session_state['selected_dataset'] = current_dataset
                    st.session_state[f'open_dialog_{panel_id}'] = True
                    st.rerun(scope="fragment")
                
            st.markdown(f"<div style='font-size: 0.8em; color: gray; text-align: center; margin-top: -10px; margin-bottom: 5px;'>{row['model_type']} | {row[sort_metric]:.4f}</div>", unsafe_allow_html=True)
            
            categories = ['V-Acc', 'V-F1', 'T-Acc', 'T-F1', 'T-Pre', 'T-Rec']
            valores = [
                row['val_accuracy'], row['val_f1_score_macro'], 
                row['test_accuracy'], row['test_f1_score_macro'], 
                row['test_precision_macro'], row['test_recall_macro']
            ]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=valores + [valores[0]],
                theta=categories + [categories[0]],
                fill='toself',
                line=dict(width=1)
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=6, color="rgba(100,100,100,0.5)"), nticks=4, angle=0),
                    angularaxis=dict(showticklabels=True, tickfont=dict(size=8, color="gray"), ticks='', direction="clockwise")
                ),
                showlegend=False,
                margin=dict(t=20, b=0, l=20, r=20), 
                height=190,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"spider_hz_{panel_id}_{row['exp_id']}_{i}")

    if st.session_state.get(f'open_dialog_{panel_id}'):
        st.session_state[f'open_dialog_{panel_id}'] = False
        render_xray(df_filtered, st.session_state['selected_exp'])
