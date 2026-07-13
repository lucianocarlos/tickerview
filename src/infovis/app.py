import streamlit as st
from views.view_ribbon import render_horizontal_ribbon, render_summary_card
from views.view_grid import render_grid_search
from views.view_xray import render_xray
from views.view_cluster import render_cluster
from utils.data_loader import load_all_metrics, get_available_datasets, load_mestre_dataset
import pandas as pd
import json
import os

st.set_page_config(layout='wide', page_title='Data Mining Viz')

def get_default_panel():
    return {
        'id': os.urandom(4).hex(),
        'dataset': 'dataset001',
        'sort_metric': 'test_f1_score_macro',
        'target_filter': 'Target Strategy',
        'split_filter': 'Split Method',
        'model_filter': 'Model Type',
        'summary_view': 'Boxplot Modelos',
        'sort_ascending': False
    }

def init_session_state():
    if 'panels' not in st.session_state:
        st.session_state['panels'] = [get_default_panel()]
    if 'open_dialog' not in st.session_state:
        st.session_state['open_dialog'] = False
        st.session_state['selected_exp'] = None
        st.session_state['selected_dataset'] = None

@st.fragment
def render_comparative_panel(idx, panel, available_datasets, metric_labels):
    panel_id = panel['id']
    df_global = load_all_metrics()
    
    if df_global.empty:
        st.error("O Datalake está vazio ou não foi encontrado.")
        return
        
    targets = ["Target Strategy"] + list(df_global['target_strategy'].unique())
    splits = ["Split Method"] + list(df_global['split_method'].unique())
    models = ["Model Type"] + list(df_global['model_type'].unique())
    
    # Colunas principais do Layout de Painel
    col_menu, col_space, col_summary, col_fita = st.columns([0.4, 0.2, 1.4, 8.0])
    
    with col_menu:
        st.markdown(f"<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)
        with st.popover("🗂️", use_container_width=True):
            st.markdown("**Selecione o Dataset:**")
            panel['dataset'] = st.radio("Dataset", available_datasets, index=available_datasets.index(panel['dataset']) if panel['dataset'] in available_datasets else 0, key=f"ds_{panel_id}", label_visibility="collapsed")
        with st.popover("↕️", use_container_width=True):
            st.markdown("**Critério de Ordenação:**")
            if 'sort_ascending' not in panel:
                panel['sort_ascending'] = False
            
            for metric_key, metric_label in metric_labels.items():
                is_selected = (panel['sort_metric'] == metric_key)
                
                if is_selected:
                    arrow = " 🔼" if panel['sort_ascending'] else " 🔽"
                    label = f"{metric_label}{arrow}"
                    b_type = "primary"
                else:
                    label = metric_label
                    b_type = "secondary"
                    
                if st.button(label, key=f"btn_sort_{panel_id}_{metric_key}", use_container_width=True, type=b_type):
                    if is_selected:
                        panel['sort_ascending'] = not panel['sort_ascending']
                    else:
                        panel['sort_metric'] = metric_key
                        panel['sort_ascending'] = False
        with st.popover("🎯", use_container_width=True):
            st.markdown("**Filtrar Target Strategy:**")
            panel['target_filter'] = st.radio("Target", targets, index=targets.index(panel['target_filter']) if panel['target_filter'] in targets else 0, key=f"target_{panel_id}", label_visibility="collapsed")
        with st.popover("✂️", use_container_width=True):
            st.markdown("**Filtrar Split Method:**")
            panel['split_filter'] = st.radio("Split", splits, index=splits.index(panel['split_filter']) if panel['split_filter'] in splits else 0, key=f"split_{panel_id}", label_visibility="collapsed")
        with st.popover("🧠", use_container_width=True):
            st.markdown("**Filtrar Model Type:**")
            panel['model_filter'] = st.radio("Model", models, index=models.index(panel['model_filter']) if panel['model_filter'] in models else 0, key=f"model_{panel_id}", label_visibility="collapsed")
        with st.popover("📊", use_container_width=True):
            st.markdown("**Visão Global:**")
            view_opts = ["Densidade", "Overfitting", "Boxplot Modelos", "Pareto", "KPIs"]
            panel['summary_view'] = st.radio("Global View", view_opts, index=view_opts.index(panel['summary_view']) if panel.get('summary_view') in view_opts else 0, key=f"view_{panel_id}", label_visibility="collapsed")
        
        if idx > 0:
            if st.button("🗑️", key=f"del_{panel_id}", help="Remover painel"):
                st.session_state['panels'].remove(panel)
                st.rerun()

    # Filtra o DataFrame Global pelo Dataset escolhido no painel
    df_filtered = df_global[df_global['dataset_version'] == panel['dataset']].copy()
    
    if panel['target_filter'] != "Target Strategy":
        df_filtered = df_filtered[df_filtered['target_strategy'] == panel['target_filter']]
    if panel['split_filter'] != "Split Method":
        df_filtered = df_filtered[df_filtered['split_method'] == panel['split_filter']]
    
    asc = panel.get('sort_ascending', False)
    df_filtered = df_filtered.sort_values(by=panel['sort_metric'], ascending=asc)
    
    if panel['model_filter'] != "Model Type":
        df_filtered = df_filtered[df_filtered['model_type'] == panel['model_filter']]
        
    with col_summary:
        render_summary_card(df_filtered, st.session_state.get('selected_exp', None), panel['sort_metric'], panel_id, panel['summary_view'])
    with col_fita:
        render_horizontal_ribbon(df_filtered, panel['sort_metric'], panel['dataset'], panel_id)
        
    st.markdown("---")

def main():
    st.markdown('''
        <style>
        .block-container {
            padding-top: 3rem;
            padding-bottom: 0rem;
            max-width: 95%;
        }
        div[data-testid="column"] {
            gap: 0.1rem !important;
        }
        div[data-testid="column"] [data-testid="stVerticalBlock"] {
            gap: 0.1rem !important;
        }
        hr {
            margin-top: 0px !important;
            margin-bottom: 10px !important;
        }
        div[data-testid="stHorizontalBlock"] {
            gap: 0.5rem !important;
        }
        
        [data-testid="stSelectbox"] > div > div {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            font-size: 0.9em;
            cursor: pointer;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            min-height: 1rem !important;
        }
        [data-testid="stSelectbox"] label { display: none; }
        div[data-testid="stSelectbox"] > label { display: none !important; }
        
        /* Estilos do Menu Vertical Popover para parecer um Toolbar limpo */
        div[data-testid="stPopover"] > button {
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
            color: gray;
            font-size: 1.5em;
            width: 100%;
        }
        div[data-testid="stPopover"] > button:hover {
            color: #5A92D8 !important;
            border-color: transparent !important;
            background-color: transparent !important;
        }
        div[data-testid="stPopover"] > button:focus {
            color: #5A92D8 !important;
            border-color: transparent !important;
            background-color: transparent !important;
            box-shadow: none !important;
        }
        </style>
    ''', unsafe_allow_html=True)
    
    init_session_state()
    
    # df_mestre = load_mestre_dataset('dataset001') # Desativado temporariamente
    
    metric_labels = {
        'val_f1_score_macro': 'Val F1 Macro',
        'test_f1_score_macro': 'Test F1 Macro',
        'val_accuracy': 'Val Acc',
        'test_accuracy': 'Test Acc',
        'test_f1_score_weighted': 'Test F1 Weighted',
        'test_precision_macro': 'Test Precision'
    }

    df_report_main = None
    available_datasets = get_available_datasets()
    
    if not available_datasets:
        st.warning("Nenhum dado encontrado no Datalake (SQLite).")
        return
        
    for idx, panel in enumerate(st.session_state['panels']):
        render_comparative_panel(idx, panel, available_datasets, metric_labels)
        
    # Adicionar Painel Comparativo (apenas no final dos painéis e se não tivermos muitos para não quebrar a máquina)
    if len(st.session_state['panels']) < 5:
        if st.button("➕ Adicionar Painel Comparativo", type="tertiary"):
            st.session_state['panels'].append(get_default_panel())
            st.rerun()

    # Abaixo, a Batalha do Grid Search
    if df_report_main is not None:
        render_grid_search(df_report_main)
    # render_cluster(df_mestre) # Desativado temporariamente

if __name__ == "__main__":
    main()
