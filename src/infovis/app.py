import streamlit as st
from views.view_ribbon import render_horizontal_ribbon, render_summary_card
from views.view_grid import render_grid_search
from utils.data_loader import load_all_metrics, get_available_datasets
import streamlit.components.v1 as components
import os

st.set_page_config(layout="wide", page_title="Data Mining Viz")


def get_default_panel(default_dataset="bateria01"):
    return {
        "id": os.urandom(4).hex(),
        "dataset": default_dataset,
        "sort_metric": "test_f1_score_macro",
        "target_filter": "Target Strategy",
        "split_filter": "Split Method",
        "model_filter": "Model Type",
        "summary_view": "Overfitting",
        "sort_ascending": False,
    }


def init_session_state(default_dataset="bateria01"):
    if "panels" not in st.session_state:
        st.session_state["panels"] = [get_default_panel(default_dataset)]
    if "open_dialog" not in st.session_state:
        st.session_state["open_dialog"] = False
        st.session_state["selected_exp"] = None
        st.session_state["selected_dataset"] = None


@st.fragment
def render_comparative_panel(idx, panel, available_datasets, metric_labels):
    panel_id = panel["id"]
    df_global = load_all_metrics()

    if df_global.empty:
        st.error("O Datalake está vazio ou não foi encontrado.")
        return

    targets = ["Target Strategy"] + list(df_global["target_strategy"].unique())
    splits = ["Split Method"] + list(df_global["split_method"].unique())
    models = ["Model Type"] + list(df_global["model_type"].unique())

    # Colunas principais do Layout de Painel (De-duplicando colunas horizontais para isolar a fita)
    col_left, col_right = st.columns([2.3, 7.7])

    with col_left:
        col_menu, col_summary = st.columns([0.8, 1.5])

        with col_menu:
            st.markdown(
                "<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True
            )
            col_m1, col_m2 = st.columns(2)

            with col_m1:
                with st.popover("🗂️", use_container_width=True):
                    st.markdown("**Selecione a Bateria:**")
                    panel["dataset"] = st.radio(
                        "Dataset",
                        available_datasets,
                        index=available_datasets.index(panel["dataset"])
                        if panel["dataset"] in available_datasets
                        else 0,
                        key=f"ds_{panel_id}",
                        label_visibility="collapsed",
                    )

                with st.popover("🎯", use_container_width=True):
                    st.markdown("**Filtrar Target Strategy:**")
                    panel["target_filter"] = st.radio(
                        "Target",
                        targets,
                        index=targets.index(panel["target_filter"])
                        if panel["target_filter"] in targets
                        else 0,
                        key=f"target_{panel_id}",
                        label_visibility="collapsed",
                    )

                with st.popover("🧠", use_container_width=True):
                    st.markdown("**Filtrar Model Type:**")
                    panel["model_filter"] = st.radio(
                        "Model",
                        models,
                        index=models.index(panel["model_filter"])
                        if panel["model_filter"] in models
                        else 0,
                        key=f"model_{panel_id}",
                        label_visibility="collapsed",
                    )

            with col_m2:
                with st.popover("↕️", use_container_width=True):
                    st.markdown("**Critério de Ordenação:**")
                    if "sort_ascending" not in panel:
                        panel["sort_ascending"] = False

                    for metric_key, metric_label in metric_labels.items():
                        is_selected = panel["sort_metric"] == metric_key

                        if is_selected:
                            arrow = " 🔼" if panel["sort_ascending"] else " 🔽"
                            label = f"{metric_label}{arrow}"
                            b_type = "primary"
                        else:
                            label = metric_label
                            b_type = "secondary"

                        if st.button(
                            label,
                            key=f"btn_sort_{panel_id}_{metric_key}",
                            use_container_width=True,
                            type=b_type,
                        ):
                            if is_selected:
                                panel["sort_ascending"] = not panel["sort_ascending"]
                            else:
                                panel["sort_metric"] = metric_key
                                panel["sort_ascending"] = False
                            st.rerun()

                with st.popover("✂️", use_container_width=True):
                    st.markdown("**Filtrar Split Method:**")
                    panel["split_filter"] = st.radio(
                        "Split",
                        splits,
                        index=splits.index(panel["split_filter"])
                        if panel["split_filter"] in splits
                        else 0,
                        key=f"split_{panel_id}",
                        label_visibility="collapsed",
                    )

                with st.popover("📊", use_container_width=True):
                    st.markdown("**Visão Global:**")
                    view_opts = ["Densidade", "Overfitting", "Pareto", "KPIs"]
                    panel["summary_view"] = st.radio(
                        "Global View",
                        view_opts,
                        index=view_opts.index(panel["summary_view"])
                        if panel.get("summary_view") in view_opts
                        else 0,
                        key=f"view_{panel_id}",
                        label_visibility="collapsed",
                    )

            if idx > 0:
                st.markdown(
                    "<div style='margin-top: 10px;'></div>", unsafe_allow_html=True
                )
                if st.button(
                    "🗑️",
                    key=f"del_{panel_id}",
                    help="Remover painel",
                    use_container_width=True,
                ):
                    st.session_state["panels"].remove(panel)
                    st.rerun()

        # Filtra o DataFrame Global pelo Dataset escolhido no painel
        df_filtered = df_global[df_global["dataset_version"] == panel["dataset"]].copy()

        if panel["target_filter"] != "Target Strategy":
            df_filtered = df_filtered[
                df_filtered["target_strategy"] == panel["target_filter"]
            ]
        if panel["split_filter"] != "Split Method":
            df_filtered = df_filtered[
                df_filtered["split_method"] == panel["split_filter"]
            ]

        asc = panel.get("sort_ascending", False)
        df_filtered = df_filtered.sort_values(by=panel["sort_metric"], ascending=asc)

        if panel["model_filter"] != "Model Type":
            df_filtered = df_filtered[
                df_filtered["model_type"] == panel["model_filter"]
            ]

        with col_summary:
            render_summary_card(
                df_filtered,
                st.session_state.get("selected_exp", None),
                panel["sort_metric"],
                panel_id,
                panel["summary_view"],
            )

    with col_right:
        # Filtra para passar os dados corretos e renderiza a fita de radar separada verticalmente
        df_filtered_right = df_global[
            df_global["dataset_version"] == panel["dataset"]
        ].copy()
        if panel["target_filter"] != "Target Strategy":
            df_filtered_right = df_filtered_right[
                df_filtered_right["target_strategy"] == panel["target_filter"]
            ]
        if panel["split_filter"] != "Split Method":
            df_filtered_right = df_filtered_right[
                df_filtered_right["split_method"] == panel["split_filter"]
            ]
        asc = panel.get("sort_ascending", False)
        df_filtered_right = df_filtered_right.sort_values(
            by=panel["sort_metric"], ascending=asc
        )
        if panel["model_filter"] != "Model Type":
            df_filtered_right = df_filtered_right[
                df_filtered_right["model_type"] == panel["model_filter"]
            ]

        render_horizontal_ribbon(
            df_filtered_right, panel["sort_metric"], panel["dataset"], panel_id
        )

    st.markdown("---")


def main():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 3rem;
            padding-bottom: 0rem;
            padding-left: 5px !important;
            padding-right: 5px !important;
            max-width: 99% !important;
        }
        div[data-testid="column"] {
            gap: 0.1rem !important;
        }
        div[data-testid="column"] [data-testid="stVerticalBlock"] {
            gap: 0.1rem !important;
        }
        div[data-testid="stHorizontalBlock"] [data-testid="stVerticalBlock"] {
            gap: 0rem !important;
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
        
        /* Scroll horizontal na fita de experimentos */
        div[data-testid="stElementContainer"]:has(.horizontal-scroll-marker) + div[data-testid="stHorizontalBlock"] {
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
            padding-top: 6px !important;
            padding-bottom: 5px !important;
            align-items: flex-start !important;
            margin-top: 0px !important; /* Reset child margin-top */
        }
        
        /* Desloca a coluna inteira da fita para cima para alinhar com o topo do resumo */
        div[data-testid="stHorizontalBlock"]:has(.horizontal-scroll-marker) > div[data-testid="stColumn"]:nth-child(2) {
            margin-top: -10px !important;
        }
        
        /* Oculta o contêiner vazio do marcador para eliminar o gap/espaço vertical */
        div[data-testid="stElementContainer"]:has(.horizontal-scroll-marker) {
            display: none !important;
        }
        
        /* Ajustes de compactação dos cards na fita */
        div[data-testid="stElementContainer"]:has(.horizontal-scroll-marker) + div[data-testid="stHorizontalBlock"] [data-testid="stColumn"] div.stButton {
            margin-top: -15px !important;
            margin-bottom: -1px !important;
        }
        div[data-testid="stElementContainer"]:has(.horizontal-scroll-marker) + div[data-testid="stHorizontalBlock"] [data-testid="stColumn"] div.stButton > button {
            height: 24px !important;
            min-height: 24px !important;
            padding: 0 !important;
            font-size: 0.85em !important;
            font-weight: bold !important;
            border: none !important;
            background: transparent !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    available_datasets = get_available_datasets()

    if not available_datasets:
        st.warning("Nenhum dado encontrado no Datalake (SQLite).")
        return

    init_session_state(available_datasets[0])

    # df_mestre = load_mestre_dataset('dataset001') # Desativado temporariamente

    metric_labels = {
        "val_f1_score_macro": "Val F1 Macro",
        "test_f1_score_macro": "Test F1 Macro",
        "val_accuracy": "Val Acc",
        "test_accuracy": "Test Acc",
        "test_f1_score_weighted": "Test F1 Weighted",
        "test_precision_macro": "Test Precision",
        "test_recall_macro": "Test Recall",
    }

    df_report_main = None

    for idx, panel in enumerate(st.session_state["panels"]):
        render_comparative_panel(idx, panel, available_datasets, metric_labels)

    # Adicionar Painel Comparativo (apenas no final dos painéis e se não tivermos muitos para não quebrar a máquina)
    if len(st.session_state["panels"]) < 5:
        if st.button("➕ Adicionar Painel Comparativo", type="tertiary"):
            st.session_state["panels"].append(get_default_panel(available_datasets[0]))
            st.rerun()

    # Abaixo, a Batalha do Grid Search
    if df_report_main is not None:
        render_grid_search(df_report_main)
    # render_cluster(df_mestre) # Desativado temporariamente

    components.html(
        """
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
                b.style.setProperty("padding-top", "6px", "important");
                b.style.setProperty("padding-bottom", "5px", "important");
                
                b.children[0].style.removeProperty("position");
                b.children[0].style.removeProperty("left");
                b.children[0].style.removeProperty("z-index");
                b.children[0].style.removeProperty("background-color");
                
                for(let i=0; i<b.children.length; i++) {
                    let col = b.children[i];
                    col.style.setProperty("min-width", "320px", "important");
                    col.style.setProperty("width", "320px", "important");
                    col.style.setProperty("max-width", "320px", "important");
                    col.style.setProperty("flex", "0 0 320px", "important");
                    
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
    """,
        height=0,
        width=0,
    )


if __name__ == "__main__":
    main()
