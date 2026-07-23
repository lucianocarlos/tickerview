import streamlit as st
from views.view_ribbon import render_horizontal_ribbon, render_summary_card
from views.view_grid import render_grid_search
from utils.data_loader import load_all_metrics, get_available_datasets
import os

st.set_page_config(layout="wide", page_title="Data Mining Viz")


def get_default_panel(default_dataset="bateria01"):
    return {
        "id": os.urandom(4).hex(),
        "dataset": default_dataset,
        "sort_metric": "test_f1_score_macro",
        "summary_view": "Overfitting",
        "sort_ascending": False,
        "top_n": 50,
        "adv_filters": {},
        "ribbon_mode": "Métricas",
        "filter_best_target": True,
        "filter_best_estimator": False,
    }


def init_session_state(default_dataset="bateria01"):
    if "app_tabs" not in st.session_state:
        st.session_state["app_tabs"] = [
            {
                "id": os.urandom(4).hex(),
                "name": "Cenário 1",
                "panels": [get_default_panel(default_dataset)],
            }
        ]
    if "panels" not in st.session_state:
        st.session_state["panels"] = [get_default_panel(default_dataset)]
    if "open_dialog" not in st.session_state:
        st.session_state["open_dialog"] = False
        st.session_state["selected_exp"] = None
        st.session_state["selected_dataset"] = None


@st.dialog("Filtros Avançados", width="large")
def render_advanced_filters_dialog(panel, df_global):
    st.markdown("### Filtros Principais")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 0.6])

    # Extrair valores únicos
    targets = list(df_global["target_strategy"].dropna().unique())
    splits = list(df_global["split_method"].dropna().unique())
    models = list(df_global["model_type"].dropna().unique())

    with col1:
        sel_targets = st.multiselect(
            "Target Strategy",
            options=targets,
            default=panel["adv_filters"].get("target_strategy", []),
        )
    with col2:
        sel_splits = st.multiselect(
            "Split Method",
            options=splits,
            default=panel["adv_filters"].get("split_method", []),
        )
    with col3:
        sel_models = st.multiselect(
            "Model Type",
            options=models,
            default=panel["adv_filters"].get("model_type", []),
        )
    with col4:
        top_n = st.number_input(
            "Top N (Lista):",
            min_value=1,
            max_value=5000,
            value=panel.get("top_n", 50),
            step=10,
        )

    st.markdown("### Filtros de Métricas (Ranges)")

    def get_metric_range(col_name):
        return float(df_global[col_name].min()), float(df_global[col_name].max())

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        rng_f1_test = st.slider(
            "Test F1 Macro",
            min_value=0.0,
            max_value=1.0,
            value=panel["adv_filters"].get("test_f1_score_macro", (0.0, 1.0)),
            step=0.01,
        )
    with m2:
        rng_f1_val = st.slider(
            "Val F1 Macro",
            min_value=0.0,
            max_value=1.0,
            value=panel["adv_filters"].get("val_f1_score_macro", (0.0, 1.0)),
            step=0.01,
        )
    with m3:
        rng_acc_test = st.slider(
            "Test Accuracy",
            min_value=0.0,
            max_value=1.0,
            value=panel["adv_filters"].get("test_accuracy", (0.0, 1.0)),
            step=0.01,
        )
    with m4:
        rng_acc_val = st.slider(
            "Val Accuracy",
            min_value=0.0,
            max_value=1.0,
            value=panel["adv_filters"].get("val_accuracy", (0.0, 1.0)),
            step=0.01,
        )

    m5, m6, m7 = st.columns(3)
    with m5:
        rng_prec_test = st.slider(
            "Test Precision",
            min_value=0.0,
            max_value=1.0,
            value=panel["adv_filters"].get("test_precision_macro", (0.0, 1.0)),
            step=0.01,
        )
    with m6:
        rng_rec_test = st.slider(
            "Test Recall",
            min_value=0.0,
            max_value=1.0,
            value=panel["adv_filters"].get("test_recall_macro", (0.0, 1.0)),
            step=0.01,
        )
    with m7:
        rng_f1_w_test = st.slider(
            "Test F1 Weighted",
            min_value=0.0,
            max_value=1.0,
            value=panel["adv_filters"].get("test_f1_score_weighted", (0.0, 1.0)),
            step=0.01,
        )

    st.markdown("---")
    st.markdown("### Filtros Dinâmicos de Arquitetura e Hiperparâmetros")

    dyn_cols = st.columns(3)

    # Extrair todas as chaves de hyperparâmetros e configs
    param_keys = {}
    for _, row in df_global.iterrows():
        d = row["parameters_dict"]
        m_type = row["model_type"]
        if isinstance(d, dict):
            for k in d.keys():
                if k not in param_keys:
                    param_keys[k] = set()
                param_keys[k].add(str(m_type))

    config_keys = set()
    for d in df_global["experiment_config_dict"].dropna():
        if "preprocessing" in d and isinstance(d["preprocessing"], dict):
            for k in d["preprocessing"].keys():
                config_keys.add(f"preprocessing.{k}")

    sel_dynamic = {}
    col_idx = 0

    # Render parameters filters
    for key in sorted(list(param_keys.keys())):
        # Extrair valores unicos para esta chave
        unique_vals = set()
        for d in df_global["parameters_dict"].dropna():
            if key in d:
                unique_vals.add(str(d[key]))

        if len(unique_vals) > 0:
            with dyn_cols[col_idx % 3]:
                models_context = ", ".join(sorted(list(param_keys[key])))
                label = f"Param: {key} ({models_context})"
                sel_dynamic[f"param_{key}"] = st.multiselect(
                    label,
                    options=sorted(list(unique_vals)),
                    default=panel["adv_filters"].get(f"param_{key}", []),
                )
            col_idx += 1

    # Render config filters
    for key in sorted(list(config_keys)):
        sub_key = key.split(".")[1]
        unique_vals = set()
        for d in df_global["experiment_config_dict"].dropna():
            if "preprocessing" in d and sub_key in d["preprocessing"]:
                val = d["preprocessing"][sub_key]
                if isinstance(val, list):
                    for v in val:
                        unique_vals.add(str(v))
                else:
                    unique_vals.add(str(val))

        if len(unique_vals) > 0:
            with dyn_cols[col_idx % 3]:
                sel_dynamic[f"config_{key}"] = st.multiselect(
                    f"Config: {key}",
                    options=sorted(list(unique_vals)),
                    default=panel["adv_filters"].get(f"config_{key}", []),
                )
            col_idx += 1

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Limpar Filtros", width="stretch"):
            panel["adv_filters"] = {}
            st.rerun()
    with col_btn2:
        if st.button("Aplicar Filtros", width="stretch", type="primary"):
            panel["top_n"] = top_n
            panel["adv_filters"] = {
                "target_strategy": sel_targets,
                "split_method": sel_splits,
                "model_type": sel_models,
                "test_f1_score_macro": rng_f1_test,
                "val_f1_score_macro": rng_f1_val,
                "test_accuracy": rng_acc_test,
                "val_accuracy": rng_acc_val,
                "test_precision_macro": rng_prec_test,
                "test_recall_macro": rng_rec_test,
                "test_f1_score_weighted": rng_f1_w_test,
            }
            for k, v in sel_dynamic.items():
                panel["adv_filters"][k] = v
            st.rerun()


@st.fragment
def render_comparative_panel(idx, panel, available_datasets, metric_labels):
    panel_id = panel["id"]
    df_global = load_all_metrics()

    if df_global.empty:
        st.error("A bateria está vazia ou não foi encontrada.")
        return

    # Sincroniza o estado do dataset a partir do session_state (antes da execução do st.radio)
    ds_key = f"ds_{panel_id}"
    if ds_key in st.session_state:
        panel["dataset"] = st.session_state[ds_key]

    # Filtra o df global para o dataset atual do painel ANTES de extrair as listas únicas
    current_dataset = panel.get(
        "dataset", available_datasets[0] if available_datasets else None
    )
    df_panel_filtered = (
        df_global[df_global["dataset_version"] == current_dataset]
        if current_dataset
        else df_global
    )

    # Colunas principais do Layout de Painel (o primeiro valor de columns ajusta a largura do gráfico golbalview)
    col_left, col_right = st.columns([1.8, 8.2])

    with col_left:
        col_menu, col_summary = st.columns([0.45, 1.85])

        with col_menu:
            with st.popover("📑", width="stretch"):
                if st.button(
                    "➕ Novo Cenário", key=f"add_tab_{panel_id}", width="stretch"
                ):
                    new_tab_id = os.urandom(4).hex()
                    new_panel = get_default_panel(
                        panel.get(
                            "dataset",
                            available_datasets[0] if available_datasets else None,
                        )
                    )
                    st.session_state["app_tabs"].append(
                        {
                            "id": new_tab_id,
                            "name": f"Cenário {len(st.session_state['app_tabs']) + 1}",
                            "panels": [new_panel],
                        }
                    )
                    st.rerun()

                if st.button(
                    "👥 Clonar Cenário", key=f"clone_tab_{panel_id}", width="stretch"
                ):
                    import copy

                    for t in st.session_state["app_tabs"]:
                        if any(p["id"] == panel_id for p in t["panels"]):
                            new_tab = copy.deepcopy(t)
                            new_tab["id"] = os.urandom(4).hex()
                            new_tab["name"] = (
                                f"Cenário {len(st.session_state['app_tabs']) + 1}"
                            )
                            for p in new_tab["panels"]:
                                p["id"] = os.urandom(4).hex()
                            st.session_state["app_tabs"].append(new_tab)
                            st.rerun()
                            break

                if len(st.session_state["app_tabs"]) > 1:
                    if st.button(
                        "❌ Remover Cenário",
                        key=f"del_tab_menu_{panel_id}",
                        width="stretch",
                    ):
                        for t in st.session_state["app_tabs"]:
                            if any(p["id"] == panel_id for p in t["panels"]):
                                st.session_state["app_tabs"].remove(t)
                                st.rerun()

            st.markdown(
                "<div style='margin-bottom: 5px;'></div>", unsafe_allow_html=True
            )

            with st.popover("🗂️", width="stretch"):
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

            with st.popover("↕️", width="stretch"):
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
                        width="stretch",
                        type=b_type,
                    ):
                        if is_selected:
                            panel["sort_ascending"] = not panel["sort_ascending"]
                        else:
                            panel["sort_metric"] = metric_key
                            panel["sort_ascending"] = False
                        st.rerun()

            if st.button(
                "⚙️",
                help="Filtros Avançados",
                key=f"btn_filter_{panel_id}",
                width="stretch",
            ):
                render_advanced_filters_dialog(panel, df_panel_filtered)

            with st.popover("🏆", width="stretch"):
                st.markdown("**Campeões:**")

                # Determinando o estado atual para o Radio (exclusivo)
                current_val = "Nenhum"
                if panel.get("filter_best_target"):
                    current_val = "Melhor por Target"
                elif panel.get("filter_best_estimator"):
                    current_val = "Melhor por Estimador"

                escolha = st.radio(
                    "Filtro de Campeões",
                    ["Nenhum", "Melhor por Target", "Melhor por Estimador"],
                    index=["Nenhum", "Melhor por Target", "Melhor por Estimador"].index(
                        current_val
                    ),
                    key=f"radio_champ_{panel_id}",
                    label_visibility="collapsed",
                )

                panel["filter_best_target"] = escolha == "Melhor por Target"
                panel["filter_best_estimator"] = escolha == "Melhor por Estimador"

            if "ribbon_mode" not in panel:
                panel["ribbon_mode"] = "Métricas"

            with st.popover("🎯", width="stretch"):
                st.markdown("**Modo da Fita:**")
                panel["ribbon_mode"] = st.radio(
                    "Modo",
                    ["Métricas", "Features"],
                    index=0 if panel["ribbon_mode"] == "Métricas" else 1,
                    key=f"mode_{panel_id}",
                    label_visibility="collapsed",
                )

            with st.popover("📊", width="stretch"):
                st.markdown("**Visão Global:**")
                if panel["ribbon_mode"] == "Métricas":
                    view_opts = [
                        "Densidade",
                        "Overfitting",
                        "PxR",
                        "Num",
                    ]
                else:
                    view_opts = [
                        "Densidade (Features)",
                        "SHAP",
                        "Matrix Dinâmica",
                    ]

                # Certifica que a view selecionada é válida para o modo
                if panel.get("summary_view") not in view_opts:
                    panel["summary_view"] = view_opts[0]

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
                    width="stretch",
                ):
                    st.session_state["panels"].remove(panel)
                    st.rerun()

        # Filtra o DataFrame Global pelo Dataset escolhido no painel
        df_filtered = df_global[df_global["dataset_version"] == panel["dataset"]].copy()

        # Aplica os filtros avançados
        adv = panel.get("adv_filters", {})

        if adv.get("target_strategy"):
            df_filtered = df_filtered[
                df_filtered["target_strategy"].isin(adv["target_strategy"])
            ]
        if adv.get("split_method"):
            df_filtered = df_filtered[
                df_filtered["split_method"].isin(adv["split_method"])
            ]
        if adv.get("model_type"):
            df_filtered = df_filtered[df_filtered["model_type"].isin(adv["model_type"])]

        # Applica ranges dinamicos das metricas
        metrics = [
            "test_f1_score_macro",
            "val_f1_score_macro",
            "test_accuracy",
            "val_accuracy",
            "test_precision_macro",
            "test_recall_macro",
            "test_f1_score_weighted",
        ]
        for m in metrics:
            if m in adv:
                m_min, m_max = adv[m]
                df_filtered = df_filtered[
                    (df_filtered[m] >= m_min) & (df_filtered[m] <= m_max)
                ]

        # Filtros Dinâmicos de Hiperparâmetros
        for k, v in adv.items():
            if df_filtered.empty:
                break
            if k.startswith("param_") and v:
                param_key = k.replace("param_", "")
                df_filtered = df_filtered[
                    df_filtered["parameters_dict"].apply(
                        lambda d: str(d.get(param_key)) in v
                    )
                ]
            if k.startswith("config_") and v:
                sub_key = k.replace("config_preprocessing.", "")

                def check_config(d):
                    if "preprocessing" in d and sub_key in d["preprocessing"]:
                        val = d["preprocessing"][sub_key]
                        if isinstance(val, list):
                            return any(str(x) in v for x in val)
                        return str(val) in v
                    return False

                df_filtered = df_filtered[
                    df_filtered["experiment_config_dict"].apply(check_config)
                ]

        # Filtros de Campeões
        if panel.get("filter_best_target"):
            if not df_filtered.empty:
                asc = panel.get("sort_ascending", False)
                df_filtered = df_filtered.sort_values(
                    by=panel["sort_metric"], ascending=asc
                )
                df_filtered = df_filtered.drop_duplicates(
                    subset=["target_strategy"], keep="first"
                )

        if panel.get("filter_best_estimator"):
            if not df_filtered.empty:
                asc = panel.get("sort_ascending", False)
                df_filtered = df_filtered.sort_values(
                    by=panel["sort_metric"], ascending=asc
                )
                df_filtered = df_filtered.drop_duplicates(
                    subset=["model_type"], keep="first"
                )

        asc = panel.get("sort_ascending", False)
        df_filtered = df_filtered.sort_values(by=panel["sort_metric"], ascending=asc)

        df_filtered_full = df_filtered.copy()

        top_n = panel.get("top_n", 50)
        df_filtered = df_filtered.head(top_n)

        with col_summary:
            render_summary_card(
                df_filtered,
                st.session_state.get("selected_exp", None),
                panel["sort_metric"],
                panel_id,
                panel["summary_view"],
                panel["ribbon_mode"],
                panel["dataset"],
                df_filtered_full,
            )

    with col_right:
        # Filtra para passar os dados corretos e renderiza a fita de radar separada verticalmente
        df_filtered_right = df_filtered.copy()

        # Sempre renderiza a fita horizontal no painel direito
        render_horizontal_ribbon(
            df_filtered_right,
            panel["sort_metric"],
            panel["dataset"],
            panel_id,
            panel["ribbon_mode"],
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
            gap: 0rem !important;
        }
        div[data-testid="column"] [data-testid="stVerticalBlock"] {
            gap: 0rem !important;
        }
        div[data-testid="stHorizontalBlock"] [data-testid="stVerticalBlock"] {
            gap: 0rem !important;
        }
        div[data-testid="stElementContainer"] {
            margin-bottom: 0rem !important;
            padding-bottom: 0rem !important;
        }
        div[data-testid="stPopover"] {
            margin-top: 0rem !important;
            margin-bottom: 0rem !important;
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
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

        /* Forçando a barra pesada contra o p, span e divs do Streamlit */
       
       div[class*="st-key-btn_hz_"] button p  {
         font-size: 8pt !important;
         line-height: 1.2 !important;
         margin: 0 !important;
         padding: 0 !important;
         white-space: pre-wrap !important;
         text-align: center !important;
         font-weight: bold !important;
         border: none !important;
         background: transparent !important;
         color: black !important;
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

    tabs_data = st.session_state["app_tabs"]
    tab_names = [t["name"] for t in tabs_data]
    tabs_ui = st.tabs(tab_names)

    for t_idx, tab_ctx in enumerate(tabs_ui):
        with tab_ctx:
            tab_data = tabs_data[t_idx]

            for p_idx, panel in enumerate(tab_data["panels"]):
                render_comparative_panel(
                    p_idx, panel, available_datasets, metric_labels
                )

            # Adicionar Painel Comparativo (apenas no final dos painéis)
            if len(tab_data["panels"]) < 5:
                if st.button(
                    "➕ Adicionar Painel Comparativo",
                    key=f"add_panel_{tab_data['id']}",
                    type="tertiary",
                ):
                    import copy

                    base_panel = tab_data["panels"][0]
                    new_panel = get_default_panel(available_datasets[0])

                    # Copiar as configurações de exibição e filtros do primeiro painel
                    new_panel["dataset"] = base_panel["dataset"]
                    new_panel["sort_metric"] = base_panel["sort_metric"]
                    new_panel["summary_view"] = base_panel["summary_view"]
                    new_panel["sort_ascending"] = base_panel["sort_ascending"]
                    new_panel["top_n"] = base_panel.get("top_n", 50)
                    new_panel["adv_filters"] = copy.deepcopy(base_panel["adv_filters"])
                    new_panel["ribbon_mode"] = base_panel.get("ribbon_mode", "Métricas")
                    new_panel["filter_best_target"] = base_panel.get(
                        "filter_best_target", False
                    )
                    new_panel["filter_best_estimator"] = base_panel.get(
                        "filter_best_estimator", False
                    )

                    tab_data["panels"].append(new_panel)
                    st.rerun()

    # Abaixo, a Batalha do Grid Search
    if df_report_main is not None:
        render_grid_search(df_report_main)
    # render_cluster(df_mestre) # Desativado temporariamente

    st.iframe(
        """
    <script>
    if (window.parent.ribbonInterval) {
        clearInterval(window.parent.ribbonInterval);
    }
    window.parent.ribbonInterval = setInterval(() => {
        const doc = window.parent.document;
        // Procura os marcadores invisíveis que injetamos logo antes da fita
        const markers = doc.querySelectorAll('.horizontal-scroll-marker');
        
        markers.forEach(marker => {
            // Sobe até o container do Streamlit que guarda o marcador
            let container = marker.closest('[data-testid="stElementContainer"]');
            if (container) {
                let b = container.nextElementSibling;
                // Procura o próximo stHorizontalBlock (pula possíveis elementos ocultos vazios do Streamlit)
                while (b && b.getAttribute('data-testid') !== 'stHorizontalBlock') {
                    b = b.nextElementSibling;
                }
                
                if (b && b.getAttribute('data-testid') === 'stHorizontalBlock') {
                    
                    b.style.setProperty("overflow-x", "auto", "important");
                    b.style.setProperty("flex-wrap", "nowrap", "important");
                    b.style.setProperty("padding-top", "6px", "important");
                    b.style.setProperty("padding-bottom", "5px", "important");
                    
                    if(b.children.length > 0) {
                        b.children[0].style.removeProperty("position");
                        b.children[0].style.removeProperty("left");
                        b.children[0].style.removeProperty("z-index");
                        b.children[0].style.removeProperty("background-color");
                    }
                    
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
                            
                            if(!col.classList.contains("scrolled-to")) {
                                const containerRect = b.getBoundingClientRect();
                                const colRect = col.getBoundingClientRect();
                                const scrollAmount = b.scrollLeft + (colRect.left - containerRect.left) - (containerRect.width / 2) + (colRect.width / 2);
                                b.scrollTo({left: scrollAmount, behavior: 'smooth'});
                                col.classList.add("scrolled-to");
                            }
                        } else {
                            col.style.removeProperty("border");
                            col.style.removeProperty("border-radius");
                            col.style.removeProperty("background-color");
                            col.style.removeProperty("padding");
                            col.classList.remove("scrolled-to");
                        }
                    }
                }
            }
        });
    }, 500);
    </script>
    """,
        height=1,
        width=1,
    )


if __name__ == "__main__":
    main()
