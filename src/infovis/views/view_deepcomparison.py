import streamlit as st
from utils.data_loader import load_all_metrics, load_xai_metadata


def compare_dicts(d1, d2, indent=0):
    """Retorna duas listas de strings (uma para cada coluna) renderizando a comparação do dicionário. Laranja para dif."""
    keys = sorted(list(set(d1.keys()) | set(d2.keys())))
    lines1, lines2 = [], []
    spaces = "&nbsp;" * (indent * 4)
    for k in keys:
        v1, v2 = d1.get(k), d2.get(k)
        if isinstance(v1, dict) or isinstance(v2, dict):
            v1_dict = v1 if isinstance(v1, dict) else {}
            v2_dict = v2 if isinstance(v2, dict) else {}
            l1, l2 = compare_dicts(v1_dict, v2_dict, indent + 1)
            lines1.append(f"{spaces}**{k}**: {{")
            lines2.append(f"{spaces}**{k}**: {{")
            lines1.extend(l1)
            lines2.extend(l2)
            lines1.append(f"{spaces}}}")
            lines2.append(f"{spaces}}}")
        else:
            str1 = str(v1) if v1 is not None else "--"
            str2 = str(v2) if v2 is not None else "--"
            if str1 != str2:
                lines1.append(f"{spaces}**{k}**: :orange[{str1}]")
                lines2.append(f"{spaces}**{k}**: :orange[{str2}]")
            else:
                lines1.append(f"{spaces}**{k}**: {str1}")
                lines2.append(f"{spaces}**{k}**: {str2}")
    return lines1, lines2


@st.dialog("Comparação Profunda de Modelos", width="large")
def render_deepcomparison(base_id, base_dataset, compare_id, compare_dataset):
    df_global = load_all_metrics()
    if df_global.empty:
        st.error("Datalake vazio.")
        return

    df_base = df_global[
        (df_global["exp_id"] == base_id)
        & (df_global["dataset_version"] == base_dataset)
    ]
    df_comp = df_global[
        (df_global["exp_id"] == compare_id)
        & (df_global["dataset_version"] == compare_dataset)
    ]

    if df_base.empty or df_comp.empty:
        st.error("Um dos experimentos não foi encontrado no datalake para comparação.")
        return

    row_base = df_base.iloc[0]
    row_comp = df_comp.iloc[0]

    # ------------------ CABEÇALHO ------------------
    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"#### **A: Base (id:{base_id})**")
        st.markdown(f"**{row_base['model_type']}** | Bateria: {base_dataset}")
        st.markdown(f"**Target:** {row_base['target_strategy']}")
    with colB:
        st.markdown(f"#### **B: Alvo (id:{compare_id})**")
        st.markdown(f"**{row_comp['model_type']}** | Bateria: {compare_dataset}")
        st.markdown(f"**Target:** {row_comp['target_strategy']}")

    st.divider()

    # ------------------ MÉTRICAS ------------------
    st.markdown("#### Métricas ( A ➔ B)")
    metrics = {
        "Acurácia (Val)": "val_accuracy",
        "F1 (Val)": "val_f1_score_macro",
        "Acurácia (Teste)": "test_accuracy",
        "F1 (Teste)": "test_f1_score_macro",
        "Precision (Teste)": "test_precision_macro",
        "Recall (Teste)": "test_recall_macro",
    }

    m_cols = st.columns(len(metrics))
    for i, (label, col_name) in enumerate(metrics.items()):
        valA = row_base.get(col_name, 0)
        valB = row_comp.get(col_name, 0)

        diff = valB - valA

        color = "blue" if diff >= 0 else "red"
        arrow = "↑" if diff >= 0 else "↓"
        diff_str = f"**:{color}[{arrow} {abs(diff):.4f}]**"

        with m_cols[i]:
            st.caption(label)
            st.markdown(f"**{valA:.4f}** ➔ {valB:.4f}")
            st.markdown(diff_str)

    st.divider()

    # ------------------ CONFIGURAÇÕES E HIPERPARÂMETROS ------------------
    st.markdown("#### Configurações (Laranja = Diferente)")

    configA = row_base.get("experiment_config_dict", {})
    if not isinstance(configA, dict):
        configA = {}
    paramsA = row_base.get("parameters_dict", {})
    if not isinstance(paramsA, dict):
        paramsA = {}

    configB = row_comp.get("experiment_config_dict", {})
    if not isinstance(configB, dict):
        configB = {}
    paramsB = row_comp.get("parameters_dict", {})
    if not isinstance(paramsB, dict):
        paramsB = {}

    mergedA = {**configA, **paramsA}
    mergedB = {**configB, **paramsB}

    # Extrai o grid_search_config para colocar no final
    gs_A = mergedA.pop("grid_search_config", {})
    gs_B = mergedB.pop("grid_search_config", {})

    l1, l2 = compare_dicts(mergedA, mergedB)

    cA, cB = st.columns(2)
    with cA:
        st.markdown(f"**A: Base (id:{base_id})**")
        st.markdown("<br>", unsafe_allow_html=True)
        if l1:
            st.markdown("<br>".join(l1), unsafe_allow_html=True)
        else:
            st.markdown("*Nenhuma configuração.*")
    with cB:
        st.markdown(f"**B: Alvo (id:{compare_id})**")
        st.markdown("<br>", unsafe_allow_html=True)
        if l2:
            st.markdown("<br>".join(l2), unsafe_allow_html=True)
        else:
            st.markdown("*Nenhuma configuração.*")

    st.divider()

    # ------------------ IMPACTO DE FEATURES ------------------
    st.markdown("####  Top Features (Mudança de Rank de A para B)")

    xai_base = load_xai_metadata(base_id, base_dataset)
    xai_comp = load_xai_metadata(compare_id, compare_dataset)

    if xai_base and xai_comp:
        # Obter top 20 para ter uma boa base de comparação
        sorted_A = sorted(xai_base.items(), key=lambda item: abs(item[1]), reverse=True)
        sorted_B = sorted(xai_comp.items(), key=lambda item: abs(item[1]), reverse=True)

        rank_A = {f[0]: i + 1 for i, f in enumerate(sorted_A)}
        rank_B = {f[0]: i + 1 for i, f in enumerate(sorted_B)}

        # Vamos listar as top 15 de B e mostrar como elas estavam em A
        top_B_15 = sorted_B[:15]

        feature_lines = []
        for feat, imp in top_B_15:
            posB = rank_B[feat]
            posA = rank_A.get(feat, None)

            if posA is None:
                # Feature não existe no top (ou no XAI todo) do A
                status = "**:blue[NEW]**"
            else:
                diff_rank = posA - posB  # PosA=10, PosB=2 -> subiu 8 posições (+8)
                if diff_rank > 0:
                    status = f"**:blue[↑ +{diff_rank}]**"
                elif diff_rank < 0:
                    status = f"**:red[↓ {diff_rank}]**"
                else:
                    status = "**:gray[=]**"

            feature_lines.append(
                f"**{posB}º** | {feat} | Imp: {abs(imp):.4f} | {status}"
            )

        cb1, cb2 = st.columns(2)
        with cb1:
            st.markdown("**Top 15 Features do Base (A)**")
            for i, (feat, imp) in enumerate(sorted_A[:15]):
                st.markdown(f"{i + 1}º | {feat} | Imp: {abs(imp):.4f}")
        with cb2:
            st.markdown("**Top 15 Features do Alvo (B) e Evolução**")
            for line in feature_lines:
                st.markdown(line)
    else:
        st.markdown(
            "*Dados de explicabilidade (XAI) indisponíveis para um dos modelos.*"
        )

    st.divider()

    # ------------------ GRID SEARCH CONFIG (EXPANDER NO FINAL) ------------------
    with st.expander("Configuração do Grid Search"):
        # Se algum deles não for dicionário (ex: nulo), converte para dict vazio
        gs_A = gs_A if isinstance(gs_A, dict) else {}
        gs_B = gs_B if isinstance(gs_B, dict) else {}

        gl1, gl2 = compare_dicts(gs_A, gs_B)

        gA, gB = st.columns(2)
        with gA:
            if gl1:
                st.markdown("<br>".join(gl1), unsafe_allow_html=True)
            else:
                st.markdown("*Nenhuma configuração de Grid Search.*")
        with gB:
            if gl2:
                st.markdown("<br>".join(gl2), unsafe_allow_html=True)
            else:
                st.markdown("*Nenhuma configuração de Grid Search.*")
