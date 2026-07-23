import os
import glob
import json
import yaml
import pandas as pd
import requests
import zipfile
import io


def fetch_cvm():
    print("Iniciando Pipeline ETL CVM Incremental (Delta Load)...")

    # Definição de caminhos
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # --- Dinâmica de Aquisição ---
    config_file = os.path.join(os.path.dirname(__file__), "companhias.yaml")
    
    # Se foi chamado pelo orquestrador, usa a pasta que o orquestrador mandou
    if "AQUISICAO_TARGET_DIR" in os.environ:
        output_dir = os.environ["AQUISICAO_TARGET_DIR"]
    else:
        # Se foi rodado manualmente (F5), salva solto no raw
        output_dir = os.path.join(project_root, "data", "raw")
    output_file = os.path.join(output_dir, "cvm_dfp.parquet")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Leitura do companhias.yaml e mapeamento
    print(f"\n[1/6] Lendo CNPJs completos do {config_file}...")
    try:
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)

        cnpj_to_ticker = {}
        for company in data.get("companies", []):
            ticker = company.get("ticker")
            cnpj = company.get("cnpj")
            if ticker and cnpj:
                # Remove os caracteres especiais do CNPJ para garantir match exato depois
                cnpj_clean = "".join(filter(str.isdigit, cnpj))
                cnpj_to_ticker[cnpj_clean] = ticker

        if not cnpj_to_ticker:
            print("Aviso: Nenhum mapeamento de CNPJ encontrado no companhias.yaml.")
            return

        print(f"      {len(cnpj_to_ticker)} ativos/CNPJs carregados.")

    except FileNotFoundError:
        print(f"Erro: Arquivo {config_file} não encontrado.")
        return
    except Exception as e:
        print(f"Erro ao processar companhias.yaml: {e}")
        return

    # 2. Lógica Incremental
    print("\n[2/6] Verificando histórico existente para carga Delta...")
    anos_alvo = list(range(2010, 2026))
    df_existente = None

    if os.path.exists(output_file):
        try:
            df_existente = pd.read_parquet(output_file)
            if not df_existente.empty and "data_referencia" in df_existente.columns:
                # 1) Verificar se todos os anos estão presentes
                anos_existentes = df_existente["data_referencia"].dt.year.unique()
                anos_faltantes = [
                    ano for ano in anos_alvo if ano not in anos_existentes
                ]
                # FORÇAR reprocessamento do ano corrente e do ano anterior (garantir relatórios recentes)
                current_year = pd.Timestamp.now().year
                if current_year not in anos_faltantes and current_year in anos_alvo:
                    anos_faltantes.append(current_year)
                if (current_year - 1) not in anos_faltantes and (current_year - 1) in anos_alvo:
                    anos_faltantes.append(current_year - 1)

                # 2) Verificar se existem novas empresas no companhias.yaml que não estão na base
                df_existente["cnpj_clean"] = df_existente["CNPJ_CIA"].str.replace(
                    r"\D", "", regex=True
                )
                cnpjs_existentes = set(df_existente["cnpj_clean"].unique())
                cnpjs_novos = set(cnpj_to_ticker.keys()) - cnpjs_existentes

                if cnpjs_novos:
                    print(
                        f"      Detectados {len(cnpjs_novos)} novos CNPJs no companhias.yaml que estão ausentes na base."
                    )
                    # Se houver novas empresas, forçamos o re-processamento de todos os anos alvo
                    anos_faltantes = anos_alvo
                
                anos_faltantes = sorted(list(set(anos_faltantes)))
        except Exception as e:
            print(
                f"Aviso: Erro ao ler o arquivo parquet existente ({e}). Iremos gerar a base inteira novamente."
            )
            anos_faltantes = anos_alvo
            df_existente = None
    else:
        print("      Base CVM não encontrada. Iniciando carga inicial (Full Load).")
        anos_faltantes = anos_alvo

    if not anos_faltantes:
        print(
            "\n=> Base CVM já está atualizada com todos os anos alvo. Encerrando execução sem novos downloads."
        )
        return

    print(f"      Anos agendados para processamento: {anos_faltantes}")

    # 3. Processamento Iterativo
    df_novos_dados = pd.DataFrame()
    contas_desejadas = ["1", "1.01.01", "2.03", "2.01.04", "2.02.01", "3.11", "3.05"]

    for ano in anos_faltantes:
        urls = [
            f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_{ano}.zip",
            f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/itr_cia_aberta_{ano}.zip"
        ]
        print(f"\n[3/6] Processando ano base: {ano}")

        df_ano_list = []
        for url in urls:
            print(f"      Download ZIP: {url}")
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()

                print("      Extraindo arquivos BPA_con, BPP_con e DRE_con...")

                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    for file_name in z.namelist():
                        if any(x in file_name for x in ["BPA_con", "BPP_con", "DRE_con"]):
                            with z.open(file_name) as f:
                                df_temp = pd.read_csv(f, sep=";", encoding="latin1")
                                df_ano_list.append(df_temp)
            except requests.exceptions.HTTPError as e:
                print(
                    f"      Aviso: O arquivo ZIP não foi encontrado no site da CVM ou a conexão falhou ({url})."
                )
            except Exception as e:
                print(f"      Erro inesperado processando a URL {url}: {e}")

        if not df_ano_list:
            print(
                f"      Aviso: Nenhum arquivo de interesse encontrado (nem DFP nem ITR) no ano de {ano}."
            )
            continue

        df_ano_concat = pd.concat(df_ano_list, ignore_index=True)

        # 4. Filtros do Ano
        # Remove pontos, barras e traços do CNPJ da tabela original para match limpo
        df_ano_concat["cnpj_clean"] = df_ano_concat["CNPJ_CIA"].str.replace(
            r"\D", "", regex=True
        )

        # Filtro das empresas mapeadas
        df_ano_concat = df_ano_concat[
            df_ano_concat["cnpj_clean"].isin(cnpj_to_ticker.keys())
        ]

        # Filtro de códigos de contas e apenas exercícios finais ('ÚLTIMO')
        df_ano_concat = df_ano_concat[
            df_ano_concat["CD_CONTA"].isin(contas_desejadas)
        ]
        df_ano_concat = df_ano_concat[df_ano_concat["ORDEM_EXERC"] == "ÚLTIMO"]

        if df_ano_concat.empty:
            print(
                f"      Aviso: Após aplicar filtros de CNPJ e conta contábil, não restaram dados úteis para {ano}."
            )
            continue

        # Mapeamento do ticker
        df_ano_concat["ticker"] = df_ano_concat["cnpj_clean"].map(cnpj_to_ticker)

        # Pivot (Wide Format)
        print("      Formatando tabela Pivot...")
        df_pivot = df_ano_concat.pivot_table(
            index=["CNPJ_CIA", "ticker", "DT_REFER"],
            columns="CD_CONTA",
            values="VL_CONTA",
            aggfunc="first",
        ).reset_index()

        df_pivot.columns.name = None

        # Garante que todas as contas desejadas existam no DF final
        for conta in contas_desejadas:
            if conta not in df_pivot.columns:
                df_pivot[conta] = pd.NA

        df_pivot.rename(columns={"DT_REFER": "data_referencia"}, inplace=True)
        df_pivot["data_referencia"] = pd.to_datetime(df_pivot["data_referencia"])

        df_novos_dados = pd.concat([df_novos_dados, df_pivot], ignore_index=True)
        print(
            f"      Ano {ano} inserido em buffer de atualização com {len(df_pivot)} registros únicos."
        )

    # 5. Concatenação Incremental
    print("\n[5/6] Consolidando Base Incremental...")
    if df_novos_dados.empty:
        print(
            "Nenhum dado novo válido foi capturado durante a execução. O banco não será alterado."
        )
        return

    if df_existente is not None and not df_existente.empty:
        df_final = pd.concat([df_existente, df_novos_dados], ignore_index=True)
        # Drop de possíveis duplicatas caso ocorra erro em rodadas interrompidas
        df_final = df_final.drop_duplicates(
            subset=["CNPJ_CIA", "ticker", "data_referencia"], keep="last"
        )
    else:
        df_final = df_novos_dados

    # 6. Escrita do Parquet
    print(f"\n[6/6] Salvando Parquet final em: {output_file}")
    df_final.to_parquet(output_file, index=False)

    print(
        "\nCarga Incremental finalizada com sucesso! O banco está enriquecido com o histórico requerido."
    )


if __name__ == "__main__":
    fetch_cvm()
