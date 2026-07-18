import sqlite3
import pandas as pd

conn = sqlite3.connect('data/datalake/bateria02/datalake.db')

try:
    df_runs = pd.read_sql("""
        SELECT m.model_name, m.id as run_id, count(fi.feature_name) as non_zero_features
        FROM feature_importances fi
        JOIN models m ON fi.model_id = m.id
        WHERE fi.importance_value > 0
        GROUP BY m.model_name, m.id
    """, conn)
    
    print("\nMédia de features com importância maior que zero (por modelo treinado):")
    print(df_runs.groupby('model_name')['non_zero_features'].mean().round(1))
    
    # max, min, median
    print("\nEstatísticas das features não-zero por modelo:")
    print(df_runs.groupby('model_name')['non_zero_features'].describe().round(1))
except Exception as e:
    print("Erro:", e)
