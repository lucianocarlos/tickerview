import os
import sys
import glob
import json
from datetime import datetime
import time
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def rodar_script(script_path, root_dir):
    """Executa um script de extração"""
    print(f"-> [INICIANDO] {script_path.name}")
    resultado = subprocess.run([sys.executable, str(script_path)], cwd=root_dir)
    if resultado.returncode != 0:
        raise RuntimeError(f"O script {script_path.name} falhou com código {resultado.returncode}!")
    return script_path.name

def criar_nova_aquisicao(root_dir):
    raw_dir = root_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    aquisicoes = glob.glob(os.path.join(raw_dir, "aquisicao_*"))
    if not aquisicoes:
        next_id = 1
    else:
        ids = []
        for d in aquisicoes:
            basename = os.path.basename(d)
            try:
                ids.append(int(basename.replace("aquisicao_", "")))
            except ValueError:
                pass
        next_id = max(ids) + 1 if ids else 1
        
    nova_pasta = f"aquisicao_{next_id:03d}"
    novo_caminho = raw_dir / nova_pasta
    novo_caminho.mkdir(exist_ok=True)
    
    # Criar json
    info_path = novo_caminho / "info_aquisicao.json"
    info = {
        "aquisicao_id": nova_pasta,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notas": ""
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
        
    return nova_pasta

if __name__ == "__main__":
    current_dir = Path(__file__).parent.resolve()
    root_dir = current_dir.parent.parent
    
    print("=== INICIANDO NOVA AQUISIÇÃO DE DADOS ===")
    nova_aquisicao = criar_nova_aquisicao(root_dir)
    # Exporta a variável de ambiente para que os subprocessos saibam onde gravar
    os.environ["AQUISICAO_TARGET_DIR"] = str(root_dir / "data" / "raw" / nova_aquisicao)
    print(f"\n[OK] Nova pasta criada e configurada: {nova_aquisicao}")
    print("\nIniciando Crawlers em Paralelo...")
    
    scripts_aquisicao = [
        current_dir / "extracao_cvm.py",
        current_dir / "extracao_fundamentos.py",
        current_dir / "extracao_opcoes.py",
        current_dir / "extracao_precos.py",
    ]
    
    # Filtra apenas os que existem
    scripts_validos = [s for s in scripts_aquisicao if s.exists()]
    
    start_time = time.time()
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futuros = [
                executor.submit(rodar_script, path, root_dir) for path in scripts_validos
            ]
            for futuro in as_completed(futuros):
                script_concluido = futuro.result()
                print(f"<- [CONCLUÍDO] {script_concluido}")
                
        end_time = time.time()
        duration_sec = round(end_time - start_time, 2)
        
        # Atualiza o json
        info_path = root_dir / "data" / "raw" / nova_aquisicao / "info_aquisicao.json"
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            info["tempo_execucao_segundos"] = duration_sec
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=4, ensure_ascii=False)
                
        print(f"\n=== AQUISIÇÃO {nova_aquisicao} FINALIZADA EM {duration_sec}s ===")
    except Exception as e:
        print(f"\n[ERRO FATAL] A aquisição falhou: {e}")
        sys.exit(1)
