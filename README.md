# TickerView - MLOps 2.0 Dashboard

O **TickerView** é um painel visual (dashboard) interativo para análise de modelos de Machine Learning (Classificação de Sinais de Ações), avaliando performance, métricas e interpretabilidade através da arquitetura MLOps 2.0.

### 🌐 Acesso Online (Recomendado)
A maneira mais fácil e rápida de interagir com o painel visual é através da nuvem. O aplicativo está publicado e pode ser acessado em qualquer navegador:
👉 **[Acessar o TickerView no Streamlit Cloud](https://lucianocarlos-ufu-ifgoiano-tickerview.streamlit.app/)**

---

## 💻 Como rodar o projeto localmente

Caso deseje executar a aplicação na sua própria máquina (para inspecionar o código fonte ou realizar testes locais), siga o passo a passo detalhado abaixo.

### 1. Pré-requisitos
Certifique-se de ter os seguintes softwares instalados em sua máquina:
- **Git** (para clonar o repositório)
- **Python 3.9+** 

### 2. Clonando o Repositório
Abra o seu terminal (Prompt de Comando, PowerShell ou Terminal do Linux/Mac) e execute o comando abaixo para baixar o código:
```bash
git clone https://github.com/lucianocarlos/tickerview.git
cd tickerview
```

### 3. Criando um Ambiente Virtual (VENV)
É uma boa prática criar um ambiente virtual Python para isolar as bibliotecas do projeto.
**No Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```
**No Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Instalando as Dependências
Com o ambiente virtual ativado, instale todas as bibliotecas necessárias executando:
```bash
pip install -r requirements.txt
```

### 5. Executando o Dashboard (Streamlit)
Após o término da instalação, você pode iniciar o servidor local da aplicação com o comando:
```bash
streamlit run src/infovis/app.py
```

O terminal exibirá um endereço local (geralmente `http://localhost:8501`). O seu navegador padrão deverá abri-lo automaticamente com a interface gráfica do projeto!

### 📂 Sobre os Dados (Datalake)
O banco de dados SQLite (`datalake.db`), que contém os metadados dos experimentos, os dicionários de hiperparâmetros, os relatórios de classificação (F1-score, Precision, Accuracy) e as importâncias (*Feature Importances* XAI), **já está embutido neste repositório**. 
Nenhum download ou processamento de dados adicional é necessário. O sistema já está pré-configurado para carregar todo o acervo analítico ao abrir.
