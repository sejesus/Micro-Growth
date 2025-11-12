# 🧪 Análise de Crescimento Microbiano

<p align="center">
  <img src="https://i.ibb.co/27dQM4bX/Sem-t-tulo.png" alt="Preview" width="600"/>
</p>


[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-GNU_V3.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow.svg)]()

## 📌 Descrição
Ferramenta modular para análise de curvas de crescimento microbiano com segmentação automática ou manual das fases **Lag**, **Exponencial**, **Estacionária** e **Declínio**. Inclui regressão linear, ajustes paramétricos (Gompertz, Logístico, Richards) e exportação para Excel.

## 🚀 Funcionalidades
- Leitura de dados em formato CSV (tempo vs absorbância)
- Segmentação automática das fases com base em derivadas e histerese
- Interface gráfica interativa (Tkinter ou PyQt5)
- Ajustes empíricos com SciPy (opcional)
- Exportação de resultados para Excel e gráficos por curva

## 📦 Requisitos
```bash
pip install numpy pandas matplotlib scipy pyqt5
```

## 🖥️ Como usar
1. Coloque os dados em `data/growth_data.csv`
2. Execute:
```bash
python gui.py
```
3. Selecione as fases manualmente ou aceite as sugestões automáticas

## 📂 Estrutura do projeto
```
├── gui.py                # Interface Tkinter para seleção de fases
├── novo_gui.py           # Interface PyQt5 alternativa
├── AnaliseCurvasCrescimento.py # Lógica principal de análise
├── data/                 # Dados de entrada
├── outputs/              # Resultados e gráficos
└── README.md
```

## 📜 Licença
Este projeto está licenciado sob os termos da licença MIT.
