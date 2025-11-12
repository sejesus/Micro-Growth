# 🧪 Micro Growth

Pequeno utilitário desenvolvido para análise de curvas de crescimento microbiano da cadeira de Microbiologia Aplicada.
Escolha das fases, regressão linear simples por mínimos quadrados (caso particular de aplicação de Monod), comparação estatística com Gompertz, Logístico e Richards.
Sílvio do Ó - 2025

Open-source ( GPL-3.0 license )
Ainda muito um W.I.P. Quem quiser pegar e expandir, força!

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-GNU_V3.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow.svg)]()


TODO:
- Painel de introdução directa de dados experimentais e leitura de folhas de cálculo (para não ser só csv)
- Opção de computação da equação de Monod e exponencial de crescimento (Nt = N0 x 2^n) se os dados o permitirem.
- Mais parâmetros de cinética calculáveis...
- Settings visuais (cores dos gráficos, ticks, grids, etc e tal...)

## Estrutura
---------
```text
micro_growth/
├─ main.py                 # GUI PyQt5 (abre CSV, ajusta spans com snap, exporta)
├─ io_data.py              # Leitura CSV + meta + μ_ref CSV + thresholds por coluna
├─ phases.py               # μ(t), segmentação automática, regressão na exponencial
├─ fits.py                 # Ajustes Gompertz/Logístico/Richards (SciPy opcional)
├─ export.py               # Gráficos, Excel (3 folhas), CSV, metadados; figuras combinadas
├─ config/
│   └─ medium_aliases.py   # Dicionário com lista de aliases extensa para meios de cultura
└─ data/
    ├─ growth_refs.py      # Algumas referências de crescimentos típicos com base na literatura
    └─ growth_data.py      # Exemplo de csv com dados experimentais de crescimento de *E. coli* em Luria-Bertani
```


## Funcionalidades
- Leitura de dados em formato CSV (tempo vs absorbância)
- Segmentação automática das fases com base em derivadas e histerese
- Interface gráfica interativa
- Ajustes a modelos Gompertz/Logístico/Richards para confirmar pontos da fase exp 
- Exportação de resultados para Excel e gráficos por curva

## Requisitos
```bash
pip install numpy pandas matplotlib scipy pyqt5
```

## Como usar
1. Colocar os dados num csv`
2. Executar:
```bash
python gui.py
```
3. Abrir csv
4. Selecionar as fases manualmente ou aceitar as sugestões automáticas

## Estrutura do projecto
```
micro_growth/
  ├─ main.py                	# GUI PyQt5 (abre CSV, ajusta spans com snap, exporta)
  ├─ io_data.py            		# Leitura CSV + meta + μ_ref CSV + thresholds por coluna
  ├─ phases.py             		# μ(t), segmentação automática, regressão na exponencial
  ├─ fits.py               		# Ajustes Gompertz/Logístico/Richards (SciPy opcional)
  ├─ export.py             		# Gráficos, Excel (3 folhas), CSV, metadados; figuras combinadas
  └─ config/
       └─ medium_aliases.py		# Dicionário com lista de aliases extensa para meios de cultura
  └─ data/
       └─ growth_refs.py		  # Algumas referências de crescimentos típicos com base na literatura
       └─ growth_data.py		  # Exemplo de csv com dados experimentais de crescimento de Escherichia coli em Luria-Bertani
```

## Licença
Este projeto está licenciado sob os termos da licença GNU General Public License v3.0 .
