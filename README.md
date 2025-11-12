
Micro Growth
=========================================
Pequeno utilitário desenvolvido para análise de curvas de crescimento microbiano da cadeira de Microbiologia Aplicada.
Escolha das fases, regressão linear simples por mínimos quadrados (caso particular de aplicação de Monod), comparação estatística com Gompertz, Logístico e Richards.
Sílvio do Ó - 2025
=========================================
Open-source ( GPL-3.0 license )
Ainda muito um W.I.P. Quem quiser pegar e expandir, força!
=========================================
TODO:
- Painel de introdução directa de dados experimentais e leitura de folhas de cálculo (para não ser só csv)
- Opção de computação da equação de Monod e exponencial de crescimento (Nt = N0 x 2^n) se os dados o permitirem.
- Mais parâmetros de cinética calculáveis...
- Settings visuais (cores dos gráficos, ticks, grids, etc e tal...)
=========================================

Estrutura
---------
micro_growth/
  ├─ gui.py                		# GUI PyQt5 (abre CSV, ajusta spans com snap, exporta)
  ├─ io_data.py            		# Leitura CSV + meta + μ_ref CSV + thresholds por coluna
  ├─ phases.py             		# μ(t), segmentação automática, regressão na exponencial
  ├─ fits.py               		# Ajustes Gompertz/Logístico/Richards (SciPy opcional)
  ├─ export.py             		# Gráficos, Excel (3 folhas), CSV, metadados; figuras combinadas
  └─ config/
       └─ medium_aliases.py		# Dicionário com lista de aliases extensa para meios de cultura
  └─ data/
       └─ growth_refs.py		# Algumas referências de crescimentos típicos com base na literatura
       └─ growth_data.py		# Exemplo de csv com dados experimentais de crescimento de Escherichia coli em Luria-Bertani


Funcionalidades principais
-----------------
- Painel **Settings** na GUI para editar espécie/estirpe/meio/temperatura,
  factores de fase, μ_ref manual/CSV, e *tunables* (R² alvo, janela de suavização, etc.).
- **Snap-to-point** ao selecionar intervalos (SpanSelector encaixa no ponto experimental mais próximo).
- **Linearização** com **tabela de resultados** (R², μ, t_d, etc.).
- **Figura combinada** por coluna: **μ(t) suavizado** (topo) + **modelos empíricos**
  (meio) + **tabela** com R² e **desvio relativo** face ao R² da regressão linear (base).
  Desvio relativo = (R²_modelo − R²_linear) / R²_linear.

Como usar
---------
1) Instalar dependências:
   pip install PyQt5 matplotlib numpy pandas xlsxwriter scipy

2) Correr:
   python main.py

3) Abrir CSV (wide; 1ª coluna = tempo [min]). Pode incluir linhas "Meta_Medium" e "Meta_Temp".

4) Ajustar spans por coluna (teclas 1–4 para escolher fase). O arrasto "agarra" ao ponto real.

5) Exportar: outputs/<timestamp>/ com PNGs, Excel, CSV de janelas e metadados.


