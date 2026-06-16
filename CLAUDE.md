# CLAUDE.md

Guia para o Claude Code trabalhar neste repositório. As duas tarefas centrais são:
1. **Escrever/revisar um artigo em LaTeX (Overleaf)** sobre os resultados deste projeto.
2. **Modificar os scripts** que geram os resultados (tabelas, gráficos, mapas) explorados no artigo.

## Visão geral do projeto

Pesquisa de Iniciação Científica sobre **previsão de séries temporais de preços de
alimentos no Brasil** (dados mensais da CONAB, por UF/estado e por produto). O objetivo
é comparar modelos estatísticos clássicos e de machine/deep learning em validação
*walk-forward* e reportar o erro (RMSE/MAPE) por horizonte de previsão, por estado e por
produto.

**Modelos comparados** (registrados em `MODEL_FUNCS`, [forecast_methods.py](Prediction_Methods/forecast_methods.py)):
- Estatísticos: `ARIMA` (pmdarima auto_arima), `ETS` (Holt-Winters), `Prophet`
- ML/DL: `Random Forest`, `LSTM`, `GRU`, `Transformer`, `Informer` (ProbSparse, Zhou et al. 2021)

## Estrutura

| Pasta / arquivo | Conteúdo |
|---|---|
| `Databases/*.xlsx` | Bases CONAB. A atual é `DatabaseConabv5.xlsx` — **uma aba por produto**, colunas = UFs, linhas = datas mensais (`index_col=0`). |
| `Data_Pre_Processing/` | Scripts de limpeza/montagem das planilhas (interpolação, remoção de zeros, padronização, pivotagem). |
| `Prediction_Methods/` | Núcleo de previsão e avaliação (ver abaixo). |
| `Charts/`, `Tables/` | Geração de figuras (boxplot, mapas, decomposição sazonal, painéis) e estatísticas descritivas para o artigo. |
| `maps_output/`, `stats_output/` | Saídas geradas (HTML/PNG/CSV/XLSX) — artefatos, não fonte. |

### `Prediction_Methods/` — módulos principais

- `main.py` — ponto de entrada. **Configuração por constantes no topo** (`EXCEL_FILE_PATH`, `SHEET_NAME`/produto, `UF`, `TEST_PERIODS`, `MODEL_TO_RUN`). Despacha entre análises (boxplots, excel, tabela de erro) ou previsão de um modelo único.
- `forecast_methods.py` — todas as funções de previsão; assinatura uniforme `(series, n_periods) -> (pd.Series, model)`. `MODEL_FUNCS` é o registro central usado por cache/análise. Resolução de device (`resolve_torch_device`) e limites de CPU via env `ML_DEVICE` / `ML_MAX_CPU_CORES`.
- `models.py` — arquiteturas PyTorch (LSTM, GRU, Transformer, Informer + ProbSparse).
- `features.py` — engenharia de atributos (lags, rolling) e `series_to_sequences` para modelos sequenciais.
- `training.py` — `train_batch` (loop de treino com early stopping) para os modelos torch.
- `metrics.py` — `calculate_rmse`, `calculate_mape`.
- `analysis.py` — validação walk-forward: `sliding_rmse_boxplots`, `sliding_rmse_excel`, `error_comparison_table`, `sliding_error_chart`.
- `cache.py` — `_worker` / `_compute_window_errors`, cache em disco por janela (MD5). **Devem permanecer top-level** (picklable para `ProcessPoolExecutor`).
- `run_all_products.py` — roda `sliding_rmse_excel` para todas as abas/produtos e consolida em uma planilha formatada.
- `charts.py`, `radar.py`, `step_rmse.py`, `map_brazil_best_model.py` — figuras específicas do artigo.

## Como rodar

Python **3.12**, venv em `.venv/` (já com dependências de `requirements.txt`).

```powershell
# A partir de Prediction_Methods/ (os caminhos das bases são relativos: ../Databases/...)
python main.py                       # usa MODEL_TO_RUN configurado no topo de main.py
python main.py --device cpu          # força CPU (default: auto -> cuda se disponível)
python main.py --max-cpu-cores 4     # limita núcleos (paralelismo walk-forward + RF + torch)
python run_all_products.py           # roda todos os produtos -> sliding_rmse_all_products.xlsx
```

### Pontos de atenção
- **Cache:** `.boxplot_cache/` guarda erros por janela. A chave inclui device/CPU-cap mas **NÃO** a lógica dos modelos — ao alterar um modelo em `forecast_methods.py`/`models.py`, **limpe o cache** (`run_all_products.py` já faz `rmtree`; para `main.py`, apague `.boxplot_cache/` manualmente) ou os resultados ficarão obsoletos.
- **Paralelismo:** análises usam `ProcessPoolExecutor`. Funções submetidas devem ser picklable (sem lambdas/closures) — mantenha `_worker` em `cache.py`.
- Modelos torch silenciam erros e retornam `Series` vazia quando a série é curta demais; análise ignora janelas falhas. Ao depurar resultados ausentes, verifique o tamanho da série e os `except` silenciosos.
- `__pycache__/` está versionado por engano (aparece em `git status`); não inclua esses `.pyc` em commits de conteúdo.

## Saídas para o artigo (LaTeX / Overleaf)

- **Figuras:** boxplots e mapas são exportados em **PDF (vetorial, preferido para LaTeX)** e PNG (`scale=3`, fallback) via kaleido. Ao mexer em figuras, regenere ambos.
- **Tabelas:** RMSE/MAPE saem em `.xlsx`; para o artigo geralmente viram tabelas LaTeX — confira se números/colunas batem com o que está escrito.
- Ao alterar um script que alimenta uma figura/tabela do artigo, **avise qual artefato precisa ser regenerado** e, idealmente, regenere-o.

## Skills relevantes

Use estas skills quando a tarefa se encaixar:

- **`xlsx`** — bases e saídas são todas planilhas (`Databases/*.xlsx`, `sliding_rmse*.xlsx`, `descriptive_stats.xlsx`). Use para inspecionar/editar/validar planilhas sem escrever código descartável.
- **`pdf`** — figuras do artigo são exportadas em PDF; use para ler/verificar/combinar PDFs de figuras.
- **`run`** — para de fato rodar os scripts de previsão e ver a saída real (não só testes).
- **`verify`** — confirmar que uma modificação em script realmente produz o resultado esperado (rodar e observar comportamento) antes de dar como pronto.
- **`code-review`** — revisar o diff de mudanças nos scripts antes de commitar (`/code-review`); `ultra` para revisão profunda na nuvem.
- **`simplify`** — limpeza/refatoração dos scripts (reuso, simplificação, eficiência), sem caçar bugs.
- **`skill-creator`** — **não há skill de LaTeX/Overleaf**. Se o trabalho de escrita do artigo virar repetitivo (montar tabelas LaTeX a partir de `.xlsx`, padronizar figuras, gerar `\includegraphics`), considere criar uma skill dedicada com esta.

> Observação: o conteúdo do artigo vive no **Overleaf** (fora deste repositório). O Claude
> assiste com texto LaTeX, estrutura, tabelas e revisão; o usuário cola/sincroniza no Overleaf.

## Convenções

- Comentários e docstrings em **inglês** (padrão do código atual); mensagens ao usuário podem ser em PT-BR.
- Ao adicionar um modelo: implemente `(series, n_periods) -> (Series, model)` em `forecast_methods.py`, registre em `MODEL_FUNCS`, e **limpe o cache** antes de reavaliar.
- Não commitar nem fazer push sem o usuário pedir.
