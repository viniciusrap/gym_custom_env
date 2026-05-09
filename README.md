# Coverage Path Planning - APS Reinforcement Learning (Insper)

Solução para o desafio de Coverage Path Planning (CPP) em GridWorld customizado, usando **RecurrentPPO + LSTM** com **transfer learning** entre tamanhos via curriculum.

> **Relatório completo:** [`RELATORIO.md`](./RELATORIO.md)

## Resultados (200 episódios deterministicos por configuração)

| Tamanho | Obstáculos | Mean Coverage | Std | Full Coverage |
|---|---|---|---|---|
| **5x5** | 3 | **99.05%** | ±6.92% | 95.5% |
| **10x10** | 12 | **97.22%** | ±10.89% | 74.0% |
| **15x15** | 27 | **92.79%** | ±16.48% | 24.5% |
| **20x20** | 48 | **87.03%** | ±21.90% | 6.0% |

5x5 e 10x10 atingem o objetivo do enunciado (cobertura próxima de 100%). O 20x20 com 48 obstáculos (12% de densidade) plateauiza em 87.03% — análise detalhada em `RELATORIO.md`.

## Setup

### Python 3.11 (recomendado)

O ecossistema Stable-Baselines3 + Gymnasium tem fricção real em Python 3.13/3.14 no Windows.

```powershell
# Windows PowerShell
py -3.11 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# Linux/macOS
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Comandos

### Avaliar os modelos treinados

```powershell
python train_grid_world_cpp.py test 5 3
python train_grid_world_cpp.py test 10 12
python train_grid_world_cpp.py test 20 48
```

### Visualização qualitativa de 1 episódio

```powershell
python train_grid_world_cpp.py run 5 3
python train_grid_world_cpp.py run 10 12
python train_grid_world_cpp.py run 20 48
```

### Re-treinar do zero (curriculum 5x5 → 10x10 → 20x20)

```powershell
$env:CPP_N_ENVS = "16"
python train_grid_world_cpp.py curriculum 5 3 200 500000
```

Tempo estimado em CPU 16 threads: ~6h.

### Curriculum interno de obstáculos no 20x20

Após ter `cpp_15x15_approved.zip` em `models/`:

```powershell
python train_grid_world_cpp.py bigtwenty
```

3 sub-etapas (16 → 32 → 48 obstáculos), ~4h.

### Verificação rápida com agente aleatório

```powershell
python run_grid_world_cpp.py 5 3 200 --headless
```

### Gerar gráficos do relatório

```powershell
python plot_results.py
```

Produz em `results/`:
- `coverage_bars.png` — cobertura média por tamanho
- `coverage_distribution.png` — distribuição por episódio
- `obstacle_density_20x20.png` — efeito da densidade no 20x20
- `comparison_table.csv` — tabela com todos os números

## Estrutura do projeto

```
cpp_project/
├── README.md                  Este arquivo
├── RELATORIO.md               Relatório completo
├── requirements.txt           Dependências Python
├── train_grid_world_cpp.py    Script principal (treino + avaliação)
├── run_grid_world_cpp.py      Verificação rápida com agente aleatório
├── cpp_policy.py              CPPFeatureExtractor (CNN + MLP)
├── plot_results.py            Geração dos gráficos
├── gymnasium_env/
│   ├── __init__.py
│   └── grid_world_cpp.py      Ambiente CPP (observação Dict invariante)
├── models/                    Modelos aprovados (.zip)
├── results/                   CSVs de avaliação e gráficos PNG
├── log/                       Logs do TensorBoard
└── checkpoints/               Snapshots best-model do EvalCallback
```

## Estratégia (resumo)

1. **Observação invariante ao tamanho:** `Dict` com `agent_pos` (2,) + `coverage` (1,) + `local_view` (3,3). Permite transfer learning entre 5x5/10x10/15x15/20x20 com os mesmos pesos.
2. **RecurrentPPO + LSTM (128):** memória de curto/médio prazo para compensar a observação parcial pequena.
3. **Recompensa modificada:** `+1.5` célula nova, `-0.4` revisita, `-0.05` step, `+25.0` cobertura completa, `-5.0` timeout. Plus `R_PINGPONG_EXTRA = -0.4` (anti-loop) e `R_FRONTIER_BONUS = +0.05` (pró-fronteira).
4. **Curriculum:** treinar 5x5 → transferir para 10x10 → transferir para 15x15 → transferir para 20x20.
5. **Curriculum interno de obstáculos no 20x20** (`bigtwenty`): subdivide o 20x20 em easy(16 obs) → medium(32 obs) → hard(48 obs).
6. **Verificação BFS de solvabilidade** no `reset()`: descarta layouts impossíveis.

Detalhes completos, justificativas e análise em [`RELATORIO.md`](./RELATORIO.md).

## Hardware usado nos experimentos

- CPU: AMD Ryzen 7 7700X (16 threads)
- RAM: 32 GB
- GPU: RTX 4070 (não utilizada — RecurrentPPO em CPU é mais rápido nessa escala)
- OS: Windows 11
- Python: 3.11.9

## Algoritmo

`RecurrentPPO` (sb3-contrib 2.3.0) com:
- LSTM de 128 unidades
- AdamW (lr=3e-4, weight_decay=1e-4)
- 16 envs paralelos via SubprocVecEnv
- EvalCallback + shootout best-vs-final na avaliação

Justificativas em `RELATORIO.md` seção 3.
