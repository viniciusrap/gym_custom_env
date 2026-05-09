# Coverage Path Planning - APS Reinforcement Learning

Solucao da APS *Um problema mais proximo da realidade* (Insper, RL).

> Relatorio completo: [`RELATORIO.md`](./RELATORIO.md)

## Resultados

200 episodios deterministicos por configuracao:

| Tamanho | Obstaculos | Mean Coverage | Std | Full Coverage |
|---|---|---|---|---|
| **5x5** | 3 | **99.05%** | ±6.92% | 95.5% |
| **10x10** | 12 | **97.22%** | ±10.89% | 74.0% |
| **15x15** | 27 | **92.79%** | ±16.48% | 24.5% |
| **20x20** | 48 | **87.03%** | ±21.90% | 6.0% |

## Setup

```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Avaliar os modelos

```powershell
python train_grid_world_cpp.py test 5 3
python train_grid_world_cpp.py test 10 12
python train_grid_world_cpp.py test 20 48
```

## Visualizar 1 episodio

```powershell
python train_grid_world_cpp.py run 5 3
python train_grid_world_cpp.py run 10 12
python train_grid_world_cpp.py run 20 48
```

## Estrutura

```
.
├── README.md                   este arquivo
├── RELATORIO.md                relatorio completo
├── requirements.txt
├── train_grid_world_cpp.py     treino + avaliacao
├── run_grid_world_cpp.py       agente aleatorio (sanity check)
├── cpp_policy.py               feature extractor da rede
├── plot_results.py             gera os graficos
├── gymnasium_env/
│   ├── __init__.py
│   └── grid_world_cpp.py       ambiente CPP modificado
├── models/                     modelos aprovados (.zip)
└── results/                    CSVs de avaliacao + graficos
```
