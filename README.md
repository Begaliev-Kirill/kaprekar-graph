# Kaprekar Analyzer

A small tool for analyzing and visualizing the Kaprekar function as a graph.
Edit `MAX_NODES` in `kaprekar_graph.py` to fit your setup.

## Requirements

- Python 3.9+
- Dependencies: `numpy`, `pyqtgraph`, `PyQt5` (see `requirements.txt`)

## Setup

```bash
git clone https://github.com/Begaliev-Kirill/kaprekar-graph.git
cd kaprekar-graph
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python kaprekar_graph.py
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python kaprekar_graph.py
```
