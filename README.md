# Reduzierung von False Positives in der Cloud-Sicherheit durch Einbeziehung von Log-Kontext in eine metrikbasierte Detection-Pipeline

Bachelorarbeit Informatik · Freie Universität Berlin · Nirosh Heintze

Diese Arbeit entwickelt und evaluiert eine zweistufige hybride Pipeline zur Reduktion von False-Positive-Alarmen auf dem CloudAnoBench-Datensatz: Ein metrischer XGBoost-Detektor (Phase 2) filtert verdächtige Cases vor; ein log-basierter Validator (TF-IDF + logistische Regression, Phase 3) verifiziert diese anhand des zugehörigen Log-Kontexts. Die Pipeline wird in einem leakage-sicheren Ablationsaufbau evaluiert.

## Repository-Struktur

```
notebooks/          Jupyter-Notebooks (Phase 1–5, in Reihenfolge ausführen)
tests/              pytest-Suite (52 Unit-Tests, 12 Testklassen)
docs/               Abbildungen und Audit-Artefakt (fusion_cases.csv)
thesis/             LaTeX-Quellen und kompilierte PDF
data/               Datensatzverzeichnis (nicht im Repo enthalten, s. u.)
requirements.txt    Python-Abhängigkeiten
Dockerfile          Lokale Jupyter-Lab-Umgebung
```

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -r requirements.txt
```

> **Rohdaten:** Der CloudAnoBench-Datensatz ist nicht im Repository enthalten.
> Er muss separat bezogen und in `data/` abgelegt werden, sodass
> `data/mali_dataset/`, `data/anom_dataset/` und `data/norm_dataset/` vorhanden sind.
> Quelle: <https://arxiv.org/abs/2508.01844>

## Notebooks ausführen

Die Notebooks müssen in dieser Reihenfolge ausgeführt werden:

```
01_data_exploration.ipynb   – Datenaufbereitung, kanonische Feature-Matrix, Parquet-Artefakte
02_metric_baseline.ipynb    – XGBoost-Metrik-Baseline (Phase 2)
03_log_component.ipynb      – TF-IDF + logistische Regression (Phase 3)
04_fusion.ipynb             – AND-Gate / Soft-Fusion / Ablation (Phase 4)
05_error_analysis.ipynb     – Fehleranalyse: False Negatives, False Positives nach Szenario,
                              malicious-Test-Szenario-Recall und Reproduzierbarkeitschecks
```

Jedes Notebook setzt die Ausgaben des vorherigen voraus (`data/processed/` und `docs/`).

## Tests

```bash
python -m pytest tests/ -v
```

Erwartet: 52 Tests, alle grün.

## Thesis kompilieren

```bash
cd thesis
pdflatex thesis.tex
biber thesis
pdflatex thesis.tex
pdflatex thesis.tex
```

Die kompilierte PDF liegt bereits als `thesis/thesis.pdf` vor.

## Docker (lokale Reproduzierbarkeit)

Die Rohdaten sind nicht im Image enthalten und müssen beim Start als Volume eingebunden werden.

```bash
docker build -t cloudanobench-thesis .

# Unix / macOS:
docker run --rm -p 8888:8888 \
  -v "${PWD}/data:/app/data" \
  cloudanobench-thesis

# Windows PowerShell:
docker run --rm -p 8888:8888 `
  -v "${PWD}/data:/app/data" `
  cloudanobench-thesis
```

Stelle sicher, dass `data/mali_dataset/`, `data/anom_dataset/` und `data/norm_dataset/`
im lokalen `data/`-Verzeichnis vorhanden sind, bevor du den Container startest.

Öffne anschließend `http://localhost:8888` im Browser.
Der Container dient ausschließlich der lokalen Notebook-Reproduzierbarkeit
und ist kein produktionsreifes Deployment.

## Artefakte

| Datei | Inhalt |
|---|---|
| `docs/fusion_cases.csv` | Audit-Trail: alle 253 Test-Cases mit Zwischen-Scores |
| `docs/pr_curve_*.png` | Precision-Recall-Kurven |
| `docs/fpr_comparison.png` | FPR-Vergleich aller Systemvarianten |
| `docs/feature_*.png` | Feature-Verteilungen und -Importances |
| `docs/tfidf_top_features.png` | Top-TF-IDF-Koeffizienten |
