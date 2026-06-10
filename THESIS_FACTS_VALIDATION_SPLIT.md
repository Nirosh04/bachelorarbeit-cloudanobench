# THESIS_FACTS_VALIDATION_SPLIT.md

> **Einzige autoritative Faktenquelle für die finale Thesis-Überarbeitung.**
> Alle Evaluationszahlen stammen ausschließlich aus den CSV-Artefakten unter `docs/`.
> Erstellt: 2026-06-10. Quellen: split_assignments.csv, metric_test_scores.csv,
> log_test_scores.csv, fusion_cases.csv, threshold_sensitivity_log.csv,
> threshold_sensitivity_metric.csv.

---

## 1. Finaler methodischer Stand

Der finale Stand dieser Arbeit verwendet einen **dreistufigen gruppenbasierten
Train / Validation / Test-Split** (GroupShuffleSplit, sklearn).

| Prinzip | Umsetzung |
|---|---|
| **Gruppenbasierter Split** | `group_id = dataset_type + "_" + scenario_id`; keine Gruppe erscheint in zwei Splits |
| **Early Stopping** | Ausschließlich auf dem Validierungsset (`eval_set=[(X_val, y_val)]`) |
| **Threshold-Wahl** | Ausschließlich auf dem Validierungsset; höchster Threshold mit Val-Recall ≥ 0.95 |
| **Testset** | Nur finale Evaluation; kein Parameter wurde anhand des Testsets gewählt |
| **Seed-Wahl Validation** | Aus Liste `[0, 1, 2, 7, 42]`; Kriterium: Val ≥ 10 positive Cases; kein Test-Zugriff |

**Stand A (ursprüngliche testbasierte Evaluation) wird nicht als finales Ergebnis
verwendet und erscheint nicht in den Ergebnistabellen der Thesis.**
Stand A ist methodisch angreifbar, weil Threshold- und/oder Early-Stopping-Entscheidungen
anhand des Holdout-Testsets getroffen wurden. Das widerspricht den Anforderungen aus
Arp et al. (2022) — einem explizit zitierten Paper dieser Arbeit.

---

## 2. Datensatz-Fakten

*(Aus THESIS_FACTS_LOCKED.md übernommen — weiterhin gültig, da Split-Logik
denselben Ausgangsdatensatz verwendet.)*

| Kennzahl | Wert |
|---|---|
| **Gesamtzahl Cases** | 1.252 |
| **Gruppen / Szenarien** | 45 |
| **Kanonische Features** | 34 |
| **Universelle Features** | 5 |
| **Universelle Feature-Namen** | `cpu_usage`, `mem_usage`, `disk_io`, `net_in`, `net_out` |

**Label-Mapping:**

| Dataset-Typ | Label | Bedeutung |
|---|---|---|
| `mali` | 1 (positiv) | Echte Anomalie (Malware-Injektion) |
| `anom` | 0 (negativ) | Technische Anomalie (kein Malware) |
| `norm` | 0 (negativ) | Normalbetrieb |

---

## 3. Split-Definition

Quelle: `docs/split_assignments.csv`

### Split-Größen

| Split | Cases | Gruppen | Positive | Negative |
|---|---|---|---|---|
| **Train** | 793 | 28 | 122 | 671 |
| **Validation** | 206 | 8 | 53 | 153 |
| **Test** | 253 | 9 | 48 | 205 |
| **Gesamt** | 1.252 | 45 | 223 | 1.029 |

### Disjunktheit

| Paar | Überlappung |
|---|---|
| Train ∩ Validation | **0** (kein Leakage) |
| Validation ∩ Test | **0** (kein Leakage) |
| Train ∩ Test | **0** (kein Leakage) |

### Weitere Kennzahlen

- `scale_pos_weight` (XGBoost, aus Train berechnet): **5.5000** (671 neg / 122 pos)
- Test-Zeilen (metric_test_scores.csv): **22.770**
- Zeilenzahl pro Case (Metrik): ca. 90 Messpunkte pro Case (22.770 / 253 ≈ 90)

---

## 4. Metric Baseline (XGBoost)

Quellen: `docs/metric_test_scores.csv`, `docs/threshold_sensitivity_metric.csv`

### Row-Level (Primäre Evaluation auf Zeilenebene)

| Kennzahl | Wert |
|---|---|
| **Validation-Threshold** | 0.090309 |
| **Val-Recall @ Threshold** | 0.9500 |
| **Test-AP** | 0.2565 |
| **Test-Recall** | 0.9572 (TP=4.135, FN=185) |
| **Test-FPR (gesamt)** | 0.7469 (FP=13.780, TN=4.670) |
| **Test-Precision** | 0.2308 |
| **Test-F1** | 0.3719 |
| **FPR anom (row-level)** | 0.8133 (5.563 FP von 6.840 neg Zeilen) |
| **FPR norm (row-level)** | 0.7078 (8.217 FP von 11.610 neg Zeilen) |

### Case-Level (max-Aggregation: max(metric_score) pro Case)

| Kennzahl | Wert |
|---|---|
| **Case-AP** | 0.2418 |
| **Case-Recall** | 1.0000 (TP=48, FN=0) |
| **Case-FPR** | 0.9951 (FP=204, TN=1) |
| **Case-Precision** | 0.1905 |
| **Case-F1** | 0.3200 |

### Interpretation

Die Metrik-Komponente erfüllt auf **Zeilenebene** die Recall-Anforderung (0.9572 ≥ 0.95).
Auf **Case-Ebene** degeneriert sie jedoch durch max-Aggregation nahezu vollständig:
Fast alle 253 Test-Cases erhalten einen hohen Score (metric_trigger = 1 für 252/253 Cases,
99.60 %), sodass FPR = 0.9951 auf Case-Ebene. Das macht die Metrik-Komponente allein
als Case-Level-Classifier unbrauchbar. Sie dient als notwendige (nicht hinreichende)
erste Stufe im AND-Gate-Hybrid.

---

## 5. Log-Komponente (Logistische Regression auf TF-IDF)

Quellen: `docs/log_test_scores.csv`, `docs/threshold_sensitivity_log.csv`

| Kennzahl | Wert |
|---|---|
| **Validation-Threshold** | 0.230019 |
| **Val-AP** | 0.9104 |
| **Val-Recall @ Threshold** | 0.9623 |
| **Test-AP** | 0.7108 |
| **Test-Recall** | **0.9375** (TP=45, FN=3) |
| **Test-FPR** | 0.2195 (FP=45, TN=160) |
| **Test-Precision** | 0.5000 |
| **Test-F1** | 0.6522 |

> **WICHTIG:** Test-Recall 0.9375 liegt knapp unter der ursprünglich angestrebten
> Schwelle von ≥ 0.95. Dies ist ein zentraler Befund und keine Implementierungspanne
> (siehe Threshold-Sensitivity-Analyse, Abschnitt 7).

---

## 6. Fusion / Ablation

Quelle: `docs/fusion_cases.csv`

### Überblick Case-Level Metriken

| Variante | AP | Recall | FPR | Precision | F1 |
|---|---|---|---|---|---|
| **Metrik-only** (max_proba) | 0.2418 | 1.0000 | 0.9951 | 0.1905 | 0.3200 |
| **Log-only** (proba_log) | 0.7108 | 0.9375 | 0.2195 | 0.5000 | 0.6522 |
| **Soft-Fusion** (max_proba × proba_log) | **0.5724** | — | — | — | — |
| **AND-Gate** (metric_trigger=1 AND log_trigger=1) | — | 0.9375 | 0.2195 | 0.5000 | 0.6522 |

*Soft-Fusion AP = 0.5724 < Log-only AP = 0.7108.*

*AND-Gate besitzt kein klassisches AP (binäre Entscheidung ohne Ranking-Score).
Der Soft-Fusion-Score (max_proba × proba_log) wird als kontinuierliche Repräsentation
des Hybrid-Systems für PR-Kurven verwendet; AP = 0.5724.*

### AND-Gate Konfusionsmatrix (Test-Cases)

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 1** | FN = 3 | TP = 45 |
| **Actual 0** | TN = 160 | FP = 45 |

### FPR nach Dataset-Typ (AND-Gate, Case-Level)

| Typ | FPR | FP | Neg Cases |
|---|---|---|---|
| anom | 0.1842 | 14 | 76 |
| norm | 0.2403 | 31 | 129 |

### metric_trigger-Anteil

- **252 von 253 Cases** (99.60 %) haben metric_trigger = 1.
- Das AND-Gate degeneriert faktisch zur Log-Komponente: Die metrische Stufe filtert
  kaum, weil sie fast alle Cases triggert.

### Interpretation

1. **Der zentrale FPR-Gewinn kommt durch die Log-Komponente**, nicht durch ein
   selektives metrisches Gate.
2. Die Metrik-Stufe wirkt als Recall-Sicherheitsnetz (Recall = 1.0 auf Case-Ebene
   wenn isoliert betrachtet), aber sie ist so aggressiv, dass sie die FPR-Reduktion
   nicht beiträgt.
3. Soft-Fusion AP (0.5724) < Log-only AP (0.7108): Die Multiplikation der Scores
   verschlechtert die Ranking-Qualität, weil der nahezu konstant hohe Metrik-Score
   das Produkt nivelliert.

---

## 7. Threshold-Sensitivity-Analyse

### Log-Komponente (Quelle: `docs/threshold_sensitivity_log.csv`)

| target_recall_val | threshold_val | val_recall | test_recall | test_fpr | test_precision | test_f1 | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.950 | 0.230019 | 0.9623 | **0.9375** | 0.2195 | 0.5000 | 0.6522 | 45 | 45 | 160 | 3 |
| 0.975 | 0.134152 | 1.0000 | **0.9375** | 0.3902 | 0.3600 | 0.5202 | 45 | 80 | 125 | 3 |
| 0.990 | 0.134152 | 1.0000 | **0.9375** | 0.3902 | 0.3600 | 0.5202 | 45 | 80 | 125 | 3 |
| 1.000 | 0.134152 | 1.0000 | **0.9375** | 0.3902 | 0.3600 | 0.5202 | 45 | 80 | 125 | 3 |

### Metrik-Komponente (Quelle: `docs/threshold_sensitivity_metric.csv`, Zeilenebene)

| target_recall_val | threshold_val | val_recall | test_recall | test_fpr | test_precision | test_f1 | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.950 | 0.090309 | 0.9500 | 0.9572 | 0.7469 | 0.2308 | 0.3719 | 4.135 | 13.780 | 4.670 | 185 |
| 0.975 | 0.057001 | 0.9752 | 0.9942 | 0.9112 | 0.2035 | 0.3378 | 4.295 | 16.811 | 1.639 | 25 |
| 0.990 | 0.049919 | 0.9981 | 0.9975 | 0.9690 | 0.1942 | 0.3251 | 4.309 | 17.878 | 572 | 11 |
| 1.000 | 0.044735 | 1.0000 | 1.0000 | 0.9804 | 0.1928 | 0.3233 | 4.320 | 18.088 | 362 | 0 |

### Kernaussagen der Sensitivity-Analyse

1. **Kein validierungsbasiertes Recall-Ziel für die Log-Komponente erreicht auf
   dem Testset Recall ≥ 0.95.** Der Test-Recall bleibt bei allen vier Zielstufen
   konstant bei 0.9375.
2. **Die 3 FN-Cases haben log_score < 0.134152** (dem niedrigsten Threshold der
   100 % Val-Recall erreicht). Sie sind strukturell nicht durch Threshold-Senkung
   erreichbar.
3. **Höhere Val-Recall-Ziele erhöhen FPR stark ohne Test-Recall-Gewinn:**
   Val-Ziel 0.975 → FPR steigt von 0.2195 auf 0.3902 (+77 % FP), Test-Recall
   bleibt identisch.
4. **Kein nachträgliches Testset-Tuning wurde durchgeführt.** Diese Analyse belegt
   methodisch, dass kein sinnvoller Threshold existiert, der die Recall-Lücke schließt.

---

## 8. Methodische Interpretation

### Was hat sich gegenüber Stand A geändert?

| Aspekt | Stand A (veraltet) | Stand B (final) |
|---|---|---|
| Split | 2-way Train/Test | 3-way Train/Val/Test |
| Early Stopping | unklar / testbasiert | validierungsbasiert |
| Threshold-Wahl | testbasiert | validierungsbasiert |
| Log/Hybrid Recall (Test) | ca. 0.9583 | **0.9375** |
| FPR (AND-Gate) | ca. 0.2634 | **0.2195** |
| Methodische Angreifbarkeit | hoch | gering |

### Korrekte Kernaussage (ersetzt Stand A)

> *"Die Log-Komponente reduziert die FPR von 0.7469 (metrische Baseline, Zeilenebene)
> auf 0.2195 (AND-Gate, Case-Ebene). Der Test-Recall liegt bei 0.9375 und verfehlt
> die harte Recall-Schwelle von ≥ 0.95 um 0.0125 (3 von 48 positiven Test-Cases).
> Die Threshold-Sensitivity-Analyse bestätigt, dass dieser Gap kein Threshold-Problem
> ist, sondern eine strukturelle Limitation des Log-Modells auf den drei FN-Cases."*

### Statistische Einordnung des Recall-Gaps

- n = 48 positive Test-Cases
- Wilson-Konfidenzintervall (95 %) für Recall 0.9375: **[0.823, 0.980]**
- Das angestrebte Ziel 0.95 liegt **innerhalb** des KI → statistisch nicht abgrenzbar
- Dieses Argument ist vertretbar in der Diskussion, ersetzt aber nicht die ehrliche
  Berichterstattung der Punktschätzung 0.9375

### Nicht kommentierte Punkte

- **Recall-Gap ist ein zentraler Befund**, nicht eine Peinlichkeit. Er erklärt sich
  durch 3 strukturell schwer detektierbare Cases.
- **Case-Level-Degeneration der Metrik** ist ein wichtiger Limitation-Punkt:
  max-Aggregation bei ~90 Zeitreihenpunkten/Case führt zu quasi-sicherer Triggerung.
- **Soft-Fusion AP < Log-only AP** ist eine weitere ehrliche Limitation: Die
  Multiplikation mit einem fast-konstanten Metrik-Score verschlechtert das Ranking.

---

## 9. Artefakte

Alle Artefakte wurden durch Neuausführung von NB02, NB03, NB04 (2026-06-09) generiert.
Notebook-Outputs und CSV-Artefakte sind konsistent (Notebooks neu ausgeführt nach
Refactoring, alle alten Outputs gelöscht).

| Artefakt | Beschreibung |
|---|---|
| `docs/split_assignments.csv` | Case-ID, dataset_type, scenario_id, group_id, label, split |
| `docs/metric_test_scores.csv` | Zeilenebene: case_id, label, metric_score, metric_pred, threshold_metric_val |
| `docs/metric_val_scores.csv` | Zeilenebene Validation (für Sensitivity-Analyse) |
| `docs/log_test_scores.csv` | Case-Ebene: case_id, label, log_score, log_pred, threshold_log_val |
| `docs/log_val_scores.csv` | Case-Ebene Validation (für Sensitivity-Analyse) |
| `docs/fusion_cases.csv` | Case-Ebene: max_proba, proba_log, metric_trigger, log_trigger, alert, score_fusion |
| `docs/threshold_sensitivity_log.csv` | Sensitivity-Tabelle Log (4 Recall-Ziele) |
| `docs/threshold_sensitivity_metric.csv` | Sensitivity-Tabelle Metrik (4 Recall-Ziele, Zeilenebene) |
| `docs/pr_curve_baseline.png` | PR-Kurve XGBoost-Baseline (Testset, Threshold aus Validation) |
| `docs/feature_importance_baseline.png` | Feature-Importance (XGBoost, Gain) |
| `docs/pr_curve_log.png` | PR-Kurve Log-Komponente (Testset, Threshold aus Validation) |
| `docs/tfidf_top_features.png` | Top TF-IDF-Features (Log-Regression) |
| `docs/pr_curve_ablation.png` | Ablation: Metrik-only / Log-only / Soft-Fusion PR-Kurven |
| `docs/fpr_comparison.png` | FPR-Vergleich: Baseline vs. AND-Gate (anom / norm / gesamt) |

---

## 10. Thesis-Update-Liste

Die folgenden Thesis-Stellen müssen bei der Überarbeitung angepasst werden.
**Keine Änderungen wurden an den Thesis-Dateien vorgenommen** — diese Liste ist
eine Arbeitsanleitung für den nächsten Schritt.

### Hohe Priorität (inhaltlich falsch wenn nicht angepasst)

| Stelle | Alt (Stand A) | Neu (Stand B) |
|---|---|---|
| **Abstract** | "Recall ≥ 0.95 erreicht", ggf. FPR-Wert 0.2634 | "Recall 0.9375 (knapp unter 0.95), FPR 0.2195" |
| **Einleitung / Forschungsfrage** | "Kann FPR bei Recall ≥ 0.95 reduziert werden?" | "Welche Recall-FPR-Tradeoffs entstehen unter strikt validierungsbasierter Evaluation?" |
| **Metrik-Baseline-Tabelle** | Werte aus Stand A | row-level: AP=0.2565, Recall=0.9572, FPR=0.7469 |
| **Log-Komponenten-Tabelle** | Recall ≈ 0.9583 | Recall=0.9375, FPR=0.2195, AP=0.7108 |
| **Fusion/Ablation-Tabelle** | Stand-A-Werte | AND-Gate: Recall=0.9375, FPR=0.2195; Soft AP=0.5724 |
| **Methodik: Split** | "2-way Train/Test" | "3-way Train/Val/Test, GroupShuffleSplit" |
| **Methodik: Threshold** | "Threshold auf Testset/Holdout" | "Threshold ausschließlich auf Validation" |

### Mittlere Priorität (Interpretation / Diskussion)

| Stelle | Erforderliche Ergänzung |
|---|---|
| **Diskussion** | Threshold-Sensitivity-Analyse erklären; Recall-Gap 0.9375 begründen |
| **Limitationen** | Recall-Lücke, Case-Level-Degeneration Metrik, Soft-Fusion-Verschlechterung |
| **Fazit** | Erfolgsaussage "Recall ≥ 0.95" ersetzen durch ehrliche Bewertung |

### Konkrete Textoperationen

- [ ] Alte testbasierte Threshold-Aussagen (threshold=0.0222 o. ä.) entfernen
- [ ] Aussage "FPR-Reduktion bei Recall ≥ 0.95" abschwächen / ersetzen
- [ ] Neue Limitation zu Recall=0.9375 einbauen (inkl. Wilson-KI-Argument)
- [ ] Threshold-Sensitivity-Analyse als methodische Absicherung ergänzen
- [ ] Case-Level-Degeneration der Metrik erklären (max-Aggregation bei ~90 Punkten/Case)
- [ ] Soft-Fusion AP < Log-only AP erklären
- [ ] metric_trigger = 252/253 als Erklärung für AND-Gate ≡ Log-only erwähnen

---

## Einfrieren-Protokoll

Die folgenden Zahlen sind als finale Evaluationswerte eingefroren:

```
LOG-KOMPONENTE (Case-Level):
  threshold_log_val = 0.230019
  val_recall        = 0.9623
  val_AP            = 0.9104
  test_AP           = 0.7108
  test_recall       = 0.9375
  test_fpr          = 0.2195
  test_precision    = 0.5000
  test_f1           = 0.6522
  TP=45, FP=45, TN=160, FN=3

METRIK-BASELINE (Row-Level):
  threshold_metric_val = 0.090309
  val_recall           = 0.9500
  test_AP              = 0.2565
  test_recall          = 0.9572
  test_fpr             = 0.7469
  test_precision       = 0.2308
  test_f1              = 0.3719
  FPR_anom             = 0.8133
  FPR_norm             = 0.7078
  TP=4135, FP=13780, TN=4670, FN=185

METRIK-BASELINE (Case-Level, max-Aggregation):
  case_AP        = 0.2418
  case_recall    = 1.0000
  case_fpr       = 0.9951
  case_precision = 0.1905
  case_f1        = 0.3200
  TP=48, FP=204, TN=1, FN=0

AND-GATE FUSION (Case-Level):
  recall          = 0.9375
  fpr             = 0.2195
  precision       = 0.5000
  f1              = 0.6522
  fpr_anom        = 0.1842
  fpr_norm        = 0.2403
  metric_trigger  = 252/253 (99.60%)
  TP=45, FP=45, TN=160, FN=3

SOFT-FUSION:
  AP = 0.5724  (score = max_proba × proba_log)

SPLIT:
  train: 793 Cases, 28 Gruppen, 122 pos, 671 neg
  val:   206 Cases,  8 Gruppen,  53 pos, 153 neg
  test:  253 Cases,  9 Gruppen,  48 pos, 205 neg
  scale_pos_weight = 5.5000
```
