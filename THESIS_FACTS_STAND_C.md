# THESIS_FACTS_STAND_C.md

> **Einzige autoritative Faktenquelle für die Thesis-Überarbeitung auf Stand C.**
> Alle Evaluationszahlen stammen ausschließlich aus den CSV-Artefakten unter `docs/`.
> Erstellt: 2026-06-13. Quellen: `split_assignments.csv`, `split_distribution.csv`,
> `metric_case_features.csv`, `metric_case_val_scores.csv`, `metric_case_test_scores.csv`,
> `metric_case_results.csv`, `log_val_scores.csv`, `log_test_scores.csv`, `log_results.csv`,
> `fusion_cases.csv`, `fusion_val_scores.csv`, `final_results_table.csv`,
> `fpr_by_dataset_type.csv`, `threshold_sensitivity_log.csv`, `threshold_sensitivity_metric.csv`.
>
> **Stand B (THESIS_FACTS_VALIDATION_SPLIT.md) wird hiermit abgelöst.**
> Stand-B-Zahlen dürfen nicht mehr in die Thesis übernommen werden;
> ausgenommen sind stabile Datensatz- und Split-Fakten, die weiterhin gültig sind.

---

## 1. Methodischer Finalstand

| Prinzip | Umsetzung |
|---|---|
| **Granularität** | Case-Ebene (eine Zeile pro case_id in allen Evaluationsartefakten) |
| **Aufgabe** | Binary Classification: `mali = 1` (Attack), `anom + norm = 0` (No-Attack) |
| **FPR-Differenzierung** | `anom` und `norm` beide negativ, aber getrennt über `FPR_anom` / `FPR_norm` |
| **Gruppenbasierter Split** | `group_id = dataset_type + "_" + scenario_id`; keine Gruppe in zwei Splits |
| **Training** | Ausschließlich auf Train-Split |
| **Early Stopping (XGBoost)** | Ausschließlich auf Validierungsset (`eval_set=[(X_val, y_val)]`) |
| **Threshold-Wahl** | Ausschließlich auf Validierungsset; höchster Threshold mit Val-Recall_mali ≥ 0.95 |
| **Testset** | Nur finale Evaluation; kein Parameter wurde anhand des Testsets gewählt |
| **Modelle** | XGBoost (Metrik), TF-IDF + Logistic Regression (Log); kein Training in NB04 |
| **Testbasierte Optimierung** | Keine — methodisch sauber nach Arp et al. (2022) |

**Binäre Klassifikation:**
- Positiv (Label 1): `mali` — reale Malware-Injektion
- Negativ (Label 0): `anom` (technische Anomalie, kein Malware) + `norm` (Normalbetrieb)

---

## 2. Datensatz- und Split-Fakten

Quellen: `docs/split_assignments.csv`, `docs/split_distribution.csv`

### Gesamtdatensatz

| Kennzahl | Wert |
|---|---|
| **Gesamt-Cases** | 1.252 |
| **Gesamt-Gruppen** | 45 |
| **Davon positiv (mali)** | 223 |
| **Davon negativ (anom + norm)** | 1.029 |
| **Universelle Features** | 5 (`cpu_usage`, `mem_usage`, `disk_io`, `net_in`, `net_out`) |

### Split-Verteilung (aus `split_distribution.csv`)

| Split | Cases | Gruppen | n_mali | n_anom | n_norm | neg gesamt | pct_mali |
|---|---|---|---|---|---|---|---|
| **Train** | 793 | 28 | 122 | 326 | 345 | 671 | 15.4 % |
| **Validation** | 206 | 8 | 53 | 58 | 95 | 153 | 25.7 % |
| **Test** | 253 | 9 | 48 | 76 | 129 | 205 | 19.0 % |
| **Gesamt** | **1.252** | **45** | **223** | **460** | **576** | **1.029** | 17.8 % |

### Gruppendisjunktheit

| Paar | Überlappung |
|---|---|
| Train ∩ Validation | **0** — kein Leakage |
| Validation ∩ Test | **0** — kein Leakage |
| Train ∩ Test | **0** — kein Leakage |

### Abgeleitete Kennzahlen

- `scale_pos_weight` (XGBoost, aus Train): **5.5** (671 neg / 122 pos = 5.4918 ≈ 5.5)
- Alle Test-Metriken beziehen sich auf **253 Cases** (Case-Ebene, keine Row-Level-Zählungen)

---

## 3. Case-Level Metric Baseline (Stand C)

Quellen: `docs/metric_case_results.csv`, `docs/metric_case_features.csv`,
`docs/metric_case_val_scores.csv`, `docs/metric_case_test_scores.csv`

### Feature Engineering

- **Feature-Anzahl:** 25 (5 universelle Metriken × 5 Aggregationen)
- **Aggregationen pro Metrik:** `mean`, `max`, `std`, `p95`, `slope`
- **Metriken:** `cpu_usage`, `mem_usage`, `disk_io`, `net_in`, `net_out`
- **Imputation:** Train-Median (kein Leakage; Val/Test verwenden Train-Mediane)
- XGBoost wird **direkt auf Case-Level-Features** trainiert (kein Row-Level-Training)

### XGBoost-Konfiguration (aus Train/Val)

| Parameter | Wert |
|---|---|
| `scale_pos_weight` | 5.5 |
| `best_iteration` (Early Stopping auf Val) | 29 |
| Validation-Threshold (Recall_mali ≥ 0.95) | **0.025625** |
| Val-Recall_mali @ Threshold | **1.0000** |
| Threshold-Quelle | `val_recall>=0.95` |

### Test-Evaluation (Case-Level, Threshold aus Validation)

| Metrik | Wert |
|---|---|
| **Test-Recall_mali** | **1.0000** |
| **Test-Precision** | 0.1897 |
| **Test-F1** | 0.3189 |
| **Test-AP** | 0.2439 |
| **Test-FPR_total** | **1.0000** |
| **Test-FPR_anom** | **1.0000** |
| **Test-FPR_norm** | **1.0000** |
| TP | 48 |
| FP | 205 |
| TN | **0** |
| FN | 0 |

### Interpretation

Das Metrik-Modell triggert **alle 253 Test-Cases** (253/253 = 100 %).
Recall ist maximal (1.0), aber FPR ist ebenfalls maximal (1.0): kein einziger negativer Case
wird korrekt abgewiesen. Die Metrik-Komponente ist auf Case-Ebene **nicht selektiv** —
sie kann allein nicht zwischen `mali` und `anom/norm` unterscheiden.
Dies ist ein zentraler Befund und bestätigt die Notwendigkeit der Log-Komponente.

---

## 4. Log-Komponente mit Normalisierung (Stand C)

Quellen: `docs/log_results.csv`, `docs/log_val_scores.csv`, `docs/log_test_scores.csv`

### Log-Normalisierung (vor TF-IDF)

Synthetische Artefakte werden durch Token-Platzhalter ersetzt — deterministisch,
regelbasiert, kein Daten-Leakage möglich:

| Muster | Token | Beispiel |
|---|---|---|
| UUIDs | `TOKEN_ID` | `550e8400-...` → `TOKEN_ID` |
| Nginx-Timestamps | `TOKEN_TIME` | `[20/Aug/2025:08:01:10 +0000]` → `TOKEN_TIME` |
| Syslog-Timestamps | `TOKEN_TIME` | `Aug 15 10:00:05` → `TOKEN_TIME` |
| ISO-Timestamps (mit Zeit) | `TOKEN_TIME` | `2024-01-12T13:45:22Z` → `TOKEN_TIME` |
| ISO-Datum | `TOKEN_DATE` | `2024-01-12` → `TOKEN_DATE` |
| Uhrzeiten | `TOKEN_TIME` | `13:45:22` → `TOKEN_TIME` |
| IPv4-Adressen | `TOKEN_IP` | `192.168.1.10` → `TOKEN_IP` |
| PIDs (in `[]`) | `TOKEN_PID` | `sshd[11532]` → `sshd[TOKEN_PID]` |
| Ports | `TOKEN_PORT` | `port 54321` → `port TOKEN_PORT` |
| Isolierte Zahlen | `TOKEN_NUM` | `retry 3` → `retry TOKEN_NUM` |

**Bewusst erhalten (Attack-relevante Semantik):** Prozessnamen (`sshd`, `CRON`, `wget`,
`curl`, `bash`), Aktionen (`Accepted`, `Failed`, `CMD`), Pfade und Domains,
Protokollbezeichner (`ssh2`, `HTTP`).

**TF-IDF:** `fit_transform` ausschließlich auf Train; `transform` auf Val/Test.
**Logistic Regression:** `class_weight='balanced'`, `C=1.0`, `solver='lbfgs'`.

### Konfiguration (aus Train/Val)

| Parameter | Wert |
|---|---|
| `class_weight` | `balanced` |
| Validation-Threshold (Recall_mali ≥ 0.95) | **0.164056** |
| Val-Recall_mali @ Threshold | **0.9623** |
| Threshold-Quelle | `val_recall>=0.95` |

### Test-Evaluation (Case-Level, Threshold aus Validation)

| Metrik | Wert |
|---|---|
| **Test-Recall_mali** | **0.9375** |
| **Test-Precision** | 0.4639 |
| **Test-F1** | 0.6207 |
| **Test-AP** | 0.6548 |
| **Test-FPR_total** | 0.2537 |
| **Test-FPR_anom** | 0.2368 |
| **Test-FPR_norm** | 0.2636 |
| TP | 45 |
| FP | 52 |
| TN | 153 |
| FN | 3 |

### Interpretation

Die Log-Komponente ist die **beste Einzelvariante** im Ablationsvergleich.
Der Test-Recall von 0.9375 liegt knapp unter der angestrebten Schwelle von ≥ 0.95
(3 von 48 positiven Test-Cases nicht detektiert). Das ist ein **methodisch ehrlicher
Befund**, keine Implementierungspanne — die Threshold-Sensitivity-Analyse (Abschnitt 6)
belegt, dass kein validierungsbasierter Threshold diesen Gap schließt.

> **Hinweis Stand C vs. Stand B:** Die Log-Normalisierung (Stand C) verändert die
> Ergebnisse gegenüber Stand B leicht: AP sinkt von 0.7108 auf 0.6548, FPR_total steigt
> von 0.2195 auf 0.2537. Das Modell lernt jetzt semantische Muster statt synthetischer
> Varianz (Timestamps, PIDs, IPs), was methodisch sauberer ist.

---

## 5. Fusion / Ablation (Stand C)

Quellen: `docs/final_results_table.csv`, `docs/fusion_cases.csv`,
`docs/fusion_val_scores.csv`, `docs/fpr_by_dataset_type.csv`

### Alle vier Varianten (Test-Set, 253 Cases)

| Variante | Recall_mali | Precision | F1 | FPR_total | FPR_anom | FPR_norm | AP | Threshold | Threshold-Quelle | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Metrics-only** | 1.0000 | 0.1897 | 0.3189 | 1.0000 | 1.0000 | 1.0000 | 0.2439 | 0.025625 | `metric_validation` | 48 | 205 | 0 | 0 |
| **Log-only** | 0.9375 | 0.4639 | 0.6207 | 0.2537 | 0.2368 | 0.2636 | 0.6548 | 0.164056 | `log_validation` | 45 | 52 | 153 | 3 |
| **AND-Fusion** | 0.9375 | 0.4639 | 0.6207 | 0.2537 | 0.2368 | 0.2636 | N/A | 0.025625 | `metric_and_log_validation` | 45 | 52 | 153 | 3 |
| **Soft-Fusion** | 0.9375 | 0.4455 | 0.6040 | 0.2732 | 0.2632 | 0.2791 | 0.5756 | 0.004789 | `product_validation` | 45 | 56 | 149 | 3 |

### FPR-Breakdown nach dataset_type (aus `fpr_by_dataset_type.csv`)

| Modell | dataset_type | FP | Neg. Cases | FPR |
|---|---|---|---|---|
| Metrics-only | anom | 76 | 76 | 1.0000 |
| Metrics-only | norm | 129 | 129 | 1.0000 |
| Log-only | anom | 18 | 76 | 0.2368 |
| Log-only | norm | 34 | 129 | 0.2636 |
| AND-Fusion | anom | 18 | 76 | 0.2368 |
| AND-Fusion | norm | 34 | 129 | 0.2636 |
| Soft-Fusion | anom | 20 | 76 | 0.2632 |
| Soft-Fusion | norm | 36 | 129 | 0.2791 |

### Trigger-Statistik (Test-Set, aus `fusion_cases.csv`)

| Trigger | Anzahl | Anteil |
|---|---|---|
| `metric_trigger = 1` | **253 / 253** | **100.0 %** |
| `log_trigger = 1` | 97 / 253 | 38.3 % |
| `and_trigger = 1` | 97 / 253 | 38.3 % |
| `soft_fusion_trigger = 1` | 101 / 253 | 39.9 % |

### Degeneration der AND-Fusion

- **metric_trigger = 253/253 = 100 %:** Das Metrik-Modell klassifiziert alle Test-Cases als positiv.
- **AND-Fusion ist identisch zu Log-only (Ja):** Da `metric_trigger` überall 1 ist,
  gilt `AND = metric_trigger AND log_trigger = log_trigger` für jeden Case.
- Konfusionsmatrizen von AND-Fusion und Log-only sind **exakt gleich**.

### Soft-Fusion

- **Score:** `score_fusion = metric_score × log_score`
- **Threshold-Bestimmung:** Auf Validation-Produkt-Scores (höchster Threshold mit Val-Recall_mali ≥ 0.95)
- **Validation-Threshold:** 0.004789
- **Soft-Fusion vs. Log-only:**
  - F1: 0.6040 vs. 0.6207 → **Soft-Fusion schlechter**
  - FPR_total: 0.2732 vs. 0.2537 → **Soft-Fusion schlechter**
  - AP: 0.5756 vs. 0.6548 → **Soft-Fusion schlechter**
- Ursache: Der fast konstant hohe `metric_score` (Trigger bei 100 % der Cases)
  nivelliert das Produkt und reduziert die Diskriminanz des Log-Scores.

---

## 6. Threshold-Sensitivity-Analyse

### Log-Komponente (aus `docs/threshold_sensitivity_log.csv`, Stand C)

| Val-Recall-Ziel | Threshold (Val) | Val-Recall | Test-Recall | Test-FPR | FPR_anom | FPR_norm | Precision | F1 | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.950 | 0.164056 | 0.9623 | **0.9375** | 0.2537 | 0.2368 | 0.2636 | 0.4639 | 0.6207 | 45 | 52 | 153 | 3 |
| 0.975 | 0.122844 | 1.0000 | **0.9375** | 0.3512 | 0.4079 | 0.3178 | 0.3846 | 0.5455 | 45 | 72 | 133 | 3 |
| 0.990 | 0.122844 | 1.0000 | **0.9375** | 0.3512 | 0.4079 | 0.3178 | 0.3846 | 0.5455 | 45 | 72 | 133 | 3 |
| 1.000 | 0.122844 | 1.0000 | **0.9375** | 0.3512 | 0.4079 | 0.3178 | 0.3846 | 0.5455 | 45 | 72 | 133 | 3 |

### Metrik-Komponente (aus `docs/threshold_sensitivity_metric.csv`, Stand C)

| Val-Recall-Ziel | Threshold (Val) | Val-Recall | Test-Recall | Test-FPR | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| 0.950 | 0.025625 | 1.0 | 1.0 | 1.0 | 48 | 205 | 0 | 0 |
| 0.975 | 0.025625 | 1.0 | 1.0 | 1.0 | 48 | 205 | 0 | 0 |
| 0.990 | 0.025625 | 1.0 | 1.0 | 1.0 | 48 | 205 | 0 | 0 |
| 1.000 | 0.025625 | 1.0 | 1.0 | 1.0 | 48 | 205 | 0 | 0 |

### Kernaussagen der Sensitivity-Analyse

1. **Log-Komponente:** Kein validierungsbasierter Recall-Zielwert verbessert den Test-Recall.
   Bei allen vier Zielwerten (0.95–1.00) bleibt Test-Recall konstant bei **0.9375**.
2. **3 FN-Cases sind strukturell nicht erreichbar:** Ihre log_scores liegen unter 0.122844
   (dem Threshold, der Val-Recall = 100 % erreicht) — Threshold-Senkung hilft nicht.
3. **Höhere Val-Ziele erhöhen FPR stark:** Von 0.95 auf 0.975 steigt FPR von 0.2537
   auf 0.3512 (+38 % mehr FP), Test-Recall bleibt identisch.
4. **Metrik-Sensitivity konstant:** Unabhängig vom Zielwert triggert das Metrik-Modell
   immer alle Cases (TN=0). Der einzige mögliche Threshold ist 0.025625.
5. **Keine testbasierte Nachoptimierung wurde durchgeführt.**

---

## 7. Finale Ergebnisstory für die Thesis

Die folgenden Aussagen sind durch die Stand-C-Artefakte belegt und methodisch korrekt:

1. **Metrics-only ist auf Case-Ebene nicht selektiv.**
   Das XGBoost-Modell, trainiert auf 25 Case-Level-Features (5 Metriken × 5 Aggregationen),
   klassifiziert alle 253 Test-Cases als positiv (FPR_total = 1.0, TN = 0).
   Es kann nicht zwischen `mali` und `anom/norm` unterscheiden, weil viele
   technische Anomalien (anom) und sogar Normalszenarien (norm) ähnlich hohe
   Metrik-Peaks erzeugen wie Malware-Szenarien.

2. **Log-only liefert die beste Trennleistung.**
   TF-IDF + Logistic Regression auf normalisierten Syslog-Texten erreicht
   FPR_total = 0.2537 bei Test-Recall_mali = 0.9375 (AP = 0.6548).
   Es ist die einzige Komponente mit echter Selektivität.

3. **Fusion verbessert Log-only nicht.**
   - AND-Fusion degeneriert zu Log-only (metric_trigger = 100 % → AND ≡ Log).
   - Soft-Fusion (Produkt-Score) ist schlechter als Log-only
     (F1: 0.6040 vs. 0.6207, FPR: 0.2732 vs. 0.2537, AP: 0.5756 vs. 0.6548).

4. **Der Hauptnutzen kommt vom Log-Kontext.**
   Syslog-Daten ermöglichen semantische Unterscheidung zwischen Malware-Aktionen
   (z. B. `wget`, `curl`, `bash`-Pipelines, `/etc/cron`-Einträge) und technischen
   Anomalien oder Normaloperationen.

5. **Die 0.95-Recall-Schwelle wird auf dem Testset knapp verfehlt.**
   Test-Recall_mali = 0.9375 (45/48 detektiert, 3 FN). Die 3 nicht-detektierten Cases
   haben log_scores unterhalb jedes validierungsbasierten Thresholds — das ist eine
   strukturelle Limitation, kein Threshold-Problem.

6. **Das ist ein methodisch ehrlicher Befund.**
   Die Threshold-Sensitivity-Analyse belegt: kein validierungsbasierter Threshold
   schließt den Recall-Gap. Kein Testset-Tuning wurde durchgeführt.
   Das Ergebnis ist reproduzierbar und ohne Testset-Bias.

### Statistische Einordnung des Recall-Gaps

- n = 48 positive Test-Cases
- Wilson-Konfidenzintervall (95 %) für Recall 0.9375: **[0.823, 0.980]**
- Das angestrebte Ziel 0.95 liegt **innerhalb** des KI → statistisch nicht abgrenzbar
- Dieses Argument ist vertretbar in der Diskussion, ersetzt aber nicht die ehrliche
  Berichterstattung der Punktschätzung 0.9375.

---

## 8. Tabelle für Hamed (Betreuer)

Aus `docs/final_results_table.csv` — Case-Level, Test-Set (253 Cases: 48 mali, 76 anom, 129 norm):

| Modell | Recall_mali | Precision | F1 | FPR_anom | FPR_norm | FPR_total | AP | Threshold-Quelle |
|---|---|---|---|---|---|---|---|---|
| Metrics-only | 1.0000 | 0.1897 | 0.3189 | 1.0000 | 1.0000 | 1.0000 | 0.2439 | `metric_validation` |
| Log-only | 0.9375 | 0.4639 | 0.6207 | 0.2368 | 0.2636 | 0.2537 | 0.6548 | `log_validation` |
| AND-Fusion | 0.9375 | 0.4639 | 0.6207 | 0.2368 | 0.2636 | 0.2537 | N/A | `metric_and_log_validation` |
| Soft-Fusion | 0.9375 | 0.4455 | 0.6040 | 0.2632 | 0.2791 | 0.2732 | 0.5756 | `product_validation` |

*AND-Fusion ohne AP, da binärer Trigger ohne kontinuierlichen Score.*
*Soft-Fusion-Threshold 0.004789 aus Validation-Produkt-Scores bestimmt.*

---

## 9. Thesis-Update-Liste

Die folgenden Stellen müssen bei der Thesis-Überarbeitung angepasst werden.
**Keine Änderungen an Thesis-Dateien wurden vorgenommen.**

### Hohe Priorität (inhaltlich falsch wenn nicht angepasst)

| Stelle | Alt (Stand B) | Neu (Stand C) |
|---|---|---|
| **Abstract** | Recall ≈ 0.9375, FPR ≈ 0.2195 (AND-Gate) | Recall = 0.9375, FPR = 0.2537 (Log-only = AND); Metrik allein FPR = 1.0 |
| **Einleitung / Forschungsfrage** | Bezug auf AND-Gate-Fusion als Kernergebnis | Log-Komponente ist dominantes Element; Fusion bringt keinen Mehrwert |
| **Methodik: Metric Baseline** | Row-Level XGBoost, Aggregation auf Case-Level | Case-Level XGBoost, 25 Features (5×5), kein Row-Level-Training |
| **Methodik: Log-Normalisierung** | Kein Normalisierungsschritt | TOKEN_IP, TOKEN_TIME, TOKEN_PID, TOKEN_PORT, TOKEN_NUM, TOKEN_DATE, TOKEN_ID vor TF-IDF |
| **Methodik: Fusion** | AND-Gate als eigenständige Stufe mit Zusatznutzen | AND-Gate degeneriert zu Log-only (metric_trigger = 100 %) |
| **Metric-Baseline-Tabelle** | Row-Level-Werte oder veraltete Case-Level-Werte | AP=0.2439, Recall=1.0, FPR=1.0, TN=0, alle Trigger = 1 |
| **Log-Komponenten-Tabelle** | Stand B: AP=0.7108, FPR=0.2195, threshold=0.230019 | Stand C: AP=0.6548, FPR=0.2537, threshold=0.164056 |
| **Fusion/Ablation-Tabelle** | Stand-B-Werte | Vollständige 4-Varianten-Tabelle aus Abschnitt 5 |
| **metric_trigger-Anteil** | 252/253 = 99.6 % (Stand B) | **253/253 = 100 %** (Stand C) |

### Mittlere Priorität (Interpretation / Diskussion)

| Stelle | Erforderliche Ergänzung |
|---|---|
| **Diskussion** | Log-Normalisierung als methodische Verbesserung erklären; Auswirkung auf AP erläutern |
| **Diskussion** | AND-Degeneration durch metric_trigger = 100 % erklären |
| **Diskussion** | Soft-Fusion schlechter als Log-only: Nivellierung durch konstanten Metrik-Score |
| **Limitationen** | Recall-Lücke 0.9375 (3 FN) als strukturelle Grenze; nicht durch Threshold behebbar |
| **Limitationen** | Case-Level-Nicht-Selektivität der Metrik: kein einziger TN auf Test-Set |
| **Fazit** | Erfolgsaussage "Recall ≥ 0.95" ersetzen durch ehrliche Bewertung der 0.9375-Punktschätzung |

### Konkrete Textoperationen

- [ ] Alle alten Threshold-Werte ersetzen: `threshold_metric_val = 0.025625` (war 0.090309), `threshold_log_val = 0.164056` (war 0.230019)
- [ ] Alle FPR-Werte für Log-only und AND-Fusion auf 0.2537 aktualisieren (war 0.2195)
- [ ] Log-AP auf 0.6548 aktualisieren (war 0.7108)
- [ ] metric_trigger-Anteil auf 253/253 = 100 % korrigieren (war 252/253 = 99.6 %)
- [ ] Metrik-Baseline auf Case-Level umschreiben: FPR_total = 1.0, TN = 0
- [ ] Log-Normalisierungsschritt in Methodik beschreiben (TOKEN-Platzhalter)
- [ ] Threshold-Sensitivity-Tabelle (Log, Stand C) einbauen
- [ ] Wilson-KI-Argument für Recall-Gap 0.9375 in Diskussion einbauen
- [ ] Soft-Fusion als schlechter als Log-only berichten, Ursache erklären

---

## 10. Artefakt-Übersicht (Stand C)

| Artefakt | Beschreibung | Quelle |
|---|---|---|
| `docs/split_assignments.csv` | case_id, dataset_type, scenario_id, group_id, label, split (1.252 Zeilen) | NB02 |
| `docs/split_distribution.csv` | Aggregierte Split-Größen | NB02 |
| `docs/metric_case_features.csv` | 25 Case-Level-Features (1.252 × 31 Spalten inkl. Meta) | NB02 |
| `docs/metric_case_val_scores.csv` | metric_score, metric_trigger pro Val-Case (206 Zeilen) | NB02 |
| `docs/metric_case_test_scores.csv` | metric_score, metric_trigger pro Test-Case (253 Zeilen) | NB02 |
| `docs/metric_case_results.csv` | Kompakte Ergebniszeile Metrik-Baseline | NB02 |
| `docs/threshold_sensitivity_metric.csv` | Metrik-Sensitivity (4 Recall-Ziele, Case-Level) | NB02 |
| `docs/feature_importance_baseline.png` | XGBoost Feature-Importance (Gain) | NB02 |
| `docs/pr_curve_baseline.png` | PR-Kurve Metrik-Baseline (Test, Threshold aus Val) | NB02 |
| `docs/log_val_scores.csv` | log_score, log_trigger pro Val-Case (206 Zeilen) | NB03 |
| `docs/log_test_scores.csv` | log_score, log_trigger pro Test-Case (253 Zeilen) | NB03 |
| `docs/log_results.csv` | Kompakte Ergebniszeile Log-Komponente | NB03 |
| `docs/log_preprocessing_examples.csv` | 15 Stichproben vor/nach Normalisierung | NB03 |
| `docs/threshold_sensitivity_log.csv` | Log-Sensitivity (4 Recall-Ziele, Case-Level) | NB03 |
| `docs/pr_curve_log.png` | PR-Kurve Log-Komponente (Test, Threshold aus Val) | NB03 |
| `docs/tfidf_top_features.png` | Top TF-IDF-Features (Log-Regression, Log-Odds) | NB03 |
| `docs/fusion_cases.csv` | 253 Test-Cases mit allen Scores und Triggern | NB04 |
| `docs/fusion_val_scores.csv` | 206 Val-Cases mit Scores und Triggern | NB04 |
| `docs/final_results_table.csv` | 4-Varianten-Ergebnistabelle (Betreuer-Format) | NB04 |
| `docs/fpr_by_dataset_type.csv` | FPR-Breakdown anom/norm je Modell | NB04 |
| `docs/pr_curve_ablation.png` | PR-Kurven für 3 Varianten mit kontinuierlichem Score | NB04 |
| `docs/fpr_comparison.png` | FPR_anom vs. FPR_norm Balkenvergleich | NB04 |

---

## Einfrieren-Protokoll Stand C

Die folgenden Zahlen sind als finale Stand-C-Evaluationswerte eingefroren:

```
DATENSATZ:
  Gesamt-Cases:   1.252
  Gesamt-Gruppen:    45
  scale_pos_weight: 5.5 (671 neg / 122 pos aus Train)

SPLIT (case-level):
  train: 793 Cases, 28 Gruppen, 122 mali,  326 anom, 345 norm, 671 neg
  val:   206 Cases,  8 Gruppen,  53 mali,   58 anom,  95 norm, 153 neg
  test:  253 Cases,  9 Gruppen,  48 mali,   76 anom, 129 norm, 205 neg

METRIC-BASELINE (Case-Level, Stand C):
  n_features          = 25 (5 metriken × 5 aggregationen: mean, max, std, p95, slope)
  scale_pos_weight    = 5.5
  best_iteration      = 29
  threshold_metric_val = 0.025625
  val_recall_mali     = 1.0000
  test_recall_mali    = 1.0000
  test_precision      = 0.1897
  test_f1             = 0.3189
  test_AP             = 0.2439
  test_FPR_total      = 1.0000
  test_FPR_anom       = 1.0000
  test_FPR_norm       = 1.0000
  TP=48, FP=205, TN=0, FN=0
  metric_trigger=1:   253/253 = 100.0%

LOG-KOMPONENTE (Case-Level, Stand C, mit Normalisierung):
  normalization:      TOKEN_IP, TOKEN_TIME, TOKEN_DATE, TOKEN_PID, TOKEN_PORT, TOKEN_NUM, TOKEN_ID
  class_weight:       balanced
  threshold_log_val   = 0.164056
  val_recall_mali     = 0.9623
  test_recall_mali    = 0.9375
  test_precision      = 0.4639
  test_f1             = 0.6207
  test_AP             = 0.6548
  test_FPR_total      = 0.2537
  test_FPR_anom       = 0.2368
  test_FPR_norm       = 0.2636
  TP=45, FP=52, TN=153, FN=3

METRICS-ONLY (Fusion):
  identisch zu Metric-Baseline oben

AND-FUSION (Case-Level, Stand C):
  recall_mali    = 0.9375
  precision      = 0.4639
  f1             = 0.6207
  FPR_total      = 0.2537
  FPR_anom       = 0.2368
  FPR_norm       = 0.2636
  AP             = N/A (binärer Trigger)
  threshold      = 0.025625 (metric) & 0.164056 (log)
  TP=45, FP=52, TN=153, FN=3
  IDENTISCH zu Log-only: Ja (metric_trigger = 100%)

SOFT-FUSION (Case-Level, Stand C):
  score          = metric_score × log_score
  threshold      = 0.004789 (Validation-Produkt-Scores)
  recall_mali    = 0.9375
  precision      = 0.4455
  f1             = 0.6040
  FPR_total      = 0.2732
  FPR_anom       = 0.2632
  FPR_norm       = 0.2791
  AP             = 0.5756
  TP=45, FP=56, TN=149, FN=3
  Soft-Fusion vs. Log-only: schlechter (F1, FPR, AP)
```
