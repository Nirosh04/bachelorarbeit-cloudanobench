# False-Positive-Analyse nach Szenario

> Qualitative Fehleranalyse — keine neue Evaluation, keine Threshold-Änderung.

## Überblick: Aggregierte FPR (Log-only)

- **FPR_anom (Log-only):** 0.2368  (18 / 76 anom-Cases)
- **FPR_norm (Log-only):** 0.2636  (34 / 129 norm-Cases)
- Norm-Fälle haben eine höhere aggregierte FPR (Δ = 0.0267).
- Beide Typen werden in ähnlichem Ausmaß fehlklassifiziert; norm hat mehr absolute FP (34 vs. 18)
  wegen der größeren Grundmenge.

## Top 5 anom-Szenarien mit höchster FPR (Log-only)

| scenario_id | n_cases | FP | FPR | mean_score |
|---|---|---|---|---|
| scenario_17 | 18 | 11 | 0.6111 | 0.3218 |
| scenario_5 | 30 | 6 | 0.2000 | 0.1405 |
| scenario_13 | 28 | 1 | 0.0357 | 0.0937 |

## Top 5 norm-Szenarien mit höchster FPR (Log-only)

| scenario_id | n_cases | FP | FPR | mean_score |
|---|---|---|---|---|
| scenario_8 | 30 | 30 | 1.0000 | 0.5226 |
| scenario_15 | 30 | 2 | 0.0667 | 0.0861 |
| scenario_6 | 24 | 1 | 0.0417 | 0.0717 |
| scenario_4 | 45 | 1 | 0.0222 | 0.0534 |

## Interpretation

- Szenarien mit FPR = 1.0 (alle Cases des Szenarios fälschlich positiv) deuten auf
  Log-Muster hin, die dem Modell verdaechtig erscheinen, obwohl kein Angriff vorliegt.
- Bei anom-Szenarien kann dies an technischen Anomalien mit ähnlichen Prozessaufrufen
  wie bei Malware liegen (z.B. häufige SSH-Events, Cronjobs, Skriptausführungen).
- Bei norm-Szenarien sind ressourcenintensive Backup- oder Batch-Prozesse typische
  Kandidaten für Log-Muster mit oberflächlicher Ähnlichkeit zu Angriffsmustern.
- Metrics-only hat in allen anom/norm-Szenarien FPR = 1.0 (triviales Ergebnis,
  da metric_trigger=1 für alle 253 Cases).

**Methodischer Hinweis:** Diese Analyse ist rein deskriptiv. Die Zahlen in
`final_results_table.csv` wurden nicht verändert.