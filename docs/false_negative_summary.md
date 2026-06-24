# False-Negative-Analyse: Log-only (Stand C)

> Qualitative Fehleranalyse — keine neue Evaluation, keine Threshold-Änderung.

## Überblick

- **Anzahl FN:** 3 (von 48 positiven Test-Cases)
- **Modell:** Log-only (TF-IDF + Logistic Regression, normalisiert)
- **Log-Threshold (Validation):** 0.164056
- **Alle drei FN-Cases haben log_score < threshold.**

## FN-Cases im Detail

| case_id | scenario_id | log_score | threshold | margin | qualitative_note |
|---|---|---|---|---|---|
| mali_6_10.csv | scenario_6 | 0.071899 | 0.164056 | -0.092157 | score moderately below threshold; several security-relevant keyword hits (e.g. cron, sshd, root) |
| mali_6_16.csv | scenario_6 | 0.067102 | 0.164056 | -0.096954 | score moderately below threshold; several security-relevant keyword hits (e.g. cron, sshd, root) |
| mali_6_24.csv | scenario_6 | 0.055297 | 0.164056 | -0.108759 | score far below threshold, logs appear weakly indicative; several security-relevant keyword hits (e.g. cron, sshd, root) |

## Score-Abstände zum Threshold

- **mali_6_10.csv** (`scenario_6`): margin = -0.0922 → moderat unter Threshold
- **mali_6_16.csv** (`scenario_6`): margin = -0.0970 → moderat unter Threshold
- **mali_6_24.csv** (`scenario_6`): margin = -0.1088 → deutlich unter Threshold

## Erkennbare Angriffshinweise in den Log-Excerpts

- **mali_6_10.csv**: {'sshd': 1, 'accepted': 1, 'cron': 80, 'root': 40, '/etc/': 40}
- **mali_6_16.csv**: {'sshd': 2, 'cron': 8, 'root': 4, '/etc/': 4}
- **mali_6_24.csv**: {'sshd': 1, 'cron': 4, 'root': 2, '/etc/': 2}

## Fazit

Die Threshold-Sensitivity-Analyse (NB03) belegt bereits, dass kein validierungsbasierter
Threshold den Test-Recall auf > 0,9375 hebt: Die drei FN-Cases liegen unterhalb jedes
auf dem Validierungsset bestimmbaren Thresholds. Die Scores dieser Cases liegen im
niedrigen Bereich — das Modell hat für diese Cases keine ausreichend starken
semantischen Muster im normalisierten Log-Text gefunden.

**Methodischer Hinweis:** Diese Analyse ist rein deskriptiv. Es wurde keine
testbasierte Optimierung vorgenommen. Die Ergebnisse aus `final_results_table.csv`
bleiben unverändert.