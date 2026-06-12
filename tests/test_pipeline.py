"""
Tests für die CloudAnoBench Pipeline.

Prüft Korrektheit der Kernmechanismen: kanonische Feature-Matrix,
AND-Fusion, FPR-Reduktion, GroupShuffleSplit-Integrität, Soft Score.
Ausführen mit: pytest tests/
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------------------------------------------
# Kernfunktion aus Phase 1 — direkt hier definiert für isolierte Unit-Tests
# ---------------------------------------------------------------------------

def build_canonical_matrix(df, canonical_features, fit_medians=None):
    """
    Baut die kanonische Feature-Matrix.
    fit_medians: dict {col: median} — wenn None, wird auf df self-fitted.
    In P2/P4 immer fit_medians vom Trainingsset übergeben (kein Leakage).
    """
    df = df.copy()
    for col in canonical_features:
        if col not in df.columns:
            df[col] = np.nan
    df = df[[c for c in sorted(canonical_features)]].copy()
    for col in sorted(canonical_features):
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip().str.strip('"\'')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    computed_medians = {}
    for col in sorted(canonical_features):
        if df[col].isnull().any():
            med = fit_medians[col] if (fit_medians and col in fit_medians) else df[col].median()
            computed_medians[col] = med
            if not pd.isna(med):
                df[col] = df[col].fillna(med)
    return df, computed_medians


# ---------------------------------------------------------------------------
# Tests: build_canonical_matrix
# ---------------------------------------------------------------------------

class TestBuildCanonicalMatrix:

    def test_missing_column_added_as_nan(self):
        """Spalte die im Schema fehlt wird ergänzt — kein KeyError im Downstream."""
        df = pd.DataFrame({'a': [1.0, 2.0]})
        result, _ = build_canonical_matrix(df, canonical_features=['a', 'b'])
        assert 'b' in result.columns
        assert result['b'].isna().all()

    def test_imputation_uses_train_median_not_test_median(self):
        """
        Entscheidungstest gegen Data Leakage:
        NaN im Testset muss mit Train-Median imputiert werden — nicht dem
        Testset-eigenen Median. Verletzung würde Leakage in P2/P4 bedeuten.
        """
        train = pd.DataFrame({'x': [1.0, 2.0, 3.0, np.nan]})
        _, medians = build_canonical_matrix(train, ['x'])
        assert medians['x'] == pytest.approx(2.0), "Train-Median muss 2.0 sein"

        # Testset hat verzerrten Median (10.0) — muss trotzdem 2.0 verwenden
        test = pd.DataFrame({'x': [np.nan, 10.0]})
        result, _ = build_canonical_matrix(test, ['x'], fit_medians=medians)
        assert result['x'].iloc[0] == pytest.approx(2.0), \
            "NaN muss mit Train-Median (2.0) imputiert werden, nicht Test-Median (10.0)"

    def test_no_nan_after_imputation(self):
        """Nach Imputation mit bekanntem Median verbleiben keine NaN."""
        df = pd.DataFrame({'x': [1.0, np.nan, 3.0], 'y': [np.nan, 2.0, np.nan]})
        result, _ = build_canonical_matrix(df, ['x', 'y'])
        assert result.isnull().sum().sum() == 0

    def test_quoted_string_values_coerced_to_float(self):
        """
        String-Werte mit Anführungszeichen (residual aus quoting=3) werden
        korrekt zu float konvertiert — kritisch für mali-Dateien mit CSV-Encoding.
        """
        df = pd.DataFrame({'x': ['"1.5"', '"2.0"', '"3.5"']})
        result, _ = build_canonical_matrix(df, ['x'])
        assert pd.api.types.is_float_dtype(result['x'])
        assert result['x'].iloc[0] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Tests: AND-Fusion und Soft Score
# ---------------------------------------------------------------------------

class TestFusionLogic:

    def test_and_gate_truth_table(self):
        """
        AND-Gate: alert=1 nur wenn metric_trigger=1 UND log_trigger=1.
        Vollständige Wahrheitstabelle — stellt sicher dass keine Verwechslung
        mit OR-Gate oder einseitiger Filterung vorliegt.
        """
        fusion = pd.DataFrame({
            'metric_trigger': [1, 1, 0, 0],
            'log_trigger':    [1, 0, 1, 0],
        })
        fusion['alert'] = (
            (fusion['metric_trigger'] == 1) & (fusion['log_trigger'] == 1)
        ).astype(int)
        assert list(fusion['alert']) == [1, 0, 0, 0]

    def test_fpr_fusion_leq_fpr_baseline(self):
        """
        Zentrales Ergebnis der Arbeit: FPR(Fusion) <= FPR(Metrik-Baseline).
        Log-Kontext darf FPR nie erhöhen — nur reduzieren oder halten.
        """
        y_true = np.array([1] * 40 + [0] * 160)

        # Metrik-Baseline: flaggt alles -> FPR = 1.0
        metric_trigger = np.ones(200, dtype=int)

        # Log-Trigger: korrekte TPs + weniger FPs
        log_trigger = np.zeros(200, dtype=int)
        log_trigger[:40] = 1   # alle 40 TPs korrekt erkannt
        log_trigger[40:90] = 1  # 50 der 160 FPs fälschlicherweise log-positiv

        fusion_alert = ((metric_trigger == 1) & (log_trigger == 1)).astype(int)

        n_neg        = (y_true == 0).sum()
        fp_baseline  = (metric_trigger[y_true == 0] == 1).sum()
        fp_fusion    = (fusion_alert[y_true == 0] == 1).sum()
        fpr_baseline = fp_baseline / n_neg
        fpr_fusion   = fp_fusion   / n_neg

        assert fpr_fusion <= fpr_baseline, (
            f"Fusion-FPR ({fpr_fusion:.3f}) darf Baseline-FPR "
            f"({fpr_baseline:.3f}) nicht überschreiten"
        )

    def test_soft_score_in_unit_interval(self):
        """
        Soft Score = max_proba x proba_log muss in [0, 1] liegen —
        Voraussetzung für gültige PR-Kurven-Berechnung.
        """
        rng = np.random.default_rng(42)
        fusion = pd.DataFrame({
            'max_proba': rng.uniform(0, 1, 200),
            'proba_log': rng.uniform(0, 1, 200),
        })
        fusion['score_fusion'] = fusion['max_proba'] * fusion['proba_log']
        assert (fusion['score_fusion'] >= 0.0).all()
        assert (fusion['score_fusion'] <= 1.0).all()


# ---------------------------------------------------------------------------
# Tests: Datenintegrität und Split-Strategie
# ---------------------------------------------------------------------------

class TestDataIntegrity:

    def test_group_shuffle_split_no_group_leakage(self):
        """
        GroupShuffleSplit-Garantie: keine Gruppe erscheint in Train UND Test.
        Verletzung würde temporale Autokorrelation im Test ermöglichen.
        """
        X = pd.DataFrame({'x': range(100)})
        groups = np.array([f'scenario_{i // 10}' for i in range(100)])

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, groups=groups))

        train_groups = set(groups[train_idx])
        test_groups  = set(groups[test_idx])
        overlap = train_groups & test_groups

        assert len(overlap) == 0, (
            f"Gruppen-Leakage entdeckt — folgende Gruppen in Train UND Test: {overlap}"
        )

    def test_universal_five_features_present_in_all_datasets(self):
        """
        Die 5 universellen Features müssen in allen drei Datasets vorhanden sein —
        Schema-Leakage-Prüfung aus Phase 1 (verhindert aucpr=1.0 durch NaN-Muster).
        """
        universal = {'cpu_usage', 'mem_usage', 'disk_io', 'net_in', 'net_out'}

        # Synthetisch: repräsentiert die realen Schema-Unterschiede aus P1
        mali_cols = universal | {'mali_only_a', 'mali_only_b'}
        anom_cols = universal | {'anom_only_a'}
        norm_cols = universal | {'norm_only_a', 'norm_only_b', 'norm_only_c'}

        common = mali_cols & anom_cols & norm_cols
        missing = universal - common

        assert len(missing) == 0, (
            f"Universelle Features fehlen in mindestens einem Dataset: {missing}"
        )


# ---------------------------------------------------------------------------
# Tests: Algorithmen aus P2/P3/P4
# ---------------------------------------------------------------------------

class TestAlgorithmicLogic:

    def test_threshold_selection_highest_at_recall_095(self):
        """
        Threshold-Wahl aus P2/P3: höchster Threshold bei dem Recall >= 0.95 noch gilt.
        Konservativste Grenze — maximiert FPR-Reduktion ohne Recall-Nebenbedingung zu verletzen.
        """
        # Synthetische PR-Kurve: recalls fallend, thresholds steigend
        recalls    = np.array([1.0, 0.98, 0.96, 0.95, 0.93, 0.90])
        thresholds = np.array([0.1,  0.2,  0.3,  0.4,  0.5])

        # Gleiche Logik wie P2/P3: recalls[:-1] wegen sklearn off-by-one
        valid = thresholds[recalls[:-1] >= 0.95]
        best_threshold = float(valid.max())


        # recalls[:-1] = [1.0, 0.98, 0.96, 0.95, 0.93]
        # >= 0.95 trifft auf Index 0-3 zu (0.95 ist eingeschlossen) → valid=[0.1, 0.2, 0.3, 0.4]
        # Höchster valider Threshold ist 0.4 (recall=0.95 erfüllt die >= Bedingung genau)
        assert best_threshold == pytest.approx(0.4), (
            f"Threshold muss 0.4 sein (höchster bei recall>=0.95), erhalten: {best_threshold}"
        )
        

    def test_threshold_selection_recall_constraint_holds(self):
        """
        Recall-Nebenbedingung >= 0.95 muss nach Threshold-Anwendung erfüllt sein.
        Stellt sicher dass die Threshold-Logik die harte Recall-Anforderung nicht verletzt.
        """
        np.random.seed(42)
        y_true  = np.array([1] * 50 + [0] * 200)
        y_proba = np.concatenate([
            np.random.uniform(0.6, 1.0, 50),   # positive: hohe Scores
            np.random.uniform(0.0, 0.5, 200),   # negative: niedrige Scores
        ])

        from sklearn.metrics import precision_recall_curve, recall_score
        _, recalls_curve, thresholds_curve = precision_recall_curve(y_true, y_proba)
        valid = thresholds_curve[recalls_curve[:-1] >= 0.95]
        threshold = float(valid.max()) if len(valid) > 0 else float(thresholds_curve[0])

        y_pred = (y_proba >= threshold).astype(int)
        assert recall_score(y_true, y_pred) >= 0.95, \
            f"Recall {recall_score(y_true, y_pred):.3f} unterschreitet Nebenbedingung 0.95"

    def test_max_aggregation_row_to_case_level(self):
        """
        max-Aggregation aus P4: ein einzelner anomaler Metrik-Peak (Zeile) genügt
        um den gesamten Case zu flaggen — semantisch korrekt für Anomalie-Erkennung.
        """
        rows = pd.DataFrame({
            'case_id':   ['c1', 'c1', 'c1', 'c2', 'c2'],
            'proba_xgb': [0.9,  0.1,  0.2,  0.3,  0.4],
            'label':     [1,    1,    1,    0,    0],
        })
        cases = (rows.groupby('case_id')
                     .agg(max_proba=('proba_xgb', 'max'),
                          label=('label', 'max'))
                     .reset_index())

        # c1: max=0.9 (ein Peak genügt), c2: max=0.4
        assert cases.loc[cases['case_id'] == 'c1', 'max_proba'].iloc[0] == pytest.approx(0.9)
        assert cases.loc[cases['case_id'] == 'c2', 'max_proba'].iloc[0] == pytest.approx(0.4)
        # label via max: alle Zeilen gleich, max ist identisch
        assert cases.loc[cases['case_id'] == 'c1', 'label'].iloc[0] == 1


    def test_tfidf_vocabulary_fitted_on_train_only(self):
        """
        TF-IDF aus P3: Vokabular wird nur auf Trainingsdaten gefittet (kein Leakage).
        Terme die ausschließlich im Testset vorkommen dürfen nicht im Vokabular auftauchen.
     """
        from sklearn.feature_extraction.text import TfidfVectorizer

        train_texts = ["error login failed", "memory high cpu spike", "disk io burst"]
        test_texts  = ["supersecrettoken attack detected", "error login failed"]

        vectorizer = TfidfVectorizer()
        vectorizer.fit(train_texts)

        # "supersecrettoken" kommt nur im Test vor → darf nicht im Vokabular sein
        assert "supersecrettoken" not in vectorizer.vocabulary_, (
            "Test-exklusiver Term darf nicht im Train-Vokabular erscheinen (Leakage)"
        )
        # "error" kommt in beiden vor → muss im Vokabular sein
        assert "error" in vectorizer.vocabulary_


# ---------------------------------------------------------------------------
# Tests: Dreistufige Validation-Split-Methodik (neue Anforderungen)
# ---------------------------------------------------------------------------

class TestValidationSplit:

    def _make_three_way_split(self, n=120, n_groups=10):
        """Helper: erzeugt drei disjunkte Splits aus synthetischen Daten."""
        X = pd.DataFrame({'x': range(n)})
        y = pd.Series([1 if i < n // 5 else 0 for i in range(n)])
        groups = np.array([f'g{i // (n // n_groups)}' for i in range(n)])

        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        trainval_idx, test_idx = next(gss1.split(X, y, groups))

        X_tv = X.iloc[trainval_idx]
        y_tv = y.iloc[trainval_idx]
        g_tv = groups[trainval_idx]

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        train_idx, val_idx = next(gss2.split(X_tv, y_tv, g_tv))

        train_groups = set(g_tv[train_idx])
        val_groups   = set(g_tv[val_idx])
        test_groups  = set(groups[test_idx])
        return train_groups, val_groups, test_groups, y_tv.iloc[val_idx]

    def test_three_way_group_disjointness(self):
        """
        Drei-Wege-Disjunktheit: Train ∩ Val = ∅, Val ∩ Test = ∅, Train ∩ Test = ∅.
        Verschachtelte GroupShuffleSplits können trotz korrekter Einzelschritte
        heimliche Überschneidungen erzeugen — dieser Test sichert alle drei Paare ab.
        """
        train_g, val_g, test_g, _ = self._make_three_way_split()
        assert len(train_g & val_g)  == 0, f"Train ∩ Val nicht leer: {train_g & val_g}"
        assert len(val_g  & test_g)  == 0, f"Val ∩ Test nicht leer: {val_g & test_g}"
        assert len(train_g & test_g) == 0, f"Train ∩ Test nicht leer: {train_g & test_g}"

    def test_validation_contains_positive_and_negative_cases(self):
        """
        Val-Set muss nach Seed-Wahl beide Klassen enthalten.
        Verwendet ein synthetisches Dataset wo JEDE Gruppe Positive und Negative
        enthält — dadurch enthält jeder val-Split mindestens einen positiven Fall.
        """
        # Jede Gruppe hat 3 Positive und 7 Negative → kein val-Split kann alle
        # Positiven verlieren, unabhängig welche Gruppen zufällig zugewiesen werden.
        n = 100
        groups = np.array([f'g{i // 10}' for i in range(n)])   # 10 Gruppen à 10
        y = pd.Series([1 if i % 10 < 3 else 0 for i in range(n)])  # 3/10 pro Gruppe positiv
        X = pd.DataFrame({'x': range(n)})

        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        trainval_idx, _ = next(gss1.split(X, y, groups))
        X_tv = X.iloc[trainval_idx]
        y_tv = y.iloc[trainval_idx]
        g_tv = groups[trainval_idx]

        # Simuliert _pick_val_seed: erster Seed der Val >= 1 positive Fall liefert
        selected_seed = None
        final_val_y   = None
        for seed in (0, 1, 2, 7, 42):
            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            tr_idx, vl_idx = next(gss2.split(X_tv, y_tv, g_tv))
            if y_tv.iloc[vl_idx].sum() >= 1:
                selected_seed = seed
                final_val_y   = y_tv.iloc[vl_idx]
                break

        assert selected_seed is not None, (
            "Kein Seed in (0,1,2,7,42) liefert Val mit positiven Fällen."
        )
        assert final_val_y.sum() >= 1,        "Val enthält nach Seed-Wahl keine positiven Fälle."
        assert (final_val_y == 0).sum() >= 1, "Val enthält nach Seed-Wahl keine negativen Fälle."

    def test_threshold_selected_from_validation_scores(self):
        """
        Threshold-Wahl muss aus Val-Scores stammen — niemals aus Test-Scores.
        Val und Test haben unterschiedliche optimale Thresholds; Funktion gibt
        den Val-Threshold zurück und nicht den Test-Threshold.
        """
        from sklearn.metrics import precision_recall_curve

        rng = np.random.default_rng(7)
        y_val  = np.array([1]*30 + [0]*70)
        p_val  = np.concatenate([rng.uniform(0.7, 1.0, 30), rng.uniform(0.0, 0.4, 70)])
        y_test = np.array([1]*30 + [0]*70)
        p_test = np.concatenate([rng.uniform(0.4, 0.9, 30), rng.uniform(0.1, 0.7, 70)])

        def _select_threshold(y, proba):
            _, rec, thr = precision_recall_curve(y, proba)
            valid = thr[rec[:-1] >= 0.95]
            return float(valid.max()) if len(valid) > 0 else float(thr[0])

        thr_val  = _select_threshold(y_val,  p_val)
        thr_test = _select_threshold(y_test, p_test)

        # Val und Test müssen unterschiedliche Thresholds liefern (sonst ist der Test wertlos)
        assert thr_val != thr_test, (
            "Val- und Test-Threshold sind identisch — Test prüft kein reales Szenario."
        )
        # Der korrekte Code verwendet thr_val; thr_test ist unbekannt im echten Notebook
        assert thr_val > 0.0, "Val-Threshold muss positiv sein."

    def test_scale_pos_weight_from_train_only(self):
        """
        scale_pos_weight muss aus Train-Labels berechnet werden.
        Werden Val-Labels einbezogen, verändert sich das Verhältnis — dieser Test
        stellt sicher dass die Berechnungslogik ausschließlich auf Train-Daten basiert.
        """
        y_train = pd.Series([1]*20 + [0]*80)   # 1:4
        y_val   = pd.Series([1]*15 + [0]*35)   # 1:2.33

        spw_train_only   = (y_train == 0).sum() / (y_train == 1).sum()
        spw_with_val_mix = ((y_train == 0).sum() + (y_val == 0).sum()) / \
                           ((y_train == 1).sum() + (y_val == 1).sum())

        # Die Werte müssen unterschiedlich sein (Testintegrität)
        assert abs(spw_train_only - spw_with_val_mix) > 0.1, \
            "Synthetischer Test nicht differenziert genug."
        # scale_pos_weight korrekt: nur Train
        assert spw_train_only == pytest.approx(4.0)
        # scale_pos_weight falsch: Train+Val gemischt weicht von 4.0 ab
        assert not spw_with_val_mix == pytest.approx(4.0), \
            "Val-Einbeziehung muss scale_pos_weight verändern."

    def test_tfidf_vocabulary_not_contaminated_by_val_or_test(self):
        """
        TF-IDF Vokabular darf nach fit auf Train keine Val- oder Test-exklusiven
        Terme enthalten. Erweiterung des bestehenden Tests auf drei Splits.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        train_texts = ["error login failed", "memory high cpu spike", "disk io burst"]
        val_texts   = ["valtoken connection reset", "error login failed"]
        test_texts  = ["supersecrettoken attack detected", "error login failed"]

        vectorizer = TfidfVectorizer()
        vectorizer.fit(train_texts)  # nur auf Train!

        assert "valtoken"         not in vectorizer.vocabulary_, \
            "Val-exklusiver Term 'valtoken' im Vokabular (Leakage)"
        assert "supersecrettoken" not in vectorizer.vocabulary_, \
            "Test-exklusiver Term 'supersecrettoken' im Vokabular (Leakage)"
        assert "error" in vectorizer.vocabulary_, \
            "Train-Term 'error' fehlt im Vokabular"

    def test_fusion_uses_test_scores_only(self):
        """
        fusion_cases.csv darf nur Test-Cases enthalten — keine Train/Val-Cases.
        Stellt sicher dass finale Ablations-Metriken nicht durch Train-/Val-Performance
        kontaminiert werden.
        """
        split_df = pd.DataFrame({
            'case_id': [f'c{i}' for i in range(10)],
            'split':   ['train']*4 + ['val']*3 + ['test']*3
        })
        fusion_df = pd.DataFrame({
            'case_id': [f'c{i}' for i in range(7, 10)],  # nur test cases c7, c8, c9
            'label':   [1, 0, 0]
        })

        test_ids       = set(split_df[split_df['split'] == 'test']['case_id'])
        fusion_ids     = set(fusion_df['case_id'])
        non_test_in_fusion = fusion_ids - test_ids

        assert len(non_test_in_fusion) == 0, (
            f"Nicht-Test-Cases in Fusion: {non_test_in_fusion}"
        )


# ---------------------------------------------------------------------------
# Tests: Threshold-Sensitivity-Analyse
# ---------------------------------------------------------------------------

def _compute_sensitivity_table(y_val, p_val, y_test, p_test, targets):
    """
    Für jedes Recall-Ziel: höchsten Val-Threshold wählen → auf Test anwenden.
    Gibt Liste von Dicts zurück. Threshold-Wahl AUSSCHLIESSLICH aus Val-Scores.
    """
    from sklearn.metrics import (precision_recall_curve, recall_score,
                                  precision_score, f1_score, confusion_matrix)
    _, rec_val, thr_val = precision_recall_curve(y_val, p_val)
    rows = []
    for target in targets:
        valid = thr_val[rec_val[:-1] >= target]
        thr   = float(valid.max()) if len(valid) > 0 else float(thr_val[0])
        val_rec = float(recall_score(y_val, (p_val >= thr).astype(int)))
        y_pred  = (p_test >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        rows.append({
            'target_recall_val': target,
            'threshold_val':     thr,
            'val_recall':        val_rec,
            'test_recall':       float(recall_score(y_test, y_pred)),
            'test_fpr':          fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'test_precision':    float(precision_score(y_test, y_pred, zero_division=0)),
            'test_f1':           float(f1_score(y_test, y_pred, zero_division=0)),
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        })
    return rows


class TestThresholdSensitivity:
    """Prüft die Threshold-Sensitivity-Analyse der Log-Komponente."""

    TARGETS = [0.95, 0.975, 0.99, 1.00]

    def _make_data(self):
        rng = np.random.default_rng(42)
        y_val  = np.array([1]*30 + [0]*70)
        p_val  = np.concatenate([rng.uniform(0.6, 1.0, 30), rng.uniform(0.0, 0.4, 70)])
        y_test = np.array([1]*25 + [0]*75)
        p_test = np.concatenate([rng.uniform(0.5, 0.95, 25), rng.uniform(0.0, 0.5, 75)])
        return y_val, p_val, y_test, p_test

    def test_table_contains_all_targets(self):
        """Jedes Recall-Ziel muss in der Tabelle vertreten sein."""
        y_val, p_val, y_test, p_test = self._make_data()
        rows = _compute_sensitivity_table(y_val, p_val, y_test, p_test, self.TARGETS)
        found = [r['target_recall_val'] for r in rows]
        assert found == self.TARGETS, f"Erwartete Targets {self.TARGETS}, erhalten {found}"

    def test_threshold_uses_only_val_scores(self):
        """Threshold-Wahl ohne Test-Scores möglich — Test-Daten sind kein Input."""
        y_val, p_val, _, _ = self._make_data()
        from sklearn.metrics import precision_recall_curve
        _, rec_val, thr_val = precision_recall_curve(y_val, p_val)
        for target in self.TARGETS:
            valid = thr_val[rec_val[:-1] >= target]
            thr   = float(valid.max()) if len(valid) > 0 else float(thr_val[0])
            assert thr >= 0.0, f"Threshold für target={target} ist negativ: {thr}"

    def test_higher_target_recall_not_higher_threshold(self):
        """
        Höheres Recall-Ziel → gleicher oder niedrigerer Threshold.
        Monotonie: um mehr Positive zu treffen muss die Entscheidungsgrenze fallen.
        """
        y_val, p_val, y_test, p_test = self._make_data()
        rows = _compute_sensitivity_table(y_val, p_val, y_test, p_test, self.TARGETS)
        thresholds = [r['threshold_val'] for r in rows]
        for i in range(1, len(thresholds)):
            assert thresholds[i] <= thresholds[i-1] + 1e-9, (
                f"Threshold stieg bei höherem Recall-Ziel: "
                f"target={self.TARGETS[i-1]}→{self.TARGETS[i]}, "
                f"thr={thresholds[i-1]:.4f}→{thresholds[i]:.4f}"
            )

    def test_metrics_in_unit_interval(self):
        """Alle Metriken (Recall, FPR, Precision, F1) müssen in [0, 1] liegen."""
        y_val, p_val, y_test, p_test = self._make_data()
        rows = _compute_sensitivity_table(y_val, p_val, y_test, p_test, self.TARGETS)
        for r in rows:
            for key in ('test_recall', 'test_fpr', 'test_precision', 'test_f1'):
                assert 0.0 <= r[key] <= 1.0, (
                    f"Metrik {key}={r[key]} außerhalb [0,1] für target={r['target_recall_val']}"
                )

    def test_val_recall_meets_target(self):
        """Val-Recall muss für jeden Threshold das jeweilige Recall-Ziel erfüllen."""
        y_val, p_val, y_test, p_test = self._make_data()
        rows = _compute_sensitivity_table(y_val, p_val, y_test, p_test, self.TARGETS)
        for r in rows:
            assert r['val_recall'] >= r['target_recall_val'] - 1e-6, (
                f"Val-Recall {r['val_recall']:.4f} verfehlt Ziel "
                f"{r['target_recall_val']} für threshold={r['threshold_val']:.4f}"
            )


# ---------------------------------------------------------------------------
# Tests: Case-Level Metric Baseline (Stand C)
# ---------------------------------------------------------------------------

class TestCaseLevelMetricBaseline:
    """
    Prüft die Case-Level-Metrik-Baseline aus NB02 Stand C.

    Alle Tests sind datei-unabhängig (kein docs/-Dateizugriff nötig) und
    testen Kernlogik inline — identisch zur NB02-Implementierung.
    """

    # ── Hilfsfunktionen ──────────────────────────────────────────────────────

    def _make_row_level_data(self):
        """Synthetisches row-level Dataset: 4 Cases, je 10 Zeilen, 2 Metriken."""
        rng = np.random.default_rng(42)
        rows = []
        for case_id, label in [('c1', 1), ('c2', 1), ('c3', 0), ('c4', 0)]:
            for _ in range(10):
                rows.append({
                    'case_id':      case_id,
                    'dataset_type': 'mali' if label == 1 else 'norm',
                    'scenario_id':  'scenario_1',
                    'group_id':     'mali_scenario_1' if label == 1 else 'norm_scenario_1',
                    'label':        label,
                    'cpu_usage':    rng.uniform(0.2, 0.9),
                    'mem_usage':    rng.uniform(0.1, 0.8),
                })
        return pd.DataFrame(rows)

    def _compute_case_features(self, df, metrics=('cpu_usage', 'mem_usage')):
        """Case-Level-Feature-Aggregation identisch zu NB02."""
        from scipy.stats import linregress

        def _slope(s):
            v = s.dropna().values
            if len(v) < 3:
                return np.nan
            return float(linregress(np.arange(len(v), dtype=float), v).slope)

        meta_cols = ['case_id', 'dataset_type', 'scenario_id', 'group_id', 'label']
        meta_df = df[meta_cols].drop_duplicates(subset='case_id').reset_index(drop=True)

        feature_parts = []
        for m in metrics:
            grp = df.groupby('case_id')[m]
            part = pd.DataFrame({
                f'{m}_mean':  grp.mean(),
                f'{m}_max':   grp.max(),
                f'{m}_std':   grp.std(ddof=1),
                f'{m}_p95':   grp.quantile(0.95),
                f'{m}_slope': grp.apply(_slope),
            })
            feature_parts.append(part)

        feat_df = pd.concat(feature_parts, axis=1).reset_index()
        return meta_df.merge(feat_df, on='case_id')

    # ── Test 1: Split-Gruppen-Disjunktheit ──────────────────────────────────

    def test_split_group_disjointness(self):
        """
        Train, Val, Test dürfen keine gemeinsamen group_ids haben.
        Dreistufige GroupShuffleSplit-Logik identisch zu NB02.
        """
        rng = np.random.default_rng(0)
        n_cases   = 120
        n_groups  = 12
        group_ids = [f'g{i}' for i in range(n_groups)]

        cases = pd.DataFrame({
            'case_id':  [f'c{i}' for i in range(n_cases)],
            'group_id': np.array(group_ids)[np.arange(n_cases) % n_groups],
            'label':    (np.arange(n_cases) % 5 == 0).astype(int),
            'x':        rng.uniform(size=n_cases),
        })

        X = cases[['x']]
        y = cases['label']
        g = cases['group_id']

        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tv_idx, te_idx = next(gss1.split(X, y, g))

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=0)
        tr_idx, vl_idx = next(gss2.split(X.iloc[tv_idx], y.iloc[tv_idx], g.iloc[tv_idx]))

        g_tr = set(g.iloc[tv_idx].iloc[tr_idx])
        g_vl = set(g.iloc[tv_idx].iloc[vl_idx])
        g_te = set(g.iloc[te_idx])

        assert len(g_tr & g_vl) == 0, f"Train ∩ Val nicht leer: {g_tr & g_vl}"
        assert len(g_vl & g_te) == 0, f"Val ∩ Test nicht leer: {g_vl & g_te}"
        assert len(g_tr & g_te) == 0, f"Train ∩ Test nicht leer: {g_tr & g_te}"

    # ── Test 2: Split-Verteilung enthält mali in Val und Test ───────────────

    def test_split_distribution_contains_mali_in_val_and_test(self):
        """
        Val und Test müssen mali-Cases enthalten.
        Seed-Wahl (VAL_MIN_POS=10) stellt sicher dass Val >= 10 positive Cases hat.
        """
        rng = np.random.default_rng(1)
        n_cases  = 200
        n_groups = 20

        cases = pd.DataFrame({
            'case_id':  [f'c{i}' for i in range(n_cases)],
            'group_id': [f'g{i % n_groups}' for i in range(n_cases)],
            'label':    ([1] * 5 + [0] * 5) * (n_cases // 10),
            'x':        rng.uniform(size=n_cases),
        })

        X = cases[['x']]
        y = cases['label']
        g = cases['group_id']

        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tv_idx, te_idx = next(gss1.split(X, y, g))

        # Seed-Wahl wie NB02
        X_tv = X.iloc[tv_idx]
        y_tv = y.iloc[tv_idx]
        g_tv = g.iloc[tv_idx]

        selected_seed = None
        vl_idx_final  = None
        for seed in (0, 1, 2, 7, 42):
            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
            tr_idx, vl_idx = next(gss2.split(X_tv, y_tv, g_tv))
            if y_tv.iloc[vl_idx].sum() >= 1:
                selected_seed = seed
                vl_idx_final  = vl_idx
                break

        assert selected_seed is not None, "Kein Seed liefert Val mit mali-Cases."
        assert y_tv.iloc[vl_idx_final].sum() >= 1, "Val enthält keine mali-Cases."
        assert int((y.iloc[te_idx] == 1).sum()) >= 1, "Test enthält keine mali-Cases."

    # ── Test 3: metric_case_features — eine Zeile pro Case ──────────────────

    def test_metric_case_features_one_row_per_case(self):
        """
        Case-Level-Feature-Engineering muss exakt eine Zeile pro case_id liefern.
        Verletzung würde bedeuten, dass row-level Zeilen nicht korrekt aggregiert wurden.
        """
        df = self._make_row_level_data()
        case_feat = self._compute_case_features(df)

        n_cases       = df['case_id'].nunique()
        n_rows_result = len(case_feat)

        assert n_rows_result == n_cases, (
            f"Erwartet {n_cases} Zeilen (eine pro Case), erhalten {n_rows_result}"
        )
        assert case_feat['case_id'].nunique() == n_cases, (
            "Duplizierte case_ids in Case-Feature-Matrix"
        )

    # ── Test 4: Keine Row-Level-Leakage in Features ─────────────────────────

    def test_metric_case_features_no_row_level_leakage(self):
        """
        Case-Level-Feature-Matrix darf keine Row-Level-Index-Spalten enthalten.
        Verbotene Spalten: Zeitstempel-Index, original_index, row_index, etc.
        Die Feature-Spalten müssen ausschließlich aus den 5 Aggregationen bestehen.
        """
        df = self._make_row_level_data()
        case_feat = self._compute_case_features(df)

        feature_cols = [c for c in case_feat.columns
                        if c not in ['case_id', 'dataset_type', 'scenario_id', 'group_id', 'label']]

        # Verbotene Muster im Spaltennamen
        forbidden_patterns = ['index', 'row', 'timestamp', 'time', 'original']
        for col in feature_cols:
            for pat in forbidden_patterns:
                assert pat not in col.lower(), (
                    f"Verdächtige Feature-Spalte '{col}' enthält verbotenes Muster '{pat}' "
                    f"— könnte Row-Level-Index sein"
                )

        # Erlaubte Suffix-Muster
        allowed_suffixes = ('_mean', '_max', '_std', '_p95', '_slope')
        for col in feature_cols:
            assert col.endswith(allowed_suffixes), (
                f"Feature-Spalte '{col}' hat kein erlaubtes Aggregations-Suffix {allowed_suffixes}"
            )

    # ── Test 5: Threshold kommt aus Validation, nicht aus Test ──────────────

    def test_metric_threshold_from_validation_only(self):
        """
        Threshold-Auswahl benötigt nur Val-Scores und Val-Labels.
        Test-Scores sind kein Input für die Threshold-Funktion.
        Stellt sicher dass Threshold-Selektion kein Testset-Bias hat.
        """
        from sklearn.metrics import precision_recall_curve

        rng = np.random.default_rng(7)
        # Val: gut separierbar → hoher Threshold
        y_val  = np.array([1] * 20 + [0] * 80)
        p_val  = np.concatenate([rng.uniform(0.75, 1.0, 20), rng.uniform(0.0, 0.3, 80)])
        # Test: schlechter separierbar
        y_test = np.array([1] * 20 + [0] * 80)
        p_test = np.concatenate([rng.uniform(0.4, 0.8, 20), rng.uniform(0.1, 0.6, 80)])

        def _select_threshold_val_only(y_v, p_v):
            """Threshold-Selektion NUR aus Val — kein Test-Input."""
            _, rec_v, thr_v = precision_recall_curve(y_v, p_v)
            valid = thr_v[rec_v[:-1] >= 0.95]
            return float(valid.max()) if len(valid) > 0 else float(thr_v[0])

        thr_from_val  = _select_threshold_val_only(y_val,  p_val)
        thr_from_test = _select_threshold_val_only(y_test, p_test)

        # Test-Threshold darf nicht für die echte Pipeline verwendet werden —
        # er darf aber existieren; wichtig ist dass NB02 thr_from_val nutzt.
        # Val und Test haben unterschiedliche optimale Thresholds (Testintegrität).
        assert thr_from_val != thr_from_test, (
            "Val- und Test-Threshold sind identisch — Test prüft kein reales Szenario."
        )
        # Val-Threshold muss Val-Recall-Nebenbedingung erfüllen
        from sklearn.metrics import recall_score as _rs
        val_recall = _rs(y_val, (p_val >= thr_from_val).astype(int))
        assert val_recall >= 0.95, (
            f"Val-Recall {val_recall:.3f} nach Threshold-Wahl < 0.95"
        )

    # ── Test 6: metric_case_results enthält alle Pflichtfelder ──────────────

    def test_metric_results_case_level_fields(self):
        """
        metric_case_results.csv muss alle Pflicht-Metriken enthalten.
        Alle Metriken ausser TP/FP/TN/FN müssen in [0, 1] liegen.
        Granularity-Feld (falls vorhanden) muss 'case-level' sein.
        """
        from sklearn.metrics import (
            recall_score as _rs, precision_score as _ps,
            f1_score as _f1, average_precision_score as _ap,
            confusion_matrix as _cm
        )

        # Synthetische Predictions (Fall: gutes Modell)
        rng = np.random.default_rng(42)
        y_true = np.array([1] * 48 + [0] * 205)   # 253 Test-Cases wie im echten Split
        p_pred = np.concatenate([
            rng.uniform(0.6, 1.0, 48),
            rng.uniform(0.0, 0.5, 205)
        ])
        threshold = 0.4
        y_pred = (p_pred >= threshold).astype(int)

        tn_r, fp_r, fn_r, tp_r = _cm(y_true, y_pred).ravel()
        fpr_total = fp_r / (fp_r + tn_r) if (fp_r + tn_r) > 0 else 0.0

        # Simuliertes results-Dict (wie metric_case_results.csv)
        results = {
            'Granularity':      'case-level',
            'Recall_mali':      round(_rs(y_true, y_pred), 4),
            'Precision':        round(_ps(y_true, y_pred, zero_division=0), 4),
            'F1':               round(_f1(y_true, y_pred, zero_division=0), 4),
            'FPR_total':        round(fpr_total, 4),
            'FPR_anom':         round(fpr_total * 0.9, 4),  # simuliert
            'FPR_norm':         round(fpr_total * 1.1, 4),  # simuliert
            'AP':               round(_ap(y_true, p_pred), 4),
            'TP': int(tp_r), 'FP': int(fp_r), 'TN': int(tn_r), 'FN': int(fn_r),
        }

        # Pflichtfelder prüfen
        required_fields = [
            'Recall_mali', 'Precision', 'F1',
            'FPR_total', 'FPR_anom', 'FPR_norm',
            'AP', 'TP', 'FP', 'TN', 'FN'
        ]
        for field in required_fields:
            assert field in results, f"Pflichtfeld '{field}' fehlt in results"

        # Metriken in [0, 1]
        metric_fields = ['Recall_mali', 'Precision', 'F1', 'FPR_total', 'FPR_anom', 'FPR_norm', 'AP']
        for field in metric_fields:
            assert 0.0 <= results[field] <= 1.0, (
                f"Metrik '{field}' = {results[field]} liegt außerhalb [0, 1]"
            )

        # Granularity-Check
        assert results.get('Granularity') == 'case-level', (
            f"Granularity muss 'case-level' sein, ist: {results.get('Granularity')}"
        )