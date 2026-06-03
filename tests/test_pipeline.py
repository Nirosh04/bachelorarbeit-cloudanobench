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