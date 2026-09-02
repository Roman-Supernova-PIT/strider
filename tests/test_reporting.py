from strider.reporting import evaluation_view, training_epoch


def test_training_epoch_is_one_readable_line(capsys) -> None:
    record = {
        "epoch": 2,
        "train": {"loss": 1.23456},
        "validation_views": {
            "generated": {"loss": 1.34567},
            "clean": {"loss": 1.11111},
        },
        "selection_score": 1.3,
        "learning_rate": 2e-4,
        "learned_scales": {"shape": 0.45, "temporal": 0.08},
    }

    training_epoch(record, total_epochs=5, seconds=12.4, saved=True)

    output = capsys.readouterr().out.strip()
    assert len(output.splitlines()) == 1
    assert "2/5" in output
    assert "1.2346/1.3000" in output
    assert "s=0.45 t=0.08" in output
    assert "* posterior" in output


def test_training_epoch_reports_each_saved_checkpoint_role(capsys) -> None:
    record = {
        "epoch": 3,
        "train": {"loss": 1.0},
        "validation": {},
        "selection_score": 1.1,
        "learning_rate": 1e-4,
        "learned_scales": {},
    }

    training_epoch(
        record,
        total_epochs=5,
        seconds=10.0,
        saved=True,
        saved_science=True,
        saved_redshift=True,
        saved_macro_redshift=True,
    )

    output = capsys.readouterr().out
    assert "* posterior,science,redshift,macro-z" in output


def test_training_epoch_reports_low_redshift_group_metrics(capsys) -> None:
    rows = []
    for index, name in enumerate(("Ia", "H-rich CC", "other")):
        rows.append(
            {
                "class_name": name,
                "N": 20 - index,
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.747,
                "median_absolute_delta_z": 0.02,
                "population_scatter_delta_z": 0.04,
                "outlier_fraction_abs_delta_z_gt_0_1": 0.1,
            }
        )
    record = {
        "epoch": 1,
        "train": {"loss": 1.2},
        "validation": {
            "z_lt_2_macro_f1_present": 0.72,
            "ia_z_lt_2_precision": 0.8,
            "ia_z_lt_2_recall": 0.7,
            "ia_z_lt_2_f1": 0.747,
            "ia_z_lt_2_median_absolute_delta_z": 0.02,
            "ia_z_lt_2_population_scatter_delta_z": 0.04,
            "source_mean_evidence_sufficiency": 0.7,
            "no_source_mean_evidence_sufficiency": 0.01,
            "z_lt_2_metrics_by_class": rows,
        },
        "selection_score": 1.3,
        "learning_rate": 2e-4,
        "learned_scales": {},
    }

    training_epoch(record, total_epochs=5, seconds=12.4, saved=False)

    output = capsys.readouterr().out
    assert "72.0" in output
    assert "z<2 class" in output
    assert "H-rich CC" in output
    assert "0.0400" in output


def test_source_evaluation_summary_states_direct_redshift_metrics(capsys) -> None:
    report = {
        "N": 100,
        "class_accuracy": 0.81,
        "class_balanced_accuracy_present": 0.75,
        "class_macro_f1_present": 0.70,
        "Ia_precision": 0.9,
        "Ia_recall": 0.8,
        "mean_evidence_sufficiency": 0.72,
        "median_delta_z": -0.002,
        "population_scatter_delta_z": 0.04,
        "median_absolute_delta_z": 0.015,
        "outlier_fraction": 0.06,
        "posterior_68_interval_coverage": 0.67,
        "metrics_by_class": [
            {
                "class_name": "Ia",
                "N": 60,
                "precision": 0.9,
                "recall": 0.8,
                "f1": 0.847,
                "median_delta_z": -0.001,
                "population_scatter_delta_z": 0.03,
                "median_absolute_delta_z": 0.01,
                "outlier_fraction": 0.05,
                "posterior_68_interval_coverage": 0.68,
                "mean_evidence_sufficiency": 0.72,
            }
        ],
    }

    evaluation_view("generated", report, outlier_delta_z=0.1)

    output = capsys.readouterr().out
    assert "clean + noise — 100 objects" in output
    assert "balanced accuracy=75.0%" in output
    assert "macro F1=70.0%" in output
    assert "median delta z=-0.0020" in output
    assert "scatter=0.0400" in output
    assert "68% coverage=67.0%" in output
    assert "Ia" in output
    assert "median |dz|" in output


def test_no_source_summary_states_class_confidence(capsys) -> None:
    report = {
        "N": 20,
        "mean_evidence_sufficiency": 0.1,
        "fraction_sufficient_evidence_above_0_5": 0.0,
        "mean_largest_joint_probability": 0.002,
        "mean_largest_class_probability": 0.12,
        "fraction_largest_class_probability_above_0_5": 0.0,
        "median_absolute_delta_z_to_simulation": 0.5,
        "fraction_within_delta_z_0_1_of_simulation": 0.1,
    }

    evaluation_view("no_source", report, outlier_delta_z=0.1)

    output = capsys.readouterr().out
    assert "largest class=0.120" in output
    assert "largest class>0.5=0.0%" in output
