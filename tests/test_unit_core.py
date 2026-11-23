"""Focused unit tests for core ZERO-TAX math, OOD guard, SqliteSaver, and benchmark formatting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.checkpointer import SqliteSaver
from core.dataset import generate_dataset
from core.distiller import NodeDistiller, compute_ece
from core.models import IntentLabel, TeacherMode
from core.teacher import LLMTeacher


def test_compute_ece_perfect_and_misaligned() -> None:
    classes = np.array(["A", "B"])
    y_true = np.array(["A", "A", "B", "B"])
    y_proba_perfect = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert compute_ece(y_true, y_proba_perfect, classes, n_bins=10) == 0.0

    y_proba_bad = np.array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1]])
    assert compute_ece(y_true, y_proba_bad, classes, n_bins=10) == pytest.approx(0.9, abs=1e-4)


def test_intent_distiller_ood_zero_vector_uniform_fallback() -> None:
    bundle = generate_dataset(seed=42)
    teacher = LLMTeacher(mode=TeacherMode.SIMULATED, sleep_on_simulated=False, seed=42)
    distiller = NodeDistiller("intent_classifier", seed=42)
    for t in bundle.train_stream[:80]:
        distiller.add_trace(t.ticket_text, teacher.classify_intent(t).prediction, t.ticket_id)
    assert distiller.evaluate_and_fit() is not None
    distiller.approve_promotion()

    _, conf, probs = distiller.predict("zzzzxqqq nonvocabulary token 998877")
    assert conf == pytest.approx(0.25, abs=1e-4)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)


def test_sqlite_saver_fork_and_rewind_errors(tmp_path: Path) -> None:
    saver = SqliteSaver(str(tmp_path / "unit_cp.db"))
    try:
        with pytest.raises(KeyError):
            saver.fork_checkpoint("missing-thread", "missing-cp", "fork-thread")
        with pytest.raises(KeyError):
            saver.rewind_thread("missing-thread", "missing-cp")
    finally:
        saver.close()
