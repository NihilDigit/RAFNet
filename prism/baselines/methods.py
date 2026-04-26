from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class MethodSpec:
    name: str
    description: str


METHOD_REGISTRY: Dict[str, MethodSpec] = {
    "convnext_logreg": MethodSpec("convnext_logreg", "ConvNeXt-only Logistic Regression"),
    "grouprec_logreg": MethodSpec("grouprec_logreg", "GroupRec-only Logistic Regression"),
    "convnext_linear_svm": MethodSpec("convnext_linear_svm", "ConvNeXt-only Linear SVM (calibrated)"),
    "grouprec_linear_svm": MethodSpec("grouprec_linear_svm", "GroupRec-only Linear SVM (calibrated)"),
    "concat_logreg": MethodSpec("concat_logreg", "Early-fusion concat Logistic Regression"),
    "concat_linear_svm": MethodSpec("concat_linear_svm", "Early-fusion concat Linear SVM (calibrated)"),
    "concat_mlp": MethodSpec("concat_mlp", "Early-fusion concat MLP"),
    "late_fusion_logreg": MethodSpec("late_fusion_logreg", "Late-fusion average of two Logistic heads"),
    "lmf_logreg": MethodSpec("lmf_logreg", "Low-rank bilinear fusion (LMF-style) + Logistic Regression"),
    "xgboost_concat": MethodSpec("xgboost_concat", "XGBoost multi-class on concatenated features (optional)"),
}


def _concat(x: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([x["grouprec"], x["convnext"]], axis=1)


def _make_logreg(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
            multi_class="auto",
        ),
    )


def _make_linear_svm(seed: int):
    base = make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", random_state=seed, dual="auto"),
    )
    return CalibratedClassifierCV(base, cv=3, method="sigmoid")


def _make_mlp(seed: int):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=400,
            random_state=seed,
        ),
    )


def _lmf_features(x_gr: np.ndarray, x_cn: np.ndarray, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w_gr = rng.normal(0, 1.0 / np.sqrt(x_gr.shape[1]), size=(x_gr.shape[1], rank)).astype(np.float32)
    w_cn = rng.normal(0, 1.0 / np.sqrt(x_cn.shape[1]), size=(x_cn.shape[1], rank)).astype(np.float32)
    z_gr = x_gr @ w_gr
    z_cn = x_cn @ w_cn
    return z_gr * z_cn


def _predict_with_model(model, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]:
    pred = model.predict(x_test)
    prob = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None
    return pred, prob


def run_method(
    method: str,
    x_train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    x_test: Dict[str, np.ndarray],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray | None, str | None]:
    """Run one method and return (y_pred, y_prob, skip_reason)."""

    if method == "convnext_logreg":
        m = _make_logreg(seed)
        m.fit(x_train["convnext"], y_train)
        return _predict_with_model(m, x_test["convnext"]) + (None,)

    if method == "grouprec_logreg":
        m = _make_logreg(seed)
        m.fit(x_train["grouprec"], y_train)
        return _predict_with_model(m, x_test["grouprec"]) + (None,)

    if method == "convnext_linear_svm":
        m = _make_linear_svm(seed)
        m.fit(x_train["convnext"], y_train)
        return _predict_with_model(m, x_test["convnext"]) + (None,)

    if method == "grouprec_linear_svm":
        m = _make_linear_svm(seed)
        m.fit(x_train["grouprec"], y_train)
        return _predict_with_model(m, x_test["grouprec"]) + (None,)

    if method == "concat_logreg":
        m = _make_logreg(seed)
        m.fit(_concat(x_train), y_train)
        return _predict_with_model(m, _concat(x_test)) + (None,)

    if method == "concat_linear_svm":
        m = _make_linear_svm(seed)
        m.fit(_concat(x_train), y_train)
        return _predict_with_model(m, _concat(x_test)) + (None,)

    if method == "concat_mlp":
        m = _make_mlp(seed)
        m.fit(_concat(x_train), y_train)
        return _predict_with_model(m, _concat(x_test)) + (None,)

    if method == "late_fusion_logreg":
        m_gr = _make_logreg(seed)
        m_cn = _make_logreg(seed + 1000)
        m_gr.fit(x_train["grouprec"], y_train)
        m_cn.fit(x_train["convnext"], y_train)

        p_gr = m_gr.predict_proba(x_test["grouprec"])
        p_cn = m_cn.predict_proba(x_test["convnext"])
        p = 0.5 * p_gr + 0.5 * p_cn
        y_pred = p.argmax(axis=1)
        return y_pred, p, None

    if method == "lmf_logreg":
        z_train = _lmf_features(x_train["grouprec"], x_train["convnext"], rank=256, seed=seed)
        z_test = _lmf_features(x_test["grouprec"], x_test["convnext"], rank=256, seed=seed)
        m = _make_logreg(seed)
        m.fit(z_train, y_train)
        return _predict_with_model(m, z_test) + (None,)

    if method == "xgboost_concat":
        try:
            from xgboost import XGBClassifier
        except Exception:
            return np.array([]), None, "xgboost_not_installed"

        m = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=4,
        )
        m.fit(_concat(x_train), y_train)
        return _predict_with_model(m, _concat(x_test)) + (None,)

    raise KeyError(f"Unknown method: {method}")
