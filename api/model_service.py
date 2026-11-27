"""Framework-free prediction service: loads the model, predicts, explains and logs.

Kept separate from FastAPI so it can be unit-tested and reused (monitoring, dashboard).
"""
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

logger = logging.getLogger("churn-api")

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
PREDICTION_LOG_PATH = Path(os.environ.get("PREDICTION_LOG_PATH", ROOT / "logs" / "predictions.jsonl"))

# Features whose values only make sense together are replaced jointly when explaining.
LINKED_FEATURES = [("Tenure Months", "Total Charges")]
MIN_IMPACT = 0.005      # ignore factors that move the probability by less than 0.5 points
TOP_FACTORS = 3


def snake(column: str) -> str:
    return column.lower().replace(" ", "_")


class ModelService:
    def __init__(self, model_dir=None, log_path=None, background_rows=50):
        model_dir = Path(model_dir or MODEL_DIR)
        for name in ("model.joblib", "model_metadata.json"):
            if not (model_dir / name).exists():
                raise FileNotFoundError(f"{model_dir / name} not found. Run the training pipeline first.")

        self.model = joblib.load(model_dir / "model.joblib")
        self.metadata = json.loads((model_dir / "model_metadata.json").read_text())
        self.features = self.metadata["features"]
        self.threshold = float(self.metadata["decision_threshold"])
        self.model_name = self.metadata["model_name"]
        self.model_version = self.metadata["model_version"]
        self.column_for = {snake(c): c for c in self.features}
        self.log_path = Path(log_path or PREDICTION_LOG_PATH)
        self._log_lock = threading.Lock()

        bg_path = model_dir / "background_sample.csv"
        self.background = pd.read_csv(bg_path)[self.features].head(background_rows) if bg_path.exists() else None
        if self.background is None:
            logger.warning("background_sample.csv missing: explanations are disabled")

        trained_with = self.metadata.get("library_versions", {}).get("scikit-learn")
        self.sklearn_mismatch = bool(trained_with and trained_with != sklearn.__version__)
        if self.sklearn_mismatch:
            logger.warning("scikit-learn %s used for training, %s installed here",
                           trained_with, sklearn.__version__)

    # ------------------------------------------------------------------ helpers
    def risk_level(self, probability: float) -> str:
        if probability >= self.threshold:
            return "high"
        if probability >= self.threshold / 2:
            return "medium"
        return "low"

    def to_frame(self, records: list[dict]) -> pd.DataFrame:
        rows = []
        for rec in records:
            rec = dict(rec)
            if rec.get("total_charges") is None:
                rec["total_charges"] = rec["tenure_months"] * rec["monthly_charges"]
            missing = set(self.column_for) - set(rec)
            if missing:
                raise ValueError(f"missing fields: {sorted(missing)}")
            rows.append({self.column_for[k]: rec[k] for k in self.column_for})
        return pd.DataFrame(rows, columns=self.features)

    def _explain(self, row: pd.DataFrame, probability: float):
        """Impact of each feature = p(customer) - average p when that feature takes the values of
        typical training customers. Model-agnostic and approximate (not exact SHAP values)."""
        if self.background is None:
            return [], []
        bg = self.background
        m = len(bg)
        linked = {c: group for group in LINKED_FEATURES for c in group if all(g in self.features for g in group)}
        groups, seen = [], set()
        for f in self.features:
            if f in seen:
                continue
            group = linked.get(f, (f,))
            seen.update(group)
            groups.append(group)

        blocks = []
        for group in groups:
            block = row.loc[row.index.repeat(m)].reset_index(drop=True)
            for col in group:
                block[col] = bg[col].to_numpy()
            blocks.append(block)
        probs = self.model.predict_proba(pd.concat(blocks, ignore_index=True))[:, 1]
        avg = probs.reshape(len(groups), m).mean(axis=1)

        factors = []
        for group, mean_p in zip(groups, avg):
            impact = float(probability - mean_p)
            label = " / ".join(group)
            value = " / ".join(str(row.iloc[0][c]) for c in group)
            factors.append({"feature": label, "value": value, "impact": round(impact, 4)})
        risk = sorted((f for f in factors if f["impact"] >= MIN_IMPACT), key=lambda f: -f["impact"])
        protect = sorted((f for f in factors if f["impact"] <= -MIN_IMPACT), key=lambda f: f["impact"])
        return risk[:TOP_FACTORS], protect[:TOP_FACTORS]

    # ------------------------------------------------------------------ public API
    def predict(self, records: list[dict], explain: bool = False, endpoint: str = "predict") -> list[dict]:
        """records: dicts with snake_case keys (+ optional customer_id)."""
        start = time.perf_counter()
        frame = self.to_frame(records)
        probabilities = self.model.predict_proba(frame)[:, 1]

        outputs = []
        for i, (rec, p) in enumerate(zip(records, probabilities)):
            p = float(p)
            out = {
                "customer_id": rec.get("customer_id"),
                "churn_probability": round(p, 4),
                "predicted_churn": p >= self.threshold,
                "risk_level": self.risk_level(p),
                "threshold": round(self.threshold, 4),
                "model_name": self.model_name,
                "model_version": self.model_version,
                "top_risk_factors": [],
                "top_protective_factors": [],
            }
            if explain:
                out["top_risk_factors"], out["top_protective_factors"] = self._explain(frame.iloc[[i]], p)
            outputs.append(out)

        latency_ms = (time.perf_counter() - start) * 1000 / max(len(records), 1)
        self._log(records, outputs, latency_ms, endpoint)
        return outputs

    def _log(self, records, outputs, latency_ms, endpoint):
        """Append one JSON line per prediction. Logging must never break a prediction."""
        try:
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            lines = []
            for rec, out in zip(records, outputs):
                inputs = {k: v for k, v in rec.items() if k != "customer_id"}
                lines.append(json.dumps({
                    "timestamp": now, "endpoint": endpoint, "model_name": self.model_name,
                    "model_version": self.model_version, "customer_id": rec.get("customer_id"),
                    "inputs": inputs, "churn_probability": out["churn_probability"],
                    "predicted_churn": out["predicted_churn"], "risk_level": out["risk_level"],
                    "latency_ms": round(latency_ms, 2),
                }))
            with self._log_lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
        except Exception:   # noqa: BLE001
            logger.exception("could not write prediction log")

    def info(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "trained_at": self.metadata.get("trained_at"),
            "decision_threshold": self.threshold,
            "metrics": self.metadata.get("metrics"),
            "features": self.features,
            "mlflow_run_id": self.metadata.get("mlflow", {}).get("run_id"),
            "git_commit": self.metadata.get("git_commit"),
            "sklearn_version_mismatch": self.sklearn_mismatch,
        }
