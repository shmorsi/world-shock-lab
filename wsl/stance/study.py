"""Policy stance: LoRA fine-tuning against a bag-of-words baseline (config/stance_protocol.yaml).

The data are the FOMC sentences labelled dovish, hawkish or neutral by Shah, Paturi and Chava
(ACL 2023), on the authors' own train/test split. 20% of train is held out for development:
every choice (the baseline's C, the adapter's learning rate, the epoch kept) is made there.
The test split is scored once, by `wsl stance evaluate --split test`, and only on a commit whose
history contains the `stance-v1` tag with the protocol unchanged since.

LoRA (Hu et al. 2021) freezes the pretrained weights W and learns a low-rank update,
W + (alpha / r) B A with A in R^{r x d} and B in R^{d x r}, here on the attention query and value
projections of every layer, plus the classification head: about 0.7% of the parameters train.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import subprocess
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from wsl.paths import ROOT

PROTOCOL = ROOT / "config" / "stance_protocol.yaml"
RESULTS = ROOT / "results" / "stance"
TAG = "stance-v1"
LABELS = ["dovish", "hawkish", "neutral"]
Log = Callable[[str], object]


class GateError(RuntimeError):
    pass


def protocol() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(PROTOCOL.read_text())
    return data


# ------------------------------------------------------------------ data
def load_split(name: str) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    d = protocol()["data"]
    path = hf_hub_download(
        d["source"], d["files"][name], repo_type="dataset", revision=d["revision"]
    )
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df[["sentence", "label", "year"]].reset_index(drop=True)


def dev_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    tr, dev = train_test_split(train, test_size=0.2, stratify=train["label"], random_state=0)
    return tr.reset_index(drop=True), dev.reset_index(drop=True)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    return {
        "macro_f1": float(f1_score(y, p, average="macro")),
        "weighted_f1": float(f1_score(y, p, average="weighted")),
        "accuracy": float(accuracy_score(y, p)),
        "f1_by_class": dict(
            zip(
                LABELS,
                f1_score(y, p, average=None, labels=[0, 1, 2]).round(4).tolist(),
                strict=True,
            )
        ),
        "confusion": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
    }


# ---------------------------------------------------------------- models
def fit_tfidf(tr: pd.DataFrame, C: float) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, lowercase=True),
        LogisticRegression(C=C, class_weight="balanced", max_iter=5000),
    )
    return pipe.fit(tr["sentence"], tr["label"])


class LoraClassifier:
    """roberta-base with LoRA adapters on query and value, trained by a plain PyTorch loop."""

    def __init__(self, lr: float, spec: dict[str, Any], seed: int = 0):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(seed)
        torch.set_num_threads(4)
        self.spec, self.lr = spec, lr
        self.tok = AutoTokenizer.from_pretrained(spec["base"], revision=spec["base_revision"])
        base = AutoModelForSequenceClassification.from_pretrained(
            spec["base"], revision=spec["base_revision"], num_labels=3
        )
        a = spec["adapters"]
        cfg = LoraConfig(
            r=a["r"],
            lora_alpha=a["alpha"],
            lora_dropout=a["dropout"],
            target_modules=a["target"],
            task_type="SEQ_CLS",
        )
        self.model = get_peft_model(base, cfg)
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.model.parameters())
        self.trainable = {"trainable": int(n_train), "total": int(n_all), "share": n_train / n_all}

    def _batches(
        self,
        texts: list[str],
        labels: list[int] | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterator[Any]:
        import torch

        idx = np.arange(len(texts))
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        bs = self.spec["batch"]
        for i in range(0, len(idx), bs):
            j = idx[i : i + bs]
            enc = self.tok(
                [texts[k] for k in j],
                padding=True,
                truncation=True,
                max_length=self.spec["max_tokens"],
                return_tensors="pt",
            )
            if labels is not None:
                enc["labels"] = torch.tensor([labels[k] for k in j])
            yield enc

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        import torch

        self.model.eval()
        out = []
        with torch.no_grad():
            for enc in self._batches(list(texts)):
                out.append(torch.softmax(self.model(**enc).logits, dim=-1).numpy())
        return np.concatenate(out)

    def fit(self, tr: pd.DataFrame, dev: pd.DataFrame, log: Log = print) -> dict[str, Any]:
        """Train for the protocol's epochs; keep the epoch with the best dev macro-F1."""
        import copy

        import torch
        from transformers import get_linear_schedule_with_warmup

        texts, labels = tr["sentence"].tolist(), tr["label"].tolist()
        steps = int(np.ceil(len(texts) / self.spec["batch"])) * self.spec["epochs"]
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)  # type: ignore[no-untyped-call]
        history: list[dict[str, float]] = []
        best = -1.0
        best_state: dict[str, Any] | None = None
        t0 = time.time()
        for ep in range(1, self.spec["epochs"] + 1):
            self.model.train()
            loss_sum, n = 0.0, 0
            for enc in self._batches(texts, labels, shuffle=True, seed=ep):
                loss = self.model(**enc).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                loss_sum += float(loss) * len(enc["labels"])
                n += len(enc["labels"])
            p = self.predict_proba(dev["sentence"].tolist()).argmax(1)
            m = metrics(dev["label"].to_numpy(), p)
            history.append(
                {
                    "epoch": ep,
                    "train_loss": loss_sum / n,
                    "dev_macro_f1": m["macro_f1"],
                    "seconds": time.time() - t0,
                }
            )
            log(
                f"lr {self.lr:g} epoch {ep}: loss {loss_sum / n:.4f}, "
                f"dev macro-F1 {m['macro_f1']:.4f}"
            )
            if m["macro_f1"] > best:
                best = m["macro_f1"]
                best_state = copy.deepcopy(
                    {
                        k: v.detach().clone()
                        for k, v in self.model.state_dict().items()
                        if "lora" in k or "classifier" in k
                    }
                )
        assert best_state is not None
        self.model.load_state_dict(best_state, strict=False)
        best_epoch = max(history, key=lambda h: h["dev_macro_f1"])["epoch"]
        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_dev_macro_f1": best,
            "seconds": time.time() - t0,
        }

    def save(self, path: Path) -> None:
        self.model.save_pretrained(str(path))

    @classmethod
    def load(cls, path: Path, lr: float, spec: dict[str, Any]) -> LoraClassifier:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        obj = cls(lr, spec)
        set_peft_model_state_dict(obj.model, load_file(str(path / "adapter_model.safetensors")))
        return obj


# ------------------------------------------------------------- the gate
def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def check_gate() -> None:
    if _git("rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}").returncode != 0:
        raise GateError(f"tag {TAG} not found; fetch tags (git fetch --tags) first")
    if _git("merge-base", "--is-ancestor", TAG, "HEAD").returncode != 0:
        raise GateError(f"{TAG} is not in this commit's history; the test split stays closed")
    if _git("diff", "--quiet", TAG, "--", str(PROTOCOL.relative_to(ROOT))).returncode != 0:
        raise GateError(f"the protocol differs from {TAG}; the test split stays closed")
    if (RESULTS / "test.json").exists():
        raise GateError("results/stance/test.json exists: the test split was scored once already")


def provenance() -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git("rev-parse", "HEAD").stdout.strip(),
        "protocol_tag": TAG,
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "packages": {
            p: md.version(p) for p in ("torch", "transformers", "peft", "scikit-learn", "numpy")
        },
    }


def bootstrap_diff(
    y: np.ndarray,
    pa: np.ndarray,
    pb: np.ndarray,
    draws: int = 10000,
    seed: int = 0,
    level: float = 0.90,
) -> dict[str, float]:
    """Paired bootstrap of macro-F1(a) - macro-F1(b) over sentences."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(draws)
    for k in range(draws):
        i = rng.integers(0, n, n)
        d[k] = f1_score(y[i], pa[i], average="macro") - f1_score(y[i], pb[i], average="macro")
    lo, hi = np.quantile(d, [(1 - level) / 2, 1 - (1 - level) / 2])
    diff = f1_score(y, pa, average="macro") - f1_score(y, pb, average="macro")
    return {"difference": float(diff), "lo": float(lo), "hi": float(hi)}


def verdict(ci: dict[str, float]) -> str:
    return "helps" if ci["lo"] > 0 else ("hurts" if ci["hi"] < 0 else "no detectable difference")


# ------------------------------------------------------------ the stages
def run_dev(log: Log = print) -> dict[str, Any]:
    """Every open choice, made on the development split; saves the selected adapter."""
    spec = protocol()["models"]
    train = load_split("train")
    tr, dev = dev_split(train)
    y = dev["label"].to_numpy()
    out: dict[str, Any] = {
        "n_train": len(tr),
        "n_dev": len(dev),
        "label_counts_train": tr["label"].value_counts().sort_index().tolist(),
    }
    majority = int(tr["label"].mode()[0])
    out["majority"] = {"label": LABELS[majority], **metrics(y, np.full(len(y), majority))}
    grid = {}
    for C in spec["tfidf_logreg"]["choose_on_dev"]["C"]:
        grid[str(C)] = metrics(y, fit_tfidf(tr, C).predict(dev["sentence"]))["macro_f1"]
    best_C = float(max(grid, key=lambda k: grid[k]))
    out["tfidf_logreg"] = {
        "grid": grid,
        "C": best_C,
        **metrics(y, fit_tfidf(tr, best_C).predict(dev["sentence"])),
    }
    lora = {}
    best: tuple[float, dict[str, Any], LoraClassifier] | None = None
    for lr in spec["lora_roberta"]["choose_on_dev"]["learning_rate"]:
        clf = LoraClassifier(lr, spec["lora_roberta"], seed=spec["lora_roberta"]["seed"])
        fit = clf.fit(tr, dev, log)
        lora[str(lr)] = fit
        if best is None or fit["best_dev_macro_f1"] > best[1]["best_dev_macro_f1"]:
            best = (lr, fit, clf)
    assert best is not None
    lr, fit, clf = best
    adapter = ROOT / "data" / "raw" / "stance" / "adapter"
    clf.save(adapter)
    out["lora_roberta"] = {
        "runs": lora,
        "learning_rate": lr,
        "epoch": fit["best_epoch"],
        "parameters": clf.trainable,
        **metrics(y, clf.predict_proba(dev["sentence"].tolist()).argmax(1)),
    }
    out["provenance"] = provenance()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "dev.json").write_text(json.dumps(out, indent=1))
    return out


def run_test(log: Log = print) -> dict[str, Any]:
    check_gate()
    spec = protocol()["models"]
    dev = json.loads((RESULTS / "dev.json").read_text())
    train = load_split("train")
    tr, _ = dev_split(train)
    test = load_split("test")
    y = test["label"].to_numpy()
    majority = int(tr["label"].mode()[0])
    p_major = np.full(len(y), majority)
    p_tfidf = fit_tfidf(tr, dev["tfidf_logreg"]["C"]).predict(test["sentence"])
    clf = LoraClassifier.load(
        ROOT / "data" / "raw" / "stance" / "adapter",
        dev["lora_roberta"]["learning_rate"],
        spec["lora_roberta"],
    )
    t0 = time.time()
    p_lora = clf.predict_proba(test["sentence"].tolist()).argmax(1)
    secs = time.time() - t0
    ci = bootstrap_diff(y, p_lora, p_tfidf)
    out: dict[str, Any] = {
        "n_test": len(y),
        "majority": metrics(y, p_major),
        "tfidf_logreg": metrics(y, p_tfidf),
        "lora_roberta": metrics(y, p_lora),
        "primary": {**ci, "verdict": verdict(ci)},
        "lora_cpu_ms_per_sentence": 1000 * secs / len(y),
        "provenance": provenance(),
    }
    (RESULTS / "test.json").write_text(json.dumps(out, indent=1))
    log(
        f"test macro-F1: LoRA {out['lora_roberta']['macro_f1']:.4f}, "
        f"TF-IDF {out['tfidf_logreg']['macro_f1']:.4f}; difference {ci['difference']:+.4f} "
        f"[{ci['lo']:+.4f}, {ci['hi']:+.4f}]: {out['primary']['verdict']}"
    )
    return out
