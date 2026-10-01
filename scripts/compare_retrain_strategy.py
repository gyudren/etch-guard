"""재학습 방식 비교: warm start fine-tuning vs 같은 7일 데이터로 처음부터(scratch) 학습.

python scripts/compare_retrain_strategy.py  →  evidence/06_retrain_strategy.json
드리프트 감지 시점(2025-07-27 13:30)까지 최근 7일·5대 장비 데이터만 사용하고,
평가는 ① 최근 홀드아웃 ② 그 다음 1주(미사용 미래) ③ 고정 정상 기준셋(망각 여부) 세 가지다.
"""
import bisect
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from semiconductor import train
from semiconductor.config import DEFAULT_DATA, STATE, SEQ_LEN, FINE_TUNE_EPOCHS, FINE_TUNE_LR
from semiconductor.data import read_data, load_scaler, transform, windows

UNTIL = datetime(2025, 7, 27, 13, 30)
NEXT = datetime(2025, 7, 27, 14, 0)


def next_week(series, scaler):
    xs, ys = [], []
    for g in series.values():
        i = bisect.bisect_left(g["time"], NEXT)
        x = transform(g["x"][:i + 336], scaler)
        xs.append(windows(x, i, i + 336)); ys.append(g["y"][i:i + 336])
    return np.concatenate(xs), np.concatenate(ys)


def evaluate(model, sets):
    return {name: round(train.metrics(y, train.predict_scores(model, x))["rmse"], 3) for name, (x, y) in sets.items()}


def main():
    from tensorflow import keras
    series = read_data(DEFAULT_DATA)
    base = STATE / "bundles" / json.loads((STATE / "local.json").read_text())["bundle"]
    scaler = load_scaler(base / "scaler.json")
    data, _ = train._recent_samples(series, scaler, UNTIL)
    sets = {"recent_holdout": (data["x_hold"], data["y_hold"]), "next_week_unseen": next_week(series, scaler),
            "golden_normal": train._golden_normal(series, scaler)}
    results = {"train_samples": int(len(data["y_train"])), "until": UNTIL.isoformat()}
    champion = keras.models.load_model(base / "model.keras", compile=False)
    results["production_v1_no_retrain"] = evaluate(champion, sets)
    for name, lr, epochs, warm in (("warm_start_finetune", FINE_TUNE_LR, FINE_TUNE_EPOCHS, True),
                                   ("scratch_same_data", 1e-3, 30, False)):
        runs = []
        for seed in (42, 7, 2025):
            keras.utils.set_random_seed(seed)
            model = keras.models.load_model(base / "model.keras", compile=False) if warm else train.build_model()
            model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
            model.fit(data["x_train"], data["y_train"] / 100, epochs=epochs, batch_size=64, verbose=0)
            runs.append(evaluate(model, sets))
        results[name] = {"lr": lr, "epochs": epochs, "seeds": runs,
                         "mean": {k: round(float(np.mean([r[k] for r in runs])), 3) for k in sets},
                         "std": {k: round(float(np.std([r[k] for r in runs])), 3) for k in sets}}
    Path("evidence").mkdir(exist_ok=True)
    Path("evidence/06_retrain_strategy.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
