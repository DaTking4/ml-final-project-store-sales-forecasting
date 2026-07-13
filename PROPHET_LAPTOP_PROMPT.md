I'm working in the `ml-final-project-store-sales-forecasting` repo (Walmart Store Sales Forecasting, Kaggle). I need you to get `notebooks/Classical Statistical Time Series/model_experiment_Prophet.ipynb` to run to completion and produce trustworthy results, the same way SARIMA/ARIMA were just fixed and completed on a different machine in this project. Read this whole prompt before touching anything — it front-loads what was already found so you don't have to re-derive it.

## Background: what was wrong with SARIMA/ARIMA, and why it matters here too

`src/arima_utils.py` (shared by the ARIMA and SARIMA notebooks) had three real, separate bugs, found by actually running the code and empirically measuring things (object sizes, timings) rather than trusting the code at face value:

1. **No parallelism in the sweep/fit loops.** They were plain sequential Python `for` loops over ~2,660 Store-Dept series — one CPU core used out of the machine's full count. Fixed by wrapping the per-series work in `joblib.Parallel`.

2. **A silent index-corruption bug in the "fit the final model" function** (`fit_arima_models`): `y.index = pd.DatetimeIndex(y.index)` was converting the DataFrame's row-position integers (e.g. `500000`) directly into **nanosecond** timestamps instead of using the real `ds` dates — collapsing a 150+ week series into a fraction of a millisecond near 1970-01-01. `.asfreq("W-FRI")` then had nothing to resample, so the model that actually gets saved/registered for predictions was silently falling back to "last known value" for almost every series, even though the notebook's own sweep metrics (which use a different, correctly-indexed code path) looked fine. Fixed by building the series with `.set_index("ds")` instead of reassigning a bogus index after the fact.

3. **A real, severe memory bloat problem**, found only after fixing #2 made fitting actually succeed: a single fitted `SARIMAX(1,1,1)(1,1,1,52)` result for a ~160-week series pickled to **~275 MB**. Root cause: with a seasonal period `s=52`, statsmodels' state-space representation folds seasonal differencing into the state vector (pushing the state dimension to ~107), and by default retains the *full per-timestep* filtered/predicted/smoothed covariance history for every week. × ~2,660 series that's ~730 GB — completely infeasible regardless of any parallelization or IPC trickery. Two candidate fixes were tested empirically:
   - `simple_differencing=True` shrank it to ~48MB but discards the first `d + s*D` (=53) weeks of history before fitting, and produced visibly implausible forecasts on a test series (real accuracy cost — **rejected**).
   - `model.ssm.set_conserve_memory(statsmodels.tsa.statespace.kalman_filter.MEMORY_CONSERVE)` shrank it to **~8.4MB (32x smaller)**, uses all the data, and produced a **numerically identical** `.forecast()` to the unreduced object (verified directly, not assumed) — this is the one that got applied.

   Also relevant: returning many of these still-multi-MB fitted objects back through a multiprocessing pipe from concurrent worker processes (before the memory-bloat fix, when objects were 275MB each) caused `BrokenProcessPool: ... MemoryError ... un-serialize`. Fixed independently by having each worker write its fitted model straight to a temp file and hand back only a path string, with the parent process loading them back one at a time from disk instead of receiving them all through IPC at once.

## Does any of this apply to Prophet?

`src/prophet_utils.py` has **the identical bug #2** in `fit_prophet_models`:

```python
# CURRENT (buggy) — around line 164:
grouped = {
    unique_id: group.sort_values("ds")["y"].astype(float)
    for unique_id, group in full_long_df[full_long_df["unique_id"].isin(ids)].groupby("unique_id")
}
```

`_fit_one_final_series` then does `y.index = pd.DatetimeIndex(y.index)` on that wrongly-indexed series, and `fit_prophet_model()` uses `y.index` directly as the `ds` column it hands to Prophet — so the *final* Prophet model (the one that gets registered and would generate real predictions) is being trained on garbage dates near 1970, not the real 2010–2012 weekly dates. This wasn't caught before because the per-config sweep evaluation (`evaluate_prophet_config`) builds its `train_by_id` series correctly elsewhere (in the notebook itself, via `.set_index("ds")`) — only the final-model-fitting path has this bug.

**Fix it exactly like the ARIMA one was fixed** — change that dict comprehension to:

```python
grouped = {
    unique_id: group.sort_values("ds").set_index("ds")["y"].astype(float)
    for unique_id, group in full_long_df[full_long_df["unique_id"].isin(ids)].groupby("unique_id")
}
```

That's the whole fix for this specific bug — one line. Do it first, before running anything.

**Bug #1 (no parallelism) does NOT apply to Prophet** — `evaluate_prophet_config` and `fit_prophet_models` already use `joblib.Parallel(..., backend="threading", ...)`. Threading, not process-based `loky`, was a deliberate choice (see the comment already in the file about `cmdstan` subprocess GIL release — concurrent first-launches of a freshly-built Stan binary were crashing under `loky`). Leave that as-is.

**Bug #3's specific IPC symptom (`BrokenProcessPool`) does NOT apply either** — threading shares memory, no cross-process pickling of fitted models.

**But the underlying lesson from #3 — "don't assume a fitted model object is small, measure it" — absolutely does apply, and you should check it before running the full sweep:**

```python
import pickle
size_mb = len(pickle.dumps(fitted_prophet_model)) / 1e6
print(size_mb)
```

Prophet with `uncertainty_samples=0` (already set in `fit_prophet_model()`) should be much lighter than a statsmodels SARIMAX object — Prophet doesn't carry a state-space covariance history the same way — but don't take that on faith. If a single fitted model comes out unexpectedly large, multiply by ~2,660 and sanity-check it against available RAM *before* running the full thing, the same way the SARIMA issue was actually caught (empirically, on a small synthetic batch) rather than discovered hours into a real run.

## What actually needs doing

1. Apply the one-line fix above to `src/prophet_utils.py`.
2. Before committing to the full ~2,660-series run, **smoke-test with a small synthetic dataset** (a dozen series, ~150 weeks each): call `fit_prophet_models` directly, confirm models actually fit (not all falling back to the naive baseline), confirm `model.predict(...)` produces forecasts consistent with real 2012+ dates (not 1970-adjacent garbage), and check the per-model pickle size as described above.
3. Run `model_experiment_Prophet.ipynb` end-to-end. Per the notebook's own captured output, the last time this ran it got a baseline (`train_wmae=1,216.90`, `val_wmae=1,855.10`) and exactly one sweep config (`underfit_1`, `val_wmae=2,792`) before the hyperparameter-sweep cell's output just stops — the full grid is 30 configs (6 underfit / 18 balanced / 6 overfit) and didn't finish. It's not clear whether that was a real crash/hang or just an interrupted session (Colab timeout, closed laptop, etc.) — watch for it happening again. If it does, diagnose before just re-running blind:
   - Check CPU/thread usage while it runs — is `joblib` actually spreading work across cores, or silently falling back to serial?
   - Check for repeated `cmdstanpy - ERROR - Chain [1] error: code '1' Operation not permitted` messages in the log — already-documented Stan-binary-cold-start flakiness under concurrency, usually self-heals via the "falling back to Newton" retry visible in the historical output, but worth ruling out as the cause if the whole sweep hangs.
   - Watch memory, per the lesson above. If `fit_prophet_models` OOMs holding ~2,660 models at once, the fix pattern is the same idea used for ARIMA (write-to-disk-per-worker, load-back-sequentially), even though threading means you won't see the same `BrokenProcessPool` traceback.
4. Sanity-check the final registered Prophet model exactly like the ARIMA fix was verified: fit on a couple of series, confirm the model's internal `history`/`ds` values are real calendar dates in the expected 2010–2012 range, not epoch-adjacent garbage.
5. Once the sweep completes for real, update `README.md`'s **Prophet მოდელი** section with the actual results:
   - It currently has an explicit caveat that the sweep didn't finish and only baseline + `underfit_1` are known — replace that with the real full-sweep results (best config, its hyperparameters, val WMAE, and however many of the 30 configs actually completed).
   - Match the existing style/tone of the other model sections in this README exactly (read a couple of them, e.g. the ARIMA or XGBoost section, before writing — Georgian language, same subsection structure: intro paragraph, "რატომ ვიყენებთ...", "Train/Validation setup", "Hyperparameter search", results table, "დასკვნა").
   - Update the final model-comparison table (in the TimesFM section's conclusion) with Prophet's real best val WMAE instead of the current baseline-only number.
6. Report back a short summary: how many of the 30 configs completed, the best config + its val WMAE, per-model memory footprint you measured, and total wall-clock time for the full sweep.

## Practical repo notes

- MLflow tracking goes through DagsHub (`init_tracking()` in `src/mlflow_setup.py`); needs `DAGSHUB_USER_TOKEN` set (and `WANDB_API_KEY` if you want the W&B logging to work too — it's wrapped in `wandb.init(...)/wandb.finish()` per run).
- Data files (`data/train.csv`, `data/test.csv`, `data/features.csv`, `data/stores.csv`) aren't in git — either already present locally or need pulling from Kaggle (`kaggle competitions download -c walmart-recruiting-store-sales-forecasting`), same as the Colab setup cell at the top of the notebook does.
- `requirements.txt` includes `prophet`/`cmdstanpy` — first import triggers a Stan model compile, which is slow once and should be cached after.
