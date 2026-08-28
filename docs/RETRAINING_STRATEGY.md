# Retraining Strategy

## Principles
1. Retrain ONLY when justified by new real data or performance degradation
2. Never automatically replace a production model
3. Every retraining run produces a new versioned artifact
4. Promotion requires explicit comparison against the current production model

## Trigger Conditions

### Scheduled
- **Monthly review**: check if new CHIRPS data or new confirmed events are available
- If new events exist, retrain and compare

### Event-Driven
- **New confirmed event added** to `temporal_design.py`
- **Performance degradation** detected by monitoring (false negatives, drift)
- **Feature pipeline change** (new features, corrected data)

## Retraining Workflow

```
1. Add new events (if any)
   python scripts/add_confirmed_event.py --apply

2. Run full pipeline
   python scripts/rebuild_pipeline.py

3. Review prod_report_{date}.json
   - Check gate status
   - Compare AUC/avg_precision with previous model
   - Check per-corridor performance

4. Promotion decision:
   - New model must pass the honesty gate
   - New model must NOT be worse than current model on any metric
   - If any metric regresses, investigate before promoting

5. Promote:
   - Update the "active model" pointer (backend config)
   - Keep old model artifacts for rollback
   - Document the change
```

## What Requires Retraining

| Change | Retrain? | Reason |
|--------|----------|--------|
| New confirmed event | Yes | More data improves generalization |
| CHIRPS data update | No | Features already use latest available data |
| New feature added | Yes | Model must learn new feature |
| Bug fix in feature pipeline | Yes | Corrected features change the data distribution |
| Threshold adjustment only | No | Just update the threshold in the report |
| Code refactoring (no data change) | No | Reproducibility test only |

## Rollback
- Keep the previous model's `.ubj` + `_features.json` + `_calib.pkl` + `_report.json`
- To rollback: point the backend at the previous tag
- Old artifacts are never deleted until a new model has been stable for 30 days

## Versioning
- Each model gets a date-based tag: `prod_real_temporal_YYYY-MM-DD`
- If multiple trains happen same day, append a suffix: `prod_real_temporal_2026-08-28_v2`
- The report JSON records: train events, val events, test events, metrics, features, hyperparameters

## Reproducibility (hard gate, blocker B6)
A model is only honestly reproducible if a clean rebuild from the owning git commit
and a known environment reproduces the same bundles.

- **Capture the environment** with a frozen snapshot. `requirements.lock` records the
  exact venv that produced the current bundles:
  ```
  python -m pip freeze > requirements.lock   # exclude the editable self-reference
  ```
  Refresh this file whenever dependencies change. `pyproject.toml` ranges are NOT a
  substitute — they do not pin the ML/science stack.
- **Commit before training.** The model records `git status --porcelain`
  (`code_version()` in `scripts/train_production_risk_model.py:83`). Training with a
  dirty tree sets `working_tree_dirty: true` and breaks the reproducibility claim.
  Sequence must be: add events → **commit** → then `python scripts/rebuild_pipeline.py`.
- **`working_tree_dirty` is captured at train start.** `code_version()` is evaluated
  BEFORE any model artifacts are written (`scripts/train_production_risk_model.py`),
  because writing the new model bundle into the tracked `data/models/` directory would
  otherwise make `git status` non-empty and the flag would always read `true`. For a
  clean record, the working tree (including untracked scratch files and stale bundles)
  must be clean at the moment training begins.
- **Clean-env rebuild steps:**
  ```
  git checkout <owning-commit>          # clean tree
  python -m venv .venv-clean
  .venv-clean/bin/pip install -r requirements.lock
  .venv-clean/bin/pip install -e .      # installs this repo itself (omitted from lock by design)
  python scripts/rebuild_pipeline.py    # downloads missing CHIRPS + builds + retrains + tests
  ```
- **Verify** the rebuilt model hashes match the report (`model_sha256`, dataset hash),
  and that the report's `code_version.git_commit` equals the checkout HEAD with
  `working_tree_dirty: false`.
- **Known gap:** a fully clean-environment rebuild against `requirements.lock` has not
  yet been executed and validated end-to-end; this remains the outstanding action for
  the reproducibility gate.
