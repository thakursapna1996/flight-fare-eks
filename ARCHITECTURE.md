# Architecture

## Request path (serving)

```
Client (browser or API caller)
   │  POST /predict (form)  or  POST /api/predict (JSON)
   ▼
Service (ClusterIP, port 80) ──► Pod :8080
   ▼
gunicorn (2 workers × 4 threads)
   ▼
Flask app  (app/main.py)
   │ 1. parse request        request.form / request.get_json()
   │ 2. build feature vector app/features.build_feature_vector()  ─┐
   │ 3. model.predict(...)   app/model.predict_fare()              │ shared
   ▼                                                               │ feature
Response: predicted fare (INR)                                     │ spec
                                                                   │
Training (scripts/train.py) ── reindex to FEATURE_ORDER ───────────┘
```

Kubernetes probes hit `/health` (liveness — is the process up?) and `/ready`
(readiness — is the model loaded and able to predict?). Traffic is only routed
to a pod once `/ready` returns 200, so a pod that is still loading the model
never receives requests.

## Training/serving data flow

The same `FEATURE_ORDER` list in `app/features.py` defines:

- the **column order of the training matrix** (`scripts/train.py` calls
  `reindex(columns=FEATURE_ORDER)`), and
- the **order of values** sent to `model.predict()` at serving time.

Because both sides derive from one list, the model can never be trained on one
column layout and served another — the classic cause of a model that "works in
the notebook" but returns garbage in production.

## The 29 features

| Group | Count | Columns |
|-------|-------|---------|
| Numeric | 9 | Total_Stops, Journey_day, Journey_month, Dep_hour, Dep_min, Arrival_hour, Arrival_min, Duration_hours, Duration_mins |
| Airline (one-hot, baseline *Air Asia*) | 11 | Air India, GoAir, IndiGo, Jet Airways, Jet Airways Business, Multiple carriers, Multiple carriers Premium economy, SpiceJet, Trujet, Vistara, Vistara Premium economy |
| Source (baseline *Banglore*) | 4 | Chennai, Delhi, Kolkata, Mumbai |
| Destination (baseline *Banglore*) | 5 | Cochin, Delhi, Hyderabad, Kolkata, New Delhi |

One-hot encoding uses `drop_first=True`; the dropped baseline category is
represented implicitly as all-zeros, exactly as at training time.

## Security posture

- Image runs as a non-root user (`uid 10001`) with a read-only root
  filesystem and all Linux capabilities dropped.
- ECR scans images on push; CI additionally fails on CRITICAL/HIGH findings
  via Trivy.
- CI/CD authenticates to AWS with GitHub OIDC — no static AWS keys in GitHub.
- ECR repository is immutable-tagged (images are addressed by git SHA).
