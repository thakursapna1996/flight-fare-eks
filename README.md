# Flight Fare Prediction — Containerised ML Service on AWS EKS

A machine-learning web service that predicts airline ticket prices from trip
details, packaged for production: containerised, deployed to Kubernetes on AWS
EKS with Terraform, and shipped through a GitHub Actions CI/CD pipeline.

The model is a `RandomForestRegressor` trained on the Kaggle-style Indian
flight-fare dataset (~10.7k rows). It is served behind a small Flask API with
health/readiness probes, autoscaling and container-security scanning.

> Originally an NCI *Cloud Machine Learning* coursework app (a Flask + Jupyter
> notebook project). This repository is that model wrapped in the lifecycle,
> deployment and operability layer a platform team would expect.

---

## Architecture

```
            ┌──────────── GitHub Actions ────────────┐
 git push → │ CI: flake8 → pytest → docker build →    │
   main     │     Trivy scan                          │
            │ CD: build → push ECR → kubectl rollout  │
            └───────────────────┬─────────────────────┘
                                │ (OIDC, no static keys)
                                ▼
   AWS (Terraform-managed):  ECR ──image──► EKS cluster
                                             └─ Deployment (2–6 pods, HPA)
                                                 └─ Service (ClusterIP)
                                                     └─ Flask + gunicorn
                                                         └─ RandomForest model
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the request path and the
training/serving data flow.

---

## Repository layout

| Path | What it is |
|------|------------|
| `app/features.py` | **Feature engineering** — the single source of truth for how raw input becomes the model's 29-feature vector. Shared by training and serving. |
| `app/model.py` | Loads the pickled model (cached) and runs predictions. |
| `app/main.py` | **Flask entry point** — routes, JSON API, `/health` and `/ready` probes. |
| `scripts/train.py` | Reproducible training; writes `model/flight_fare_rf.pkl`. |
| `tests/` | `test_features.py` (pure unit tests), `test_api.py` (HTTP integration). |
| `Dockerfile` | Multi-stage, non-root, read-only-root-filesystem image. |
| `infra/k8s/` | Namespace, Deployment, Service, HPA (Kustomize). |
| `infra/terraform/` | VPC + EKS + ECR. |
| `.github/workflows/` | `ci.yaml` (lint/test/scan), `cd.yaml` (deploy to EKS). |
| `notebooks/` | Original exploratory notebook (EDA + model selection). |

---

## Run it locally

```bash
make install        # install dependencies
make train          # produce model/flight_fare_rf.pkl (needed before serving)
make test           # 19 tests
make run            # serve on http://localhost:8080
```

Predict via JSON:

```bash
curl -X POST http://localhost:8080/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"Dep_Time":"2024-06-15T22:20","Arrival_Time":"2024-06-16T01:10",
       "stops":"1","airline":"IndiGo","Source":"Delhi","Destination":"Cochin"}'
# {"currency":"INR","fare":4361.33}
```

## Run it in a container

```bash
make docker-build
docker run -p 8080:8080 flight-fare:local
```

## Deploy to AWS

```bash
cd infra/terraform
terraform init && terraform apply          # creates VPC, EKS, ECR
aws eks update-kubeconfig --region <region> --name flight-fare-eks
cd ../.. && make k8s-apply                   # namespace, deployment, svc, HPA
```

CI/CD then takes over: pushing to `main` builds the image, pushes it to ECR
and rolls it out to EKS. The pipeline authenticates to AWS with GitHub OIDC —
no long-lived cloud keys are stored in GitHub. Required repo config:
`AWS_REGION`, `ECR_REPOSITORY`, `EKS_CLUSTER` (variables) and
`AWS_DEPLOY_ROLE_ARN` (secret).

---

## Model

| | |
|---|---|
| Algorithm | RandomForestRegressor (100 trees, max_depth 12) |
| Features | 29 (9 numeric + one-hot airline/source/destination) |
| Test R² | 0.81 |
| Test MAE | ~1184 INR |

Retrain with `make train`. The model is pinned to the scikit-learn version in
`requirements.txt`; because pickled trees are version-sensitive, retrain
whenever that version changes so the artifact and the runtime never disagree.
The `.pkl` is tracked with Git LFS (see `.gitattributes`).

### Training/serving parity

`scripts/train.py` and the web app both import `FEATURE_ORDER` and the
encoders from `app/features.py`. The training matrix is `reindex`-ed onto that
exact column list, so the features the model learns and the features it is
asked to predict on are guaranteed identical and in the same order — the usual
source of silent production ML bugs is designed out.

---

## Notes

- `terraform validate` and the Trivy/Docker steps run against the live
  registry/daemon in CI; they are not exercised in this offline checkout.
- Two encoding bugs from the original coursework app were fixed here (a
  destination that set two one-hot columns, and a `New Delhi` / `New_Delhi`
  value mismatch); both are pinned by tests in `tests/test_features.py`.
