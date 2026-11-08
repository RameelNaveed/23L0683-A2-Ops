# VisionGuard: AI vs Real image classifier (23L0683_A2_Ops)

![CI](https://github.com/RameelNaveed/23L0683_A2_Ops/actions/workflows/ci.yml/badge.svg)

Upload an image, get a verdict (AI generated or real), a confidence value and an honest caveat.
Every request is stored in a database and listed on the History page.

| Link | |
|---|---|
| DagsHub (code, data, MLflow) | https://dagshub.com/RameelNaveed/23L0683_A2_Ops |
| Model API (Cloud Run) | https://<service>.run.app |

## Layout

```
visionguard/   shared preprocessing, upload validation, predictor (used by training AND serving)
trainer/       Tasks 1 to 3, one command line: python -m trainer --help
experiments.yaml  training presets (one preset = one MLflow run)
labs/          tool labs: MLflow toy, CI toy
app/           Flask web app + database trail (Tasks 4, 5, 8)
api/           cloud model API (Task 7)
tests/         pytest suite used by CI
Dockerfile, docker-compose.yml, .github/workflows/ci.yml   Task 6
```

## Run everything with one command

```
copy .env.example .env      # then edit the values
python -m trainer export
docker compose up -d --build
```
Open http://localhost:5000

## Run without Docker (Windows)

```
.venv\Scripts\activate
waitress-serve --listen=127.0.0.1:5000 app.wsgi:app
```
