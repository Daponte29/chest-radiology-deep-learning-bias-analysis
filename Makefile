.PHONY: train evaluate bias-eval app test lint format docker-build docker-push deploy clean

# ── Training ──────────────────────────────────────────────────────────────────
train:
	python src/train.py --config src/configs/train_original.yaml

train-gb:
	python src/train.py --config src/configs/train_gb.yaml

train-ps:
	python src/train.py --config src/configs/train_ps.yaml

train-ce:
	python src/train.py --config src/configs/train_ce.yaml

train-pr:
	python src/train.py --config src/configs/train_pr.yaml

# ── Evaluation ────────────────────────────────────────────────────────────────
evaluate:
	python src/evaluate.py --config src/configs/train_original.yaml

bias-eval:
	python src/bias_eval.py --config-dir src/configs/archive_results_configs/config_1/

# ── App ───────────────────────────────────────────────────────────────────────
app:
	streamlit run deploy/app.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# ── Lint / format ─────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t chexpert-bias:latest -f deploy/Dockerfile .

docker-push:
	docker tag chexpert-bias:latest $(ECR_REPO):latest
	docker push $(ECR_REPO):latest

deploy: docker-build docker-push

# ── Misc ──────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
