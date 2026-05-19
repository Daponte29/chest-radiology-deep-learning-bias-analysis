.PHONY: train evaluate test lint format deploy clean

# ── Training ──────────────────────────────────────────────────────────────────
train:
	python src/train.py --config configs/experiment_01.yaml

sweep:
	wandb sweep configs/sweep.yaml

# ── Evaluation ────────────────────────────────────────────────────────────────
evaluate:
	python src/evaluate.py --config configs/experiment_01.yaml

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# ── Lint / format ─────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/

# ── Deploy ────────────────────────────────────────────────────────────────────
export:
	python deploy/export.py

docker-build:
	docker build -t my_project:latest -f deploy/Dockerfile .

docker-push:
	# TODO: set ECR_REPO in .env
	docker tag my_project:latest $(ECR_REPO):latest
	docker push $(ECR_REPO):latest

deploy: docker-build docker-push

# ── Misc ──────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
