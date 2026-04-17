.PHONY: install train eval test clean lint

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

train:
	python scripts/train_stage1.py

eval:
	python scripts/evaluate_stage1.py

test:
	pytest

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache

lint:
	python -m compileall configs src scripts tests
