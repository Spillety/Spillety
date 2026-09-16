.PHONY: data notebooks clean
DATA_DIR=data/elliptic_raw
ARCHIVE=archive.zip

data: ## one-command: скачать/распаковать elliptic в data/elliptic_raw
	@python scripts/download_elliptic.py

notebooks: data ## поставить зависимости для ноутбуков
	@pip install -r docs/notebooks/requirements-notebooks.txt

clean: ## удалить деривативы (сырьё не трогает)
	@rm -rf data/processed
	@find docs/notebooks -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true

help: ## показать цели
	@grep -E '^[a-zA-Z_-]+:.*?##' Makefile | awk 'BEGIN{FS=":.*?##"} {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

