.PHONY: run test lint format

run:
	streamlit run app.py

test:
	python -m pytest tests -q

lint:
	python -m ruff check .

format:
	python -m black .
