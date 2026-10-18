install:
	pip install -r requirements.txt && pip install -e .

run:
	uvicorn app.api:app --reload --port 8000

stats:
	python -m app.cli stats

upload:
	python -m app.cli upload $(FILE)

retrieve:
	python -m app.cli retrieve "$(Q)"

clean:
	rm -rf data/chroma data/uploads