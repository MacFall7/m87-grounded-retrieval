PYTHON ?= python3
DSN ?= postgresql://postgres:postgres@localhost:5432/postgres
CORPUS ?= corpus
INDEX ?= default
Q ?= how does the kernel handle a policy denial

.PHONY: install test test-all ingest ingest-stub query db-up db-down

install:
	$(PYTHON) -m pip install -e .

# Default target excludes integration so a fresh clone runs green with no database.
test:
	$(PYTHON) -m pytest -m "not integration"

test-all:
	GROUNDED_RETRIEVAL_DSN=$(DSN) $(PYTHON) -m pytest

db-up:
	docker compose up -d

db-down:
	docker compose down

ingest:
	GROUNDED_RETRIEVAL_DSN=$(DSN) $(PYTHON) -m grounded_retrieval.ingest \
		--corpus $(CORPUS) --dsn $(DSN) --index $(INDEX)

# Stub embeddings for a smoke run with no model download. Retrieval quality numbers
# from this path are meaningless and must never be reported as measurements.
ingest-stub:
	GROUNDED_RETRIEVAL_DSN=$(DSN) $(PYTHON) -m grounded_retrieval.ingest \
		--corpus $(CORPUS) --dsn $(DSN) --index $(INDEX) --stub-embeddings

# Invoked through the public API rather than a CLI module so this target does not
# depend on a command surface that is still being settled. Exit status is the leash
# decision: 0 answered, 2 refused.
query:
	@GROUNDED_RETRIEVAL_DSN=$(DSN) $(PYTHON) -c 'import sys; \
	from grounded_retrieval import ChunkStore, GroundedRetriever, SentenceTransformerBackend; \
	r = GroundedRetriever(ChunkStore(dsn="$(DSN)", index_name="$(INDEX)"), SentenceTransformerBackend()); \
	a, receipt = r.query("$(Q)"); \
	print(receipt.to_json()); \
	print(a.answer or "REFUSED: " + a.verdict.reason); \
	sys.exit(a.exit_code)'

calibrate:
	PYTHONPATH=src python3 scripts/calibrate.py
