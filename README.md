# Standalone RAG backend

This is a separate Python project from `webcrawler/`. It never imports crawler
code or reads its working directory. It serves only the corpus snapshot copied
into this folder.

## Setup

```sh
uv sync
cp ../webcrawler/data/chunks.jsonl data/source/chunks.jsonl
cp ../webcrawler/data/chunk_report.json data/source/chunk_report.json
export OLLAMA_BASE_URL=http://YOUR_PRIVATE_OLLAMA_HOST:11434
uv run python -m app.index
uv run uvicorn app.api:app --host 127.0.0.1 --port 8000
```

`OLLAMA_BASE_URL` is required. The default embedding model is
`qwen3-embedding:8b-q8_0`; the default chat model is `qwen3:8b`. Override them
with `OLLAMA_EMBEDDING_MODEL` and `OLLAMA_CHAT_MODEL` if needed.

Retrieval routes questions deterministically: bond-specific questions use bond
metadata filters, general questions use blog chunks, and mixed questions use
separate bond and blog retrieval lanes. Set `RAG_MIN_RELEVANCE_SCORE` only
after calibrating a value from retrieval traces; when unset, no score cutoff is
applied.

Bond searches such as "5 bonds with more than 12% rate of return" use the
snapshot's observed yield to maturity (YTM). Requests need no special opening
phrase. The backend supports above/more than (`>`), at least (`>=`), below
(`<`), and at most (`<=`), including percentages written as "percent" or
"per cent". It checks recorded values and excludes records flagged as matured.
Counts such as "5 bonds" and "five bonds" are honored; the default is four
examples, with a maximum of 20 per response. Results include the total match
count and citations, ordered by ISIN. Unsupported numeric conditions, including
coupon filters and ranges, request clarification instead of retrieving blogs.
Educational questions still use general sources. Chat and streaming share this
behavior. The observations do not establish current availability or guaranteed returns.

LangSmith tracing is opt-in. Set `LANGSMITH_TRACING=true`,
`LANGSMITH_API_KEY`, and optionally `LANGSMITH_PROJECT` before starting the
API. Each `/v1/chat` request is recorded as a `rag_chat` trace, with routed
retrieval and LangChain model calls nested under it. Streaming requests retain
the retrieval and model traces.

For a private-network deployment, choose the Uvicorn bind address in the
process manager or command line and keep firewall access limited to trusted
clients. The API does not enable CORS or authentication.

## Refreshing the corpus

After generating a new audited chunk artifact in `webcrawler/`, copy both
files again and rebuild explicitly:

```sh
cp ../webcrawler/data/chunks.jsonl data/source/chunks.jsonl
cp ../webcrawler/data/chunk_report.json data/source/chunk_report.json
uv run python -m app.index
```

The indexer rejects a report without a passed audit, nonempty errors, mismatched
checksum, invalid rows, and mismatched record counts. It embeds `embedding_text`
only. A new index becomes active only after every chunk is stored successfully;
a failed rebuild leaves the previous index active.

## API

`GET /healthz` returns readiness for the copied corpus/index and Ollama host.
It returns HTTP 503 while the service cannot answer requests.

`POST /v1/chat` accepts a single stateless question:

```json
{"question": "What is a fixed deposit?"}
```

It returns a completed answer and citations containing the source URL and
observation metadata. The service is informational only: it does not provide
personalized investment advice, live availability/pricing, or exhaustive
numeric rankings.

`POST /v1/chat/stream` accepts the same JSON body and returns server sent
events (`text/event-stream`). It sends a `citations` event containing the same
citation objects as `/v1/chat`, then `token` events with `{ "text": "..." }`,
and finally a `done` event. Questions answered without the model emit their
full answer in one `token` event. Validation and index errors return normal
HTTP 422/503 responses before streaming begins; model failures during a stream
emit an `error` event with a `detail` field.

## Tests

```sh
uv run python -m unittest discover -s tests -v
```
