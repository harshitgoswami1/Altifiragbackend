# Standalone RAG backend

This is a separate Python project from `AltifiScraper/`. It never imports crawler
code or reads its working directory. It serves only the corpus snapshot copied
into this folder.

## Setup

```sh
uv sync
cp ../AltifiScraper/data/chunks.jsonl data/source/chunks.jsonl
cp ../AltifiScraper/data/chunk_report.json data/source/chunk_report.json
export OLLAMA_BASE_URL=http://YOUR_PRIVATE_OLLAMA_HOST:11434
uv run python -m app.index
uv run uvicorn app.api:app --host 127.0.0.1 --port 8000
```

`OLLAMA_BASE_URL` is required. The default embedding model is
`qwen3-embedding:8b-q8_0`; the default chat model is `qwen3:8b`. Override them
with `OLLAMA_EMBEDDING_MODEL` and `OLLAMA_CHAT_MODEL` if needed.

## Query routing

Routing separates intent, bond references, filters, ordering, and explanation needs.
Clear requests use deterministic rules. Named bonds come directly from the snapshot;
only educational context uses vector retrieval. Both chat endpoints share the same
routing and citation behavior. This change requires no corpus migration or index rebuild.

| Request | Behavior |
| --- | --- |
| “How is YTM different from coupon?” | Search blog/page sources; numeric examples remain educational. |
| “What does IN0020010081 pay?” | Retrieve that bond without embeddings. |
| “Compare IN0020010081 and INE808K08012” | Resolve every instrument and compare recorded fields; identify missing values. |
| “Show Tata Capital bonds” | Find matching recorded titles; there is no separate issuer field. |
| “Show secured AAA bonds above 10% YTM with monthly payments” | Apply every supported condition across the snapshot. |
| “Five highest-YTM bonds between 10% and 12%” | Filter, sort, and state the eligible snapshot comparison set. |
| “Find bonds above 10% and explain their risks” | Preserve deterministic results and add source-grounded educational context. |
| “Tell me about the Tata Capital bond” | Ask for an exact title or ISIN if several records match. |
| “What about its rating?” | Ask for a bond reference; the API has no conversation history. |
| “Which should I buy?” / “What is available today?” | Explain personal-advice/live-data limits; answer separable supported portions. |
| “Hello” / “Write a recipe” | Return a short capability response without vector retrieval. |

### Search fields and limits

- YTM, coupon, and observed minimum investment support `>`, `>=`, `<`, `<=`,
  explicit equality (“exactly”), and inclusive “between … and …” ranges.
  Percentages accept `%`, “percent”, or “per cent”. Money accepts INR, rupees,
  lakh/lac, and crore, such as “minimum investment at most 1 lakh”.
- Maturity accepts ISO dates or day/month-name/year dates, with before, after,
  on, on-or-before, and on-or-after conditions. “Maturing in 2030” includes the
  full calendar year; “before 2030” means before January 1, 2030.
- Rating, security, instrument category, interest-payment frequency, and title
  constraints use exact values or documented aliases. Ratings retain modifiers
  such as `(CE)`. “Secured” includes “Senior Secured”; “Unsecured” includes
  explicitly unsecured variants. “Subordinated” does not imply either.
- Annual/yearly and half-yearly/semiannual payment aliases are normalized.
  Payment frequency refers to interest payments; principal-payment filters are
  unsupported.
- Conditions combine with AND. OR, exclusions, relative dates, rating thresholds,
  ambiguous bare percentages, and unrecognized constraints request clarification.
  Put all screening conditions before a separate explanation request. A request
  is not executed if a condition cannot be accounted for.
- “Return” means observed YTM, with that interpretation stated in the answer.
  Coupon and YTM remain separate fields. Missing/malformed required fields are
  excluded and counted; malformed values never satisfy filters.
- Discovery excludes records flagged `maturity_matured` unless the request says
  “including matured records”. It does not recompute maturity from today's date.
  Exact lookups and named comparisons can show matured records with their flags.
- The default is four examples, capped at 20, with a total match count. Results
  use ISIN order unless the request names highest/lowest YTM, coupon, or minimum
  investment, or earliest/latest maturity. Ties use ISIN. “Best”, “safest”, and
  “top” without a measurable ordering criterion request clarification.

Only English questions are supported. Unfamiliar phrasing may need clarification.
Snapshot values and rankings do not establish current availability, market-wide
rankings, suitability, or guaranteed returns.

Set `RAG_MIN_RELEVANCE_SCORE` only after calibrating it against retrieval traces;
when unset, educational retrieval has no score cutoff. An empty retrieval result
is reported as insufficient retrieved evidence, not proof of absence from the corpus.

### Optional model fallback

Unresolved phrasing can use one Ollama classification call with temperature zero
and a Pydantic structured schema. The model proposes an intent, references, and
conditions; local code checks source phrases, resolves catalog identities, and
revalidates all constraints. Model output is never executed as a database filter.
Timeout, invalid output, unknown references, and unhandled conditions produce
clarification without retrying. Deterministic routes do not depend on this model.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `RAG_ROUTER_ENABLED` | `false` | Enable fallback after the evaluation gate passes. |
| `RAG_ROUTER_MODEL` | `OLLAMA_CHAT_MODEL` | Optional routing model override. |
| `RAG_ROUTER_TIMEOUT_SECONDS` | `15` | Positive, finite classification timeout. |

Before enabling fallback on a deployment, evaluate its actual Ollama model:

```sh
uv run python -m app.evaluate_routing tests/routing_eval.json
```

This command requires `OLLAMA_BASE_URL`. It uses a synthetic bond catalog and
40 labeled paraphrases, calls the model only when rules need fallback, and does
not retrieve documents or generate answers. It prints per-case decisions and
exits successfully only with at least 95% exact decision accuracy, zero unsafe
executed decisions (including dropped conditions or invented references), and at
least one model call. It does not change deployment settings. Add representative
held-out production questions before rollout, and rerun after model changes.
Enable `RAG_ROUTER_ENABLED=true` only after passing. The offline unit suite tests
model handling with fakes; it does not establish real-model accuracy.

LangSmith tracing is opt-in. Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`,
and optionally `LANGSMITH_PROJECT`. Both endpoints emit a routing trace containing
method, intent, reason, resolved references, applied filters, ordering, fallback
latency, limitations, and clarification outcome. Retrieval and model calls retain
their existing traces; `/v1/chat` also has a parent `rag_chat` trace.

For a private-network deployment, choose the Uvicorn bind address in the
process manager or command line and keep firewall access limited to trusted
clients. The API does not enable CORS or authentication.

## Refreshing the corpus

After generating a new audited chunk artifact in `AltifiScraper/`, copy both
files again and rebuild explicitly:

```sh
cp ../AltifiScraper/data/chunks.jsonl data/source/chunks.jsonl
cp ../AltifiScraper/data/chunk_report.json data/source/chunk_report.json
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
personalized investment advice, live availability/pricing, or market-wide
rankings. Explicit numeric ordering is limited to the eligible snapshot records.

`POST /v1/chat/stream` accepts the same JSON body and returns server sent
events (`text/event-stream`). It sends a `citations` event containing the same
citation objects as `/v1/chat`, then `token` events with `{ "text": "..." }`,
and finally a `done` event. Questions answered without the model emit their
full answer in one `token` event. Validation and index errors return normal
HTTP 422/503 responses before streaming begins; model failures during a stream
emit an `error` event with a `detail` field. Mixed requests emit the deterministic
search/comparison result before explanation tokens. If explanation generation
fails for a non-streaming mixed request, the deterministic result is still returned
with a short explanation-unavailable message.

## Tests

```sh
uv run python -m unittest discover -s tests -v
```
