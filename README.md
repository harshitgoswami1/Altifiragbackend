# Finance text RAG and bond database backend

This is a separate Python project from `AltifiScraper/`. It never imports crawler
code or reads its working directory. It indexes audited blog/page text copied
into this folder and queries `scraped_bonds` in PostgreSQL for bond records.

## Setup

```sh
uv sync
cp ../AltifiScraper/data/chunks.jsonl data/source/chunks.jsonl
cp ../AltifiScraper/data/chunk_report.json data/source/chunk_report.json
export OLLAMA_BASE_URL=http://YOUR_PRIVATE_OLLAMA_HOST:11434
export DATABASE_URL='postgres://USER:PASSWORD@pooled.db.prisma.io:5432/postgres?sslmode=require'
uv run python -m app.index
uv run uvicorn app.api:app --host 127.0.0.1 --port 8000
```

`OLLAMA_BASE_URL` is required. The default embedding model is
`qwen3-embedding:8b-q8_0`; the default chat model is `qwen3:8b`. Override them
with `OLLAMA_EMBEDDING_MODEL` and `OLLAMA_CHAT_MODEL` if needed.
`DATABASE_URL` must be the Prisma Postgres pooled runtime URL. Keep it in the
environment, never in source control. The database account needs read access to
`scraped_bonds`; this service makes no schema changes. The index command uses
only `OLLAMA_BASE_URL` and the copied corpus, while chat also requires the database.

## Query routing

Routing separates bond lookup/search from educational questions. Bond values come
from PostgreSQL on each request; audited blog/page text supplies general finance
explanations. The optional model router proposes a structured route but never SQL.
Both chat endpoints use the same routing, result, and citation behavior.

| Request | Behavior |
| --- | --- |
| “How is YTM different from coupon?” | Search audited blog/page text. |
| “What does INE157D14EA9 pay?” | Query its database row by ISIN without embeddings. |
| “Compare INE157D14EA9 and another ISIN” | Fetch both rows and identify missing values. |
| “Show Clix Capital bonds” | Match issuer names; clarify ambiguous single-bond references by ISIN. |
| “Five highest-YTM bonds between 8% and 10%” | Filter and order active database rows. |
| “Show bonds with face value above 5 lakh issued before 2024” | Apply both SQL conditions. |
| “Show matured bonds” | Match `bond_status` across active and inactive rows. |
| “Find bonds above 8% and explain their risks” | Keep database results and add text-grounded context. |
| “Show secured AAA bonds” | Ask for a supported field; these attributes are absent from the table. |
| “Which should I buy?” / “What is available today?” | Explain advice and live-market limits. |
| “Hello” / “Write a recipe” | Return a short capability response without vector retrieval. |

### Search fields and limits

- YTM, coupon, minimum investment, and face value support `>`, `>=`, `<`, `<=`,
  explicit equality (“exactly”), and inclusive “between … and …” ranges.
  Percentages accept `%`, “percent”, or “per cent”. Money accepts INR, rupees,
  lakh/lac, and crore, such as “minimum investment at most 1 lakh”.
- Maturity and issue dates accept ISO dates or day/month-name/year dates, with
  before, after, on, on-or-before, and on-or-after conditions. Year filters
  cover the full calendar year.
- Bond status supports exact, case-insensitive equality. A status request such
  as “matured bonds” includes matching inactive rows. Other discovery searches
  use only `is_active = true`; exact ISIN lookups may show inactive rows.
- Issuer names identify candidate bonds. A shared issuer name does not identify
  a unique instrument; use ISIN to disambiguate. Rating, security, category, and
  payment frequency are unavailable in `scraped_bonds` and cannot filter results.
- Conditions combine with AND. OR, exclusions, relative dates, rating thresholds,
  ambiguous bare percentages, and unrecognized constraints request clarification.
  Put all screening conditions before a separate explanation request. A request
  is not executed if a condition cannot be accounted for.
- “Return” means fetched YTM, with that interpretation stated in the answer.
  Coupon and YTM remain separate fields. Null required fields never satisfy
  filters or sorting.
- `is_active` and `bond_status` describe source data. Neither establishes
  current purchase availability. `fetched_at` is shown as the observation time.
- The default is four examples, capped at 20, with a total match count. Results
  use ISIN order unless the request names highest/lowest YTM, coupon, minimum
  investment, or face value, or earliest/latest maturity or issue date. Bare
  “top” means highest fetched YTM among eligible database rows. Ties use ISIN.
  “Best” and “safest” still request clarification.

Only English questions are supported. Unfamiliar phrasing may need clarification.
Fetched values and rankings do not establish current availability, market-wide
rankings, suitability, or guaranteed returns.

Set `RAG_MIN_RELEVANCE_SCORE` only after calibrating it against retrieval traces;
when unset, educational retrieval has no score cutoff. An empty retrieval result
is reported as insufficient retrieved evidence, not proof of absence from the corpus.

### Optional model routing

When enabled, every query makes one Ollama classification call with temperature
zero and a Pydantic structured schema. The model proposes an intent, references,
conditions, ordering, count, and requested output fields. Local code checks
source phrases, resolves catalog identities, rejects unaccounted constraints,
and validates supported values and operators. Only validated fields and bound
values reach the SQL retriever. A validated model discovery route may handle wording the
rules do not recognize; a complete rule route remains the fallback if the model
times out or returns invalid output. Otherwise, the API asks for clarification.
This adds routing latency to every enabled request, including greetings and exact
bond lookups.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `RAG_ROUTER_ENABLED` | `false` | Enable model routing for every query after the evaluation gate passes. |
| `RAG_ROUTER_MODEL` | `OLLAMA_CHAT_MODEL` | Optional routing model override. |
| `RAG_ROUTER_TIMEOUT_SECONDS` | `15` | Positive, finite classification timeout. |

Before enabling model routing on a deployment, evaluate its actual Ollama model:

```sh
uv run python -m app.evaluate_routing tests/routing_eval.json
```
This command requires `OLLAMA_BASE_URL`. It uses a synthetic bond catalog and
49 labeled paraphrases, calls the model once for every case, and does
not retrieve documents or generate answers. It prints per-case decisions and
exits successfully only with at least 95% exact decision accuracy, zero unsafe
executed decisions (including dropped conditions or invented references), one
model call per case, and at least one accepted model route. It reports accepted
model routes separately and does not change deployment settings. Add
representative held-out production questions before rollout, and rerun after
model changes.
Enable `RAG_ROUTER_ENABLED=true` only after passing. The offline unit suite tests
model handling with fakes; it does not establish real-model accuracy.

LangSmith tracing is opt-in. Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`,
and optionally `LANGSMITH_PROJECT`. Both endpoints emit a routing trace containing
method, router outcome, intent, reason, resolved references, applied filters,
ordering, router latency, limitations, and clarification outcome. Retrieval and
model calls retain their existing traces; `/v1/chat` also has a parent `rag_chat`
trace.

For a private-network deployment, choose the Uvicorn bind address in the
process manager or command line and keep firewall access limited to trusted
clients. The API does not enable CORS or authentication.

## Refreshing the corpus

After generating a new audited chunk artifact in `AltifiScraper/`, copy both
files again and rebuild the text index explicitly:

```sh
cp ../AltifiScraper/data/chunks.jsonl data/source/chunks.jsonl
cp ../AltifiScraper/data/chunk_report.json data/source/chunk_report.json
uv run python -m app.index
```

The indexer rejects a report without a passed audit, nonempty errors, mismatched
checksum, invalid rows, and mismatched record counts. It embeds `embedding_text`
for blog/page rows only. Bond-only changes in the copied artifact do not stale
the text index. A new index becomes active only after every text chunk is stored;
a failed rebuild leaves the previous index active. Rebuild once when upgrading
from an older index that contains bond chunks.

## API

`GET /healthz` returns readiness for the text index, Ollama host, and bond database.
It returns HTTP 503 while the service cannot answer requests.

`POST /v1/chat` accepts a single stateless question:

```json
{"question": "What is a fixed deposit?"}
```

It returns a completed answer and citations containing the source URL and
observation metadata. Bond citations have `document_type: "bond"`, a stable
`bond:<ISIN>` chunk ID, and `fetched_at` in `observed_at`; text citations retain
their source chunk IDs. The service does not provide personalized investment
advice, verified live availability/pricing, or market-wide rankings. Explicit
numeric ordering is limited to matching database rows.

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
