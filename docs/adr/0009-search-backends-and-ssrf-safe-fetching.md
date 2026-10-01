# ADR-0009: Search backends (Tavily, then no search) and SSRF-safe page fetching

**Status**: Accepted. Amended 2026-10-01: the self-hosted SearXNG level
that came first is superseded by Tavily only, to keep the project strictly
in the GCP free tier (see "History" below). The SSRF-safe page fetching
decision stands unchanged.

## Context

Web search ran on Tavily only, through the `tavily-python` SDK: every search
at `search_depth="advanced"` (2 credits), up to three searches per question,
and the chat tool on top. The free plan is 1,000 credits a month, so a few
hundred questions used it up, after which users got a canned "search
unavailable" reply. Without `TAVILY_API_KEY` there was no search at all.
The SDK also pulled in `tiktoken` and ran its synchronous client in the
default thread pool, next to Firestore.

Tavily also read the pages behind URLs users sent. There was no page
fetching of our own, and adding one is risky: the service can reach the
metadata server (and any private network it is ever connected to), and the
URLs come from users and from search results.

## Decision

### Two levels, in order

1. **Tavily**, over plain `httpx` (the SDK is gone): `search_depth` basic,
   `auto_parameters` off (it can switch to advanced), `include_usage` on,
   `country` only with the general topic (the only one Tavily accepts it
   for). With `TAVILY_API_KEY` it sends `Authorization: Bearer`; without
   one, `X-Tavily-Access-Mode: keyless` (free, rate-limited, same answers)
   only with `TAVILY_KEYLESS_ENABLED=true`. Keyless is an opt-in because
   v2 documented an empty key as "no web search"; such a `.env` keeps
   Tavily off (`.env.example` opts in). A 429 skips Tavily for its
   `Retry-After`, a plan limit or a refused key for an hour.
2. **No search**: the answer tiers reply from the model's own knowledge,
   prefixed with "（目前無法使用搜尋服務，以下依模型既有知識回答，可能不是最新資訊）".

`ChainedSearchBackend` asks the configured backends in order (Tavily alone
today; it is the seam another backend plugs into, in the composition
root's `build_search_backend`) and moves on after any exception or an
empty answer; when none answers it returns an empty response marked
`degraded` instead of raising, and `all_failed` when every backend failed
rather than one finding nothing. A search that found nothing with its
topic, time-range or country filters gets one relaxed retry without them;
one that every backend failed does not, since it would only fail again.
The web search agent and the chat agent's `web_search` tool share the one
chain, built in the composition root (ADR-0005); with no backend configured
the tool is not offered at all. `/health/detail` shows
each backend's state (Tavily's mode and the credits it reported, its
cooldown and the last error) and never a key.

### Reading pages ourselves, safely

The URLs a user sends are read here first, then by Tavily extract (basic
depth, 1 credit per 5 pages) for the ones that failed. A page still unread
is named to the model in a system note that forbids guessing its content,
and the reply starts with 「（無法開啟你提供的連結，以下回答沒有參考該網頁內容）」
(「（部分連結無法開啟，以下只依讀到的網頁內容回答）」 when other links were
read). For searches, the top `SEARCH_FETCH_TOP_N` (3)
results are read too, so the model sees page text rather than snippets;
all reads of one search share `SEARCH_FETCH_BUDGET_SECONDS` (6 s) and
whatever finished by then is used. The whole search stage must end 20 s
before the reply deadline, so the answer keeps its time. The page reader
gets the stage's deadline and stops 0.5 s before it, so the pages read by
then are used; it cancels the rest, even when it is cancelled itself, so
no fetch outlives the stage. An answer that has the whole turn (a thinking
answer, in any chat, or any answer starting with under `MIN_TIER_SECONDS`
of the reply window; it goes by push once the reply token is gone)
measures the stage from its own deadline, the turn's, once the reply
window is too short to search in.

`PageFetcher` (`src/services/search/page_fetcher.py`) refuses anything
that could reach inside:

- http and https only, ports 80 and 443 only;
- no `localhost`, `metadata`, `metadata.google.internal`, `*.internal`,
  `*.local`, `*.localhost`, before any lookup;
- the host is resolved by the fetcher (getaddrinfo, 2 s, on four threads of
  the fetcher's own, never the loop's resolver that LINE and Cloud Tasks
  connections share: a lookup that hangs cannot be stopped), and **every**
  address must be public: IPv4-mapped and 6to4 addresses are unwrapped
  first, and private, loopback, link-local (169.254.169.254 among them),
  multicast, reserved (NAT64 among them), unspecified, IPv6 site-local
  (`fec0::/10`, which Python counts as global) and CGNAT (100.64.0.0/10)
  addresses are refused. Numeric forms such as
  `2130706433` resolve like names, so they are caught too;
- the connection goes to the address that passed (a custom httpcore
  network backend), so a second lookup cannot rebind it; TLS keeps SNI,
  the certificate check and `Host` on the hostname;
- redirects are followed by hand, at most 5, each hop checked again;
- only `text/html`, `application/xhtml+xml` and `text/plain` are read, at
  most `SEARCH_FETCH_MAX_BYTES` (2 MB) after decompression, which is
  bounded as it inflates; bodies stay in memory;
- `trust_env` is off: an `HTTP(S)_PROXY` would route around all of it;
- no cookies are kept: one client serves every user and chat, so a
  `Set-Cookie` from one fetch must never go out with another.

httpx has no public hook for the network backend, so the fetcher's
transport swaps its connection pool for one that has it. httpx and httpcore
are pinned in `uv.lock`, and `tests/test_page_fetcher.py` fails
if the swap stops taking effect.

Text comes out of **trafilatura 2.2.0** (fast, precision-first, no comments,
tables or links), imported lazily inside its own two-thread pool: the
import takes about a second and some 30–50 MB, which a cold start must not
pay and the event loop must not wait for. The body is decoded here first
(Content-Type charset, `<meta>` charset, UTF-8; when none decodes every
byte, the one losing fewest characters, a guess only when even that loses
many, and a page still mostly undecodable is a failed read), because
trafilatura ignores `<meta>` and guesses, which goes wrong on Big5 pages.
When trafilatura is missing, fails or finds nothing, a small
readability-style extractor takes over.

### Optional rerank

With `SEARCH_RERANK_ENABLED` (off by default), the embed role
(nemotron-3-embed-1b) embeds the query once and the titles and snippets in
one batch, and the results are sorted by similarity before the top ones
are read; a 3 s budget, and any failure keeps Tavily's order. Off because
it adds two embedding calls per search against NVIDIA's shared budget for
a gain not yet measured on zh-TW.

## Consequences

- Tavily, at 1 credit a search (basic depth) and 1 per 5 pages extracted,
  is the only search cost: the free plan's 1,000 credits a month, and
  keyless mode makes the key optional. `WEB_SEARCH_MONTHLY_QUOTA` is gone
  (a startup note names a stale one).
- GCP cost of search: nothing. No VM, no external IPv4, no Direct VPC
  egress; Cloud Run keeps its own internet egress for LINE, NVIDIA,
  OpenRouter and Tavily.
- When Tavily is rate limited, out of credits or off, replies go without
  search and say so; there is no free fallback search.
- The image grows by about 50 MB (trafilatura's dependencies, mostly
  Babel and lxml), all manylinux wheels. `tiktoken` and the Tavily SDK
  are gone.
- An old `.env` that still sets `SEARXNG_BASE_URL`,
  `SEARXNG_TIMEOUT_SECONDS` or `SEARXNG_TIMEOUT_LIMIT_SECONDS` keeps
  working: the keys are no longer read, startup and `envfile.py check-v3`
  name them, and `migrate-v3` deletes them.

## History: SearXNG first (2026-09-29, superseded 2026-10-01)

As first accepted, the chain had three levels: a self-hosted SearXNG on
the Always Free e2-micro in us-west1, reached at a reserved private
address over Direct VPC egress, then Tavily, then no search. Searches cost
nothing per query while SearXNG answered, and Tavily only covered its
gaps. It came with a provisioning path (an egress subnet, an internal
address, firewall rules scoped to the subnet and to SSH through IAP, the
VM itself, a lockdown of `default-allow-internal`), a VM runbook, and
deploy-time VPC flags.

On 2026-10-01 the deployment chose a strict GCP free tier instead. The
VM's ephemeral external IPv4 (needed to reach the search engines) is
billed past 720 hours a month (about US$0.12 in a 31-day month) and
inter-zone Direct VPC traffic is billed per GiB, so SearXNG kept the
project just outside the free tier; upstream engines also CAPTCHA
self-hosted instances, which needed a person to clear. The backend, its
settings, its runbook and VM files, the provisioning steps and the
deploy's VPC wiring were removed rather than kept as an unused option.
A Cloud Run service once deployed with them keeps its VPC connection until
`gcloud run services update --clear-network`, and a VM created by the old
steps is deleted by hand.
