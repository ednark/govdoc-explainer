# search-v2 experiment: int8 vectors + lazy loading

**Status: promoted to production (2026-09-24).** Round 3 of the benchmark
confirmed the design end-to-end: 23 of 24 fp32-vs-int8 comparisons at 100%
top-8 agreement (the one 88% is a single near-tie flip), 67–76% smaller cold
payloads, latency parity. The promoted pipeline is:

- `embeddings.py` writes `embedding_q8.json` at build time (both per-doc and
  `assets/embedding_q8.json`) — canonical quantizer, round-trip verified
  against the experiment script.
- `assets/semantic_search.js` lazily fetches `embedding_q8.json` on first
  search interaction (prefetch on input focus), ranks on raw int8, falls back
  to the fp32 file for older builds. Asset version bumped to `?v=12`.
- This directory remains as the regression harness: re-run
  `benchmark.html` after any future model or format change.

## What's being compared

| | production (status quo) | v2 experiment |
|---|---|---|
| Vector format | fp32 floats in JSON | symmetric int8 + per-vector scale (base64) |
| Vector fetch | eager, on DOMContentLoaded — every visitor pays | lazy, on first search interaction |
| Model | `Xenova/all-MiniLM-L6-v2`, 23 MB int8, lazy, cached | same (dtype pinned explicitly) |
| Ranking | fp32 dot product | int8 dot (scale cancels in cosine) |

Why int8 is safe for ranking: cosine similarity is scale-invariant per vector,
so `cos(q, s·v) === cos(q, v)` — the client ranks on raw int8 bytes and never
dequantizes. Quantization error is bounded at ±0.4% of each vector's max
component; `--verify` measures the actual effect.

## Files

- `quantize_embeddings.py` — converts `assets/embedding.json` and every
  `sources/*/embedding.json` to `embedding_q8.json` siblings; `--verify` reports
  cosine deviation and top-8 neighbour agreement against fp32.
- `semantic_search_v2.js` — search module mirroring `assets/semantic_search.js`
  behaviour with lazy loads + dual format support. Exports `initEmbedder`,
  `embedQuery`, `loadVectors`, `rankVectors`.
- `benchmark.html` — in-browser harness (payload load, model init, search
  latency, top-8 agreement between fp32 and int8 paths).

## Running

```sh
npm run serve          # pagefind --serve on the repo root
# open http://127.0.0.1:PORT/experiments/search-v2/benchmark.html
```

Click sections in order: 1 (payload), 2 (model, one-time), 3 (search).
Do one "cold" pass, then reload the page and repeat for warm-cache numbers.

Expected before/after (worst page, NIST SP 800-53):

| stage | fp32 eager | int8 lazy |
|---|---|---|
| page load cost for non-searchers | ~8.6 MB | 0 bytes |
| search payload | ~8.6 MB | ~2.2 MB |
| model | see "Model download" below | same |

## Production integration (if results are accepted)

1. `render.py`: reference a v2 script instead of `semantic_search.js`, drop the
   eager `loadEmbeddings` on DOMContentLoaded, keep the same DOM ids.
2. `embeddings.py`: ~~write int8 format directly at build time~~ (skip the fp32
   JSON entirely) — still open; build-side clients never need the fp32 JSON.
3. ~~Delete the dead `assets/lunr_index.json`~~ — **done 2026-09-24**: lunr
   generation removed from `render.py`/`cli.py` (it was referenced by no JS),
   1.9 MB no longer shipped.

## Client-side search index landscape (2026)

Where lunr.js and its peers sit, relative to what this site already does:

| Library | Approach | Index size | Notes for govdoc-explainer |
|---|---|---|---|
| **Pagefind** (in use) | build-time inverted index, chunked WASM fragments, loads only fragments needed per query | ~0 on page load; lazy fragments | Best default for static sites; multilingual out of the box (German sources work). Already our keyword channel. |
| **lunr.js** (artifact shipped, unused) | inverted index built in-browser from a JSON corpus (our `lunr_index.json`, 1.9 MB) | large — full index in memory | Mature, stemming + 14 languages via plugins, but slow by modern benchmarks (~11k q/s vs MiniSearch ~30k) and effectively unmaintained (Hugo community flagged abandonment in 2025). Its index is our *only* eagerly-loaded search payload. |
| **MiniSearch** | in-memory inverted index, BM25-ish ranking, fuzzy + prefix + suggestions | much smaller than lunr's | Fastest well-maintained pure-JS option; would replace lunr if we ever need client-built indexes over dynamic data. |
| **FlexSearch** | highly optimized inverted index, workers, phonetic matching, persistent IndexedDB index | 4.5–16 kB gzip lib | Fastest query throughput of all (their benchmark: ~500k q/s single-term); heavier config surface. |
| **Fuse.js** | no inverted index — linear fuzzy scan over JSON | no index; needs full documents | Only sensible for tiny datasets; memory-hungry (benchmark: 247k units vs FlexSearch 16). Not relevant at our scale. |
| **Orama** | full-text **+ vector/hybrid** search in JS/WASM | moderate | The only mainstream client lib with native vector + hybrid search; overlaps with what our bespoke MiniLM cosine does, without model-choice control (same-model-both-sides constraint). |
| **elasticlunr** | lunr fork, better boosting | large | Unmaintained; skip. |

Takeaways:

1. **Pagefind + our baked vectors is already the right architecture** — the
   dominant client-side libs either need a build-time index (Pagefind wins) or
   an in-memory index over full documents (lunr/MiniSearch — why our unused
   lunr artifact is pure cost).
2. Nothing in the landscape changes the embeddings story: client-side *vector*
   search at our scale means "precomputed vectors + tiny query encoder", which
   is what search-v2 optimizes.
## Benchmark round 2 findings (post model switch, per-doc pages)

Round 2 on per-doc pages (NIST 800-53, DFARS) still showed 50–88% fp32-vs-int8
agreement while the home page hit 100%. Root cause — same bug class, second
instance: `paraphrase-multilingual-MiniLM-L12-v2` ships **without a Normalize
module**, so fastembed returns section vectors with norm 2.7–4.7. Storing them
raw made the per-doc fp32 dot product rank by `cosine × section norm` again.
Fix: `generate_embeddings_for_text_sections` now L2-normalizes every section
vector before storing. After re-embedding all 140 sources with cached text:

- per-doc and home vector norms: all 1.000
- database-side int8-vs-fp32 top-8 agreement: ≥ 99% everywhere, 100% for the
  majority of files
- expected client agreement after this fix: ~100%, with sub-100% only inside
  near-tie clusters (NIST/DFARS contain many near-duplicate control statements
  whose similarity scores differ by less than quantization noise)

Latency note: on the large per-doc pages the int8 scan measured ~0.2 ms slower
than fp32 (0.5 → 0.7 ms) — irrelevant next to the 73–75% payload reduction
(8.81 MB → 2.18 MB cold on NIST 800-53).

## Benchmark round 1 findings (home aggregate, all-MiniLM-L6-v2)

Payload and latency confirmed the v2 design (60% smaller cold fetch, 15 ms → 3 ms;
ranking scan ~0.1 ms either way). The low fp32-vs-int8 "top-8 agreement" (13–63%)
was **not** quantization error — it exposed a production bug:

- `assets/embedding.json` vectors were **sums** of section embeddings, never
  re-normalized (norms 0.0–515, median 3.6), while the client's `cosineSimilarity`
  is a plain dot product. Home-page ranking was therefore `cosine × norm` —
  heavily biased toward chunk-heavy documents. Per-doc pages were unaffected
  (their vectors are unit-norm).
- The int8 path ranks by true cosine (scales cancel), so it was the *correct*
  one; the fp32 production path was the distorted one.
- Fix: `generate_main_embeddings` now stores the **mean of section embeddings,
  L2-normalized**, making the client dot product equal cosine. Both paths should
  agree near-100% after the rebuild.

## Model: paraphrase-multilingual-MiniLM-L12-v2 (switched 2026-09-24)

Both build (`embeddings.py`) and client (semantic_search.js + v2 module) now use
`paraphrase-multilingual-MiniLM-L12-v2` — same 384-dim, XLM-R vocab (250k tokens),
proper German support for the GDPR/EU-AI-Act/DSA pages.

**Model download cost (corrected — the ~30 MB estimate earlier in this file was wrong):**
the 250k vocab's embedding table dominates: int8 ONNX is **118 MB** plus
~32 MB of tokenizer assets (tokenizer.json 17 MB + unigram.json 15 MB), ≈ **150 MB
one-time, browser-cached** — 6.5× the 23 MB all-MiniLM-L6-v2 needed. fp16 would be
235 MB; q4 variants are larger, not smaller. This is the floor for any
XLM-R-vocab multilingual encoder client-side.

Switching the model invalidated every stored vector: all `embedding.json` files
were regenerated from cached `.txt` extractions during the rebuild (embeddings are
model-specific — never mix vectors from two models in one index).