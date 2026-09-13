# PaperPilot architecture

This document explains *why* the system is shaped the way it is. The README
covers what it does and how to run it.

---

## The central constraint

A language model asked to summarise a paper will produce fluent, plausible prose
containing page numbers, table values and quotations that may or may not exist.
The failure is invisible at the point of reading — which is exactly what makes it
dangerous for research work.

Every significant design decision here follows from one rule:

> **A model may propose a claim. It may not be the only thing that decides whether
> that claim is true of the paper.**

So the pipeline is built as a funnel. Model output enters at the top and passes
through progressively narrower, progressively more deterministic gates. Anything
that fails a gate is not deleted — it is **labelled and demoted**, so the failure
stays visible to the reader.

```
model output
    → schema coercion         (malformed evidence is dropped, not defaulted)
    → quote matching          (is this text actually in the paper?)
    → page/section/element    (does this location exist?)
    → numeric equality        (do these numbers appear in the evidence?)
    → independent verification(does re-retrieved evidence support the claim?)
    → deterministic override  (checks cap the verdict; they never inflate it)
    → bounded retry           (rejected agents re-run with specific feedback)
    → deterministic rendering (the report has no model call at all)
```

---

## Layer boundaries

```
API / CLI  ──────────────────► application service
                                      │
                                      ▼
                            orchestrator (state graph)
                                      │
        ┌──────────────┬──────────────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼              ▼
   ingestion    paper index    specialist      verification    synthesizer
                               agents            agent
        │              │              │              │              │
        └──────────────┴──────────────┴──────────────┴──────────────┘
                                      │
                                      ▼
                 adapters: storage · model provider · embeddings · OCR
```

Dependencies point inwards. The domain models know nothing about FastAPI,
SQLite, PyMuPDF or any model provider. Every outward dependency sits behind an
interface (`LLMProvider`, `EmbeddingProvider`, `OCREngine`, `Retriever`,
`DocumentRepository`, `AnalysisRepository`, `FileStorage`, `TaskRunner`), which
is what makes the whole suite runnable with no credentials and no network.

---

## Decisions worth defending

### Why a hand-written state graph instead of LangGraph

The workflow needs three things: named nodes over a typed state, conditional
edges, and safe parallel execution of independent nodes. That is about 120 lines
(`orchestration/engine.py`), with no dependency, no version churn and a stack
trace that points at our own code.

LangGraph is still supported: `orchestration/langgraph_adapter.py` rebuilds the
*same node functions* as a LangGraph graph for anyone who wants its
checkpointing, streaming or tracing. Because the nodes are the same objects,
behaviour cannot drift between the two engines.

This follows the brief's rule that a dependency must earn its place.

### Why nodes return partial state

A node returns only the keys it changed, and `merge_state` folds them in:
append-only keys (`events`, `usage`, `errors`) accumulate, dict keys
(`agent_results`, `retrieved_context`) merge key-wise, everything else is
replaced. Parallel branches therefore cannot clobber each other, and the merge
order is fixed so runs stay reproducible.

### Why specialist agents run in parallel but retrieval is targeted

The four specialists never read each other's output, so they are safe to run
concurrently — and being IO-bound on model calls, a thread pool is the right
tool.

What they must *not* do is each re-read the whole paper. Each agent issues a
handful of targeted retrieval queries and receives a bounded context
(`MAX_CONTEXT_CHARS`). That is cheaper, but more importantly it is what makes
citations checkable: an agent that was handed chunk `page-4-chunk-03` and quotes
from it can be audited; an agent handed 40 pages cannot.

### Why the planner cannot enable an agent

The planner may *disable* a specialist it thinks is unnecessary, and it may
reorder priority sections. It may not *enable* one for content that does not
exist in the index.

Whether a paper contains three equations is a fact about the document, established
by parsing. Letting a model overrule it would reintroduce exactly the failure
mode the system exists to prevent. The heuristic plan is therefore always
computed first and always survives as the floor; the model refines within it.

### Why the Math Agent's activation is deterministic

Same reason, plus cost: running a Math Agent on a qualitative paper spends tokens
to produce filler. The three-branch rule (equation count, equations + vocabulary,
vocabulary alone) has a deliberate escape hatch for scanned papers whose
equations did not survive extraction — the prose still says the maths is there.
The decision and its reason are always reported.

### Why the verifier re-retrieves

If a claim's evidence is chosen by the same agent that wrote the claim, the
"verification" is a self-assessment. The verifier searches the index again from
the claim text — plus exact lexical lookups for any numbers in it, since numbers
are strong anchors — and rules on what *it* found.

### Why deterministic checks can only cap a verdict

The model sees semantics; the checks see facts. A model can reasonably judge
whether a passage supports a claim. It cannot be the authority on whether `81.4`
appears in the paper, because that question has an exact answer.

So the reconciliation is asymmetric: a check can downgrade `supported` to
`partially_supported` or `unsupported`, but nothing can promote a verdict the
checks reject. A "supported" verdict with no verifiable quote becomes
`unsupported`, and the reason is recorded.

### Why the report synthesizer makes no model call

The final artefact is rendered deterministically from validated structures. It is
therefore *structurally incapable* of introducing a claim, a number or a citation
that did not survive verification — there is nothing for it to invent from.

It also means the report's shape is testable, and that the same findings always
render the same way.

### Why failures are shown rather than removed

Section 11 of every report lists what could not be grounded, with the reason and
any numbers that were not found. Dropping those claims would produce a cleaner
document that silently hid the system's failures. The unsupported-claim rate is a
headline evaluation metric for the same reason.

### Why element cohesion exists in the retriever

A table is two chunks: a caption (prose, wins on lexical similarity) and a body
(numbers, loses). An agent that retrieves only the caption can cite "Table 1" but
cannot read a value from it — so it either omits the number or invents one.
Retrieving either chunk now pulls in its siblings. This was found by watching the
Experiment Agent produce an empty results table on the fixture paper.

### Why the mock provider is evidence-driven

A stub returning canned strings would make the integration test assert that
fixtures round-trip. Instead the mock reads the evidence it was given and builds
answers from verbatim sentences in it, so quote validation, numeric checking,
citation rendering and report synthesis are all genuinely exercised offline.

It also emits exactly one fabricated claim per run. A verification system that is
never handed a bad claim has not been tested, so the integration suite asserts
that the planted claim is caught, labelled `unsupported`, and still visible in
the report.

### Why page numbers are PDF indices

Printed labels are ambiguous (roman front matter, offsets, per-section numbering)
and frequently absent. PDF indices are unambiguous and always available, which
matters because every range check and every citation depends on them. Printed
labels are preserved alongside as `printed_label`, and the report states the
convention rather than leaving the reader to guess.

### Why whitespace is preserved during extraction but collapsed for matching

Runs of spaces in extracted PDF text are how column alignment survives — the
table detector depends on them. But quote comparison must be tolerant of
whitespace differences. So `normalize_text` (storage) keeps spacing, and
`normalize_for_match` (comparison) collapses it. Both sides of a comparison go
through the same normaliser, so indexed text and agent quotes are compared like
with like.

---

## Data flow of one claim

Following a single claim from model output to printed citation:

1. **Summary Agent** returns JSON containing a claim and an evidence object.
2. `parse_finding` coerces it into typed models. Evidence with a non-integer page
   or an empty quote is **dropped**, not defaulted.
3. `validate_evidence` resolves the owning chunk (even when the agent did not say
   which), checks the page is in range, matches the quote in tiers, corrects the
   section label from the index, and drops an element id that does not belong to
   the cited text.
4. `validate_finding` reconciles status with the result: no verified evidence →
   `unsupported`; a number missing from the evidence → `partially_supported`;
   confidence capped at its best evidence.
5. `extract_high_impact_claims` scores it — numbers, superlatives, universal
   quantifiers and missing evidence all raise its priority.
6. **Verification Agent** re-retrieves evidence for it independently, runs the
   numeric check, asks the model, then applies the deterministic override.
7. `apply_verification` replaces the authoring agent's self-assessment with the
   verifier's verdict.
8. `decide_retry` may re-run the authoring agent with the rejected claim quoted
   back at it.
9. **Synthesizer** renders `[p. 4, §4.2 Main Results, Table 1]` — or, if the
   evidence never passed, places the claim in the unsupported section instead.

---

## Testing strategy

348 tests, no network, no credentials.

- **Unit** — extraction, OCR decisions, structure detection, chunk metadata,
  BM25, dense retrieval, filtering, schema coercion, quote matching, numeric
  checks, citation formatting, planning, retry logic, the graph engine.
- **Adversarial** — fabricated quotes, out-of-range pages, invented element ids,
  wrong numbers, over-confident verifiers, malformed provider output, provider
  outages, hostile filenames, non-PDF uploads. These assert that the *checks*
  work, which is the actual product.
- **Integration** — the full workflow on a synthetic paper of known structure,
  asserting that three or more agents run, math activation is conditional,
  findings carry evidence, verification changes at least one status, and the
  report contains valid page citations.
- **Evaluation** — the harness scores itself against known ground truth, so a
  regression in grounding shows up as a metric change rather than a test failure
  nobody reads.

The synthetic fixture (`fixtures/synthetic_paper.py`) is *generated from source*
rather than committed as a binary, so its expected sections, tables, figures and
equations stay visible and reviewable in code.
