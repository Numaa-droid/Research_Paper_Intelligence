# Claude Code Build Prompt: PaperPilot

## Role

You are a senior AI/ML platform engineer. Build a production-minded MVP named **PaperPilot**: a verifiable, citation-grounded, multi-agent system for analyzing research papers.

Do not claim that this system is Artificial General Intelligence. The system is a domain-specific multi-agent research-paper intelligence application.

Implement the project incrementally. Before writing substantial code, inspect the repository, identify existing conventions, and create a short implementation plan. Do not replace existing project files unless necessary. Prefer small, testable modules with clear interfaces.

## Product objective

A user uploads one research-paper PDF. PaperPilot parses and indexes the paper, dynamically plans an analysis, invokes specialized agents, verifies factual claims against the original paper, and returns a structured research report with page-level evidence.

The system must prioritize factual grounding over fluent prose. Every important factual statement in the final report should either have source evidence or be explicitly marked as an inference, uncertainty, or unsupported claim.

## MVP scope

Implement the following workflow first:

1. Upload one PDF.
2. Extract text with page boundaries.
3. Use OCR as a fallback for pages with insufficient native text.
4. Detect basic sections, equations, tables, figures, captions, and references where practical.
5. Build a paper index with dense and keyword retrieval.
6. Let an Orchestrator create an analysis plan.
7. Run the Summary Agent, Methodology Agent, and Experiment Agent.
8. Activate the Math Agent conditionally when the paper contains meaningful mathematical content.
9. Collect structured findings.
10. Send high-impact claims to the Verification Agent.
11. Retry selected analyses when claims are unsupported or weakly evidenced.
12. Generate a final Markdown report with page, section, table, figure, and equation evidence.
13. Expose the workflow through a simple API and a minimal usable interface or CLI.
14. Include automated tests and an evaluation harness.

Do not begin with multi-paper search, web browsing, autonomous code execution, or a complex frontend. Make the single-paper workflow reliable first.

## Non-goals

Do not build AGI.

Do not describe the system as human-level or generally intelligent.

Do not allow agents to invent page numbers, quotations, table values, equations, or experimental results.

Do not make every agent process the entire PDF independently when targeted retrieval is sufficient.

Do not use an LLM for deterministic tasks such as page numbering, schema validation, numeric parsing, or checking whether a citation refers to an existing page.

Do not add a large framework merely for appearance. Every dependency must support a concrete requirement.

## Recommended technical defaults

Use the following defaults unless the existing repository requires alternatives:

| Concern | Default |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI |
| Orchestration | LangGraph or a small typed state graph with an adapter for LangGraph |
| Validation | Pydantic v2 |
| PDF parsing | PyMuPDF |
| OCR | Tesseract fallback, isolated behind an interface |
| Vector retrieval | Qdrant or a local FAISS implementation |
| Keyword retrieval | BM25 |
| Reranking | FlashRank, optional |
| Persistence | SQLite for MVP, with repository interfaces for PostgreSQL later |
| Cache and workflow state | In-memory first; Redis adapter optional |
| LLM provider | Provider abstraction supporting OpenAI-compatible APIs and local models |
| Embeddings | Configurable embedding provider |
| Testing | pytest |
| Packaging | pyproject.toml |
| Deployment | Dockerfile and docker-compose only if useful |

If an existing repository already has an established stack, preserve it and adapt these choices rather than rewriting the project.

## Architecture

Implement these layers:

```text
API or CLI
    |
    v
Application service
    |
    v
Orchestrator / state graph
    |
    +--> Paper ingestion service
    +--> Paper intelligence index
    +--> Summary Agent
    +--> Methodology Agent
    +--> Experiment Agent
    +--> Conditional Math Agent
    +--> Verification Agent
    +--> Report Synthesizer
    |
    v
Storage, retrieval, model provider, and observability adapters
```

### Paper intelligence layer

Create a normalized representation of the paper before running agents. At minimum, store:

- Document identifier.
- Page number.
- Extracted text.
- Section and subsection when detected.
- Element type: paragraph, table, figure, equation, reference, or unknown.
- Element identifier when available.
- Character offsets or bounding box when available.
- Extraction method: native text or OCR.
- Extraction confidence.

Use page-aware chunks. Do not split away the page metadata. Preserve the original page numbering and clearly document whether page numbers are PDF indices or printed page labels.

Example indexed chunk:

```json
{
  "chunk_id": "paper-123-page-7-chunk-02",
  "document_id": "paper-123",
  "page": 7,
  "section": "4.2 Main Results",
  "element_type": "table",
  "element_id": "table-3",
  "text": "...",
  "source": "native_text",
  "bbox": null,
  "confidence": 0.98
}
```

Implement a retriever interface with:

- Semantic search.
- BM25 or exact-term search.
- Metadata filtering by page, section, and element type.
- Optional reranking.
- A merged result format with scores and source metadata.

## Agent contracts

All agents must return validated Pydantic models. Do not pass unstructured agent prose between nodes.

Each agent receives only the relevant task, paper metadata, retrieved evidence, and prior structured state required for its work.

### Evidence model

```python
class Evidence(BaseModel):
    page: int
    section: str | None = None
    source_type: Literal["paragraph", "table", "figure", "equation", "reference", "unknown"]
    element_id: str | None = None
    quote: str
    chunk_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
```

The evidence quote must be copied from retrieved paper text or clearly marked as a normalized representation. Agents must not fabricate quotations.

### Finding model

```python
class Finding(BaseModel):
    claim: str
    explanation: str
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    status: Literal[
        "supported",
        "partially_supported",
        "unsupported",
        "contradicted",
        "inferred",
        "unknown"
    ] = "unknown"
```

### Summary Agent

Extract and explain:

- Research problem.
- Motivation.
- Main contribution.
- Dataset or data source.
- Proposed method.
- Main results.
- Limitations.
- Assumptions.

Return a `SummaryResult` containing findings, a concise summary, and unresolved questions.

### Methodology Agent

Explain the proposed method step by step:

- Inputs and outputs.
- Architecture or pipeline.
- Training procedure.
- Inference procedure.
- Important design decisions.
- Comparison with prior approaches.
- Pseudocode or a numbered execution flow when supported by the paper.

Do not infer implementation details that the paper does not state. Mark reasonable interpretation as `inferred`.

### Experiment Agent

Analyze:

- Dataset and splits.
- Baselines.
- Metrics.
- Main results.
- Ablations.
- Hyperparameters.
- Error analysis.
- Statistical significance when reported.
- Whether the conclusions are supported by the experiments.

Numbers must include evidence. Add deterministic numeric validation where possible.

### Math Agent

Activate conditionally. It should be skipped when no meaningful mathematical content is detected.

When activated, explain:

- Equations.
- Variables and symbols.
- Shapes or dimensions when available.
- Loss functions and optimization objectives.
- Derivation steps stated by the authors.
- Mathematical intuition.
- Differences between author-stated derivation and agent interpretation.

Preserve the source equation in a safe representation such as LaTeX, extracted text, or an explicit unavailable marker. Never silently reconstruct an unreadable equation.

### Verification Agent

Verify every high-impact claim against the indexed paper evidence.

For each claim, return:

- Claim text.
- Verification status.
- Supporting or contradicting evidence.
- Page and section.
- Citation quality.
- Confidence.
- Explanation of any mismatch.

Use the following decision rules:

- `supported`: evidence directly supports the claim.
- `partially_supported`: evidence supports only part of the claim.
- `unsupported`: no adequate evidence was found.
- `contradicted`: evidence conflicts with the claim.
- `inferred`: the claim is a reasonable interpretation but is not explicitly stated.
- `unknown`: verification could not be completed.

The verifier must not approve a citation only because the page contains related words. It must check semantic support and numerical consistency.

### Report Synthesizer

Generate a final Markdown report containing:

1. Paper metadata.
2. Executive summary.
3. Research problem and motivation.
4. Contributions.
5. Methodology.
6. Mathematical explanation, if applicable.
7. Dataset and experimental setup.
8. Results and ablations.
9. Limitations and threats to validity.
10. Reproducibility checklist.
11. Verified findings.
12. Unsupported, contradicted, or inferred findings.
13. Open questions.
14. Evidence appendix.

Use citations in the format `[p. 7, §4.2, Table 3]`. Generate these citations only from validated evidence objects.

## Orchestrator behavior

Implement a typed state object similar to:

```python
class ResearchState(TypedDict, total=False):
    document_id: str
    file_path: str
    paper_metadata: PaperMetadata
    index_status: str
    analysis_plan: AnalysisPlan
    retrieved_context: dict[str, list[Evidence]]
    agent_results: dict[str, AgentResult]
    findings: list[Finding]
    verification_results: list[VerificationResult]
    retry_count: int
    final_report: str
    errors: list[str]
```

The Orchestrator must:

1. Validate the uploaded file.
2. Ingest and index the paper.
3. Inspect metadata and available paper elements.
4. Create an analysis plan.
5. Activate specialist agents.
6. Aggregate results.
7. Extract high-impact claims.
8. Verify claims.
9. Retry only failed or weakly evidenced tasks.
10. Stop after a bounded retry count.
11. Synthesize the final report.

The plan should contain conditional decisions such as:

```json
{
  "run_summary": true,
  "run_methodology": true,
  "run_experiments": true,
  "run_math": true,
  "run_table_analysis": true,
  "priority_sections": ["Method", "Experiments", "Appendix"],
  "verification_threshold": 0.85,
  "max_retries": 1
}
```

Use parallel execution for independent specialist agents when the chosen orchestration framework supports it safely. Do not parallelize operations that depend on prior state.

## Prompting rules

Create versioned prompts in separate files or constants. Prompts must instruct agents to:

- Use only supplied paper context.
- Cite evidence for factual claims.
- Distinguish explicit statements from inferences.
- Preserve numbers exactly.
- Avoid filling missing information from general knowledge.
- Return the required structured schema.
- Report uncertainty honestly.

Include the agent name, prompt version, model name, and execution metadata in traces.

## Deterministic validation

Implement non-LLM validation for:

- Page numbers that fall within the document range.
- Evidence quotes that occur in the indexed text where possible.
- Valid section and element references.
- Numeric values copied from evidence.
- Required output fields.
- Confidence range validation.
- Retry limits.
- File size and file type checks.

If exact quote matching fails because of OCR normalization, mark the evidence as approximate and lower its confidence rather than accepting it silently.

## API requirements

Implement a minimal API with endpoints similar to:

```text
POST /papers
GET  /papers/{document_id}
POST /papers/{document_id}/analyze
GET  /analyses/{analysis_id}
GET  /analyses/{analysis_id}/report
GET  /health
```

The upload response should return a document identifier. Analysis should return an analysis identifier and status. Long-running analysis should not block indefinitely; use a background task abstraction or a polling-friendly status model.

For the MVP, local filesystem storage is acceptable. Keep storage behind interfaces so it can later be replaced by object storage.

## CLI requirements

Also provide a simple CLI path for local development, such as:

```bash
python -m paperpilot analyze ./papers/example.pdf --output ./reports/example.md
```

The CLI must print progress by stage and return a nonzero exit code for failures.

## Configuration and security

Use environment variables for secrets and provider configuration. Provide `.env.example` without real credentials.

Never commit API keys, uploaded documents, generated reports containing sensitive data, or local databases.

Add `.gitignore` entries for temporary uploads, indexes, caches, logs, `.env`, and generated artifacts.

Validate PDF MIME type and file size. Do not execute content from uploaded documents.

## Testing requirements

Write tests before declaring the MVP complete.

### Unit tests

Cover:

- PDF page extraction.
- OCR fallback decision.
- Chunk metadata preservation.
- Section detection.
- Retrieval filtering.
- Pydantic schema validation.
- Citation formatting.
- Evidence quote matching.
- Numeric validation.
- Conditional Math Agent activation.
- Retry and stopping logic.

### Integration tests

Use a small fixture paper or synthetic paper fixture. Test the complete workflow with a mocked model provider.

The integration test must verify that:

- A paper is ingested.
- At least three agents run.
- The Math Agent is conditionally activated or skipped correctly.
- Findings contain evidence.
- Verification changes the status of at least one finding.
- The final report contains page citations.

### Evaluation harness

Create an evaluation script that can process a directory of benchmark papers and save JSON results.

Measure at least:

| Metric | Definition |
|---|---|
| Claim coverage | Important expected claims present in the report |
| Citation precision | Citations that actually support their claims |
| Citation recall | Factual claims that contain evidence |
| Numeric accuracy | Reported numbers matching source values |
| Unsupported-claim rate | Claims without adequate support |
| Verification accuracy | Correct supported and unsupported classifications |
| Latency | End-to-end execution time |
| Estimated cost | Model usage cost when provider data is available |

Make the evaluator provider-independent. Store raw outputs, normalized findings, verification results, and aggregate metrics.

## Observability

Log structured events for:

- Ingestion started and completed.
- Number of pages and chunks.
- OCR pages.
- Retrieval queries and result counts.
- Agent start and completion.
- Token usage when available.
- Verification outcomes.
- Retries.
- Final report completion.
- Errors.

Do not log secrets or full sensitive document text by default. Use document and chunk identifiers.

## User experience

The first interface may be a CLI or minimal HTML page. It must show:

- Upload status.
- Parsing status.
- Indexing status.
- Active agents.
- Verification status.
- Retry status.
- Final report link.

A user should be able to understand why an agent was run and why a claim was marked unsupported.

## Suggested repository structure

Adapt to the existing repository, but aim for a structure similar to:

```text
paperpilot/
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── src/
│   └── paperpilot/
│       ├── api/
│       ├── cli.py
│       ├── config.py
│       ├── domain/
│       │   ├── models.py
│       │   ├── enums.py
│       │   └── state.py
│       ├── ingestion/
│       │   ├── pdf_parser.py
│       │   ├── ocr.py
│       │   └── elements.py
│       ├── retrieval/
│       │   ├── index.py
│       │   ├── bm25.py
│       │   ├── vector.py
│       │   └── retriever.py
│       ├── agents/
│       │   ├── base.py
│       │   ├── summary.py
│       │   ├── methodology.py
│       │   ├── experiments.py
│       │   ├── math.py
│       │   ├── verification.py
│       │   └── prompts/
│       ├── orchestration/
│       │   ├── graph.py
│       │   ├── planner.py
│       │   └── retry.py
│       ├── reporting/
│       │   ├── synthesizer.py
│       │   └── citations.py
│       ├── providers/
│       │   ├── llm.py
│       │   └── embeddings.py
│       └── validation/
├── tests/
├── fixtures/
├── scripts/
└── reports/
```

## Development milestones

Implement in this order:

### Milestone 1: Foundation

Create project configuration, domain models, provider interfaces, configuration loading, logging, and a health check.

### Milestone 2: Ingestion and retrieval

Implement PDF extraction, page-aware chunks, OCR fallback interface, metadata, local index, BM25 retrieval, and tests.

### Milestone 3: Single-agent baseline

Implement a grounded Summary Agent and report output. This baseline is important for comparing whether the multi-agent design improves quality.

### Milestone 4: Specialist agents

Add Methodology, Experiment, and conditional Math Agents with typed outputs.

### Milestone 5: Orchestration

Implement planning, routing, aggregation, bounded retries, and progress events.

### Milestone 6: Verification

Implement claim extraction, evidence retrieval, citation validation, numeric checks, and Verification Agent.

### Milestone 7: Evaluation and interface

Add benchmark evaluation, API/CLI usability, documentation, sample output, and Docker support if appropriate.

Do not move to the next milestone while the previous milestone lacks tests and a working demonstration.

## Definition of done

The MVP is complete only when all of the following are true:

- A user can analyze a local research-paper PDF from the CLI.
- The API can accept a PDF and expose analysis status.
- The paper is indexed with page-aware metadata.
- The Orchestrator creates a plan rather than blindly running all agents.
- At least three specialist agents produce typed results.
- Math analysis is conditional.
- The Verification Agent checks important claims against paper evidence.
- Unsupported claims are explicitly labeled.
- The final report contains valid page-level evidence.
- The system has bounded retries and clear failure handling.
- Unit and integration tests pass.
- A mocked end-to-end test runs without external model credentials.
- `.env.example`, README, setup instructions, and architecture documentation exist.
- No secrets or generated sensitive files are committed.

## README requirements

Document:

- Product overview.
- What the system is and is not.
- Architecture diagram in Mermaid.
- Local setup.
- Environment variables.
- CLI usage.
- API usage.
- Agent responsibilities.
- State graph.
- Evidence and verification design.
- Evaluation methodology.
- Known limitations.
- Future extensions.

Include a short interview explanation:

> PaperPilot is a domain-specific, citation-grounded multi-agent system. Its Orchestrator dynamically decomposes paper analysis, routes targeted evidence to specialized agents, verifies high-impact claims against the original paper, and retries weak analyses before producing a structured report.

## Engineering principles

Prefer correctness, observability, and testability over feature count.

Keep all model calls behind interfaces so a mocked provider can run the tests.

Use deterministic validation wherever possible.

Make uncertainty visible to users.

Preserve evidence throughout the workflow.

Use small commits or logically separated changes if Git is available.

At the end of each milestone, run tests and provide a concise summary of what changed, what passed, and what remains.

## First action

Start by inspecting the repository and runtime environment. Then:

1. State the detected project structure.
2. Propose the first milestone plan.
3. Implement only Milestone 1 initially.
4. Run the available tests and linters.
5. Show the files changed and the commands used for verification.
6. Ask for no unnecessary confirmation; proceed with reasonable engineering defaults.

Do not generate a fake completion summary. Report actual implementation status and test results.
