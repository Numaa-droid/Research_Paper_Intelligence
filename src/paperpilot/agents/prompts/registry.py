"""The prompt registry.

Grounding rules are stated once in :data:`GROUNDING_RULES` and inherited by every
agent prompt, so the evidence policy cannot drift between agents.

Every prompt asks for **Markdown in a fixed layout**, never JSON. The layout is
the contract: headings name fields, tables carry rows, blockquotes carry
verbatim evidence. :mod:`paperpilot.response` parses it deterministically, and
each agent reads its fields from the parsed document by heading and column name.
"""

from __future__ import annotations

from dataclasses import dataclass

GROUNDING_RULES = """\
You are one specialist in a citation-grounded research-paper analysis system.

Absolute rules:
1. Use ONLY the paper context supplied below. You have no other knowledge of this
   paper, and you must not fill gaps from general knowledge of the field.
2. Every factual statement you make must carry evidence copied from the supplied
   context. Quote the paper verbatim; never paraphrase inside a quote.
3. Cite sources ONLY by their labels from the context, e.g. [S3]. Never write page
   numbers, chunk ids or section names as citations — they are filled in
   automatically from the label.
4. Preserve numbers exactly as written in the paper, including decimal places and
   units. Do not round, convert or recompute them.
5. Distinguish what the paper states from what you infer:
   - supported : the context states this explicitly.
   - inferred  : a reasonable reading of the context, not stated outright.
   - unknown   : the context does not settle it.
   Mark honestly. An honest "unknown" is more useful than a confident guess.
6. If the context does not contain what you were asked about, write
   "Not stated." under that heading, and write no finding rather than invent one.
7. Keep the paper's own work apart from the work it cites. A source whose header
   says "REPORTS CITED WORK [N]" is the paper describing what another work did
   or found. State such content as that work's ("According to [N], …", "A study
   the paper cites ([N]) found …"), never as this paper's method, dataset,
   result, contribution or limitation. The paper's own work is what it does in
   its own voice ("we", "this article", its own tables and counts).

Output format — Markdown only:
- Answer in GitHub-flavoured Markdown using EXACTLY the headings of the template,
  in the same order. Do not rename, merge or skip headings; do not add a title.
- Do NOT output JSON, and do not wrap the answer in a code fence.
- Key-value lines are bullets written as "- Key: value".
- Evidence is a blockquote line that starts with the source label, followed by
  the verbatim quote, one quote per line:
  > [S3] exact sentence copied from source S3
- In table cells, escape a literal pipe as \\| and keep each row on one line.
- Tables in the context are Markdown rows ("| Model | 28.4 | 41.8 |"). To cite a
  table, quote the row lines you rely on exactly as written, one per line — never
  a description of the table.

Every quote you write is automatically re-checked against the original paper. An
unverifiable quote is reported to the user as an unsupported claim, so a cautious
citation is always better than an impressive one you cannot support."""

FINDINGS_TEMPLATE = """\
## Findings

### Finding 1
- Claim: <one self-contained claim>
- Status: supported | inferred | unknown
- Category: {category}
- Confidence: <0.0-1.0>
- Impact: <0.0-1.0, how much a reader relies on this claim>
- Explanation: <one or two sentences>

> [S1] <verbatim quote supporting the claim>

### Finding 2
..."""


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """One versioned prompt."""

    name: str
    version: str
    system: str
    user_template: str

    def render(self, **kwargs) -> str:
        # The reference guide is optional: a caller without one renders without it.
        kwargs.setdefault("guidance", "")
        return self.user_template.format(**kwargs)


_CONTEXT_BLOCK = """\
{guidance}PAPER METADATA
{metadata}

RETRIEVED PAPER CONTEXT
Each block below is one source from the paper, labelled S1, S2, … . Cite a source
only by its label, e.g. [S2].

{context}
"""


def _findings(category: str) -> str:
    # Literal braces survive str.format() only when doubled.
    return FINDINGS_TEMPLATE.replace("{category}", category).replace("{", "{{").replace("}", "}}")


SUMMARY = PromptTemplate(
    name="summary",
    version="v4",
    system=GROUNDING_RULES,
    user_template=_CONTEXT_BLOCK
    + """
TASK
Summarise this paper for a researcher who has not read it. Write 5-10 findings,
each an individual claim with its own evidence.

Answer with exactly this Markdown template:

## Research Problem
<the problem the paper addresses>

## Motivation
<why the problem matters, as the authors argue it>

## Main Contribution
<what the authors claim is new>

## Dataset
<datasets or data sources used>

## Proposed Method
<the method, in two or three sentences>

## Main Results
<headline results, with the paper's own numbers>

## Limitations
<limitations the authors state or concede anywhere in the paper: costs and
trade-offs ("at the cost of"), settings where the method is worse or untested,
and what they leave to future work>

## Assumptions
<assumptions the method relies on>

## Concise Summary
<3-5 sentences a reader could quote>

## Unresolved Questions
- <what the paper leaves open>

"""
    + _findings("problem | contribution | dataset | method | results | limitations")
    + "\n",
)

METHODOLOGY = PromptTemplate(
    name="methodology",
    version="v5",
    system=GROUNDING_RULES,
    user_template=_CONTEXT_BLOCK
    + """
TASK
Explain the proposed method step by step, strictly from the context.

Do NOT infer implementation details the paper does not state. Where you must
interpret, set the step's "Inferred" to yes and the finding's status to inferred.

Answer with exactly this Markdown template:

## Inputs
<what the method takes as input>

## Outputs
<what it produces>

## Architecture
<components and how they connect>

## Training Procedure
<objective, optimiser, schedule>

## Inference Procedure
<what happens at test time>

## Design Decisions
- <a design decision the authors state>

## Comparison With Prior Work
<how the authors position the method>

## Pipeline Steps

### Step 1: <short name>
- Inferred: no

<what this step does, as a paragraph>

> [S1] <verbatim quote describing this step>

### Step 2: <short name>
...

## Pseudocode
```text
<numbered execution flow, or write "Not stated." instead of this block>
```

"""
    + _findings("methodology")
    + "\n",
)

EXPERIMENTS = PromptTemplate(
    name="experiments",
    version="v4",
    system=GROUNDING_RULES,
    user_template=_CONTEXT_BLOCK
    + """
TASK
Analyse the experimental evaluation.

Every number you report must be copied character-for-character from the context
and cited with the label of the source that contains it. Numbers are checked
automatically against that source; an uncited number is reported as unsupported.

Assess honestly whether the experiments support the paper's conclusions.

Answer with exactly this Markdown template:

## Datasets
- <dataset>

## Splits
<train/validation/test splits as stated>

## Baselines
- <baseline system>

## Metrics
- <metric>

## Reported Results
One row per reported number. "Best" is yes only if the paper marks it best.
"Quote" is the verbatim text containing the value; leave it empty for a table cell.

| System | Dataset | Split | Metric | Value | Best | Source | Quote |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <system> | <dataset> | <split> | <metric> | <value exactly as written> | no | [S1] | <verbatim text> |

## Ablations
- <ablation and its stated effect>

## Hyperparameters
| Name | Value |
| --- | --- |
| <hyperparameter> | <value exactly as written> |

## Error Analysis
<failure cases the authors discuss>

## Statistical Significance
<seeds, variance, confidence intervals, significance tests>

## Conclusions
- Supported: supported | partially_supported | unsupported | contradicted | unknown
<one paragraph justifying that assessment>

"""
    + _findings("experiments")
    + "\n",
)

MATH = PromptTemplate(
    name="math",
    version="v4",
    system=GROUNDING_RULES,
    user_template=_CONTEXT_BLOCK
    + """
TASK
Explain the mathematical content.

Copy every equation exactly as it appears in the context into its code block.
Set "Representation" to:
  - latex          if the context contains LaTeX you are copying,
  - extracted_text if you are copying extracted plain text,
  - unavailable    if the equation did not survive extraction legibly.
Never reconstruct, complete or tidy an equation you cannot read: mark it
unavailable and say so. If the paper does not state a derivation, write
"Not stated." for it rather than deriving one yourself, and put your own reading
in "Agent interpretation" with "Interpretation differs" set to yes.

Answer with exactly this Markdown template:

## Equations

### Equation <number or name as the paper labels it>
- Source: [S1]
- Representation: latex | extracted_text | unavailable
- Purpose: <what the equation is for>
- Derivation stated by authors: <or Not stated.>
- Agent interpretation: <your reading>
- Interpretation differs: yes | no
- Intuition: <plain-language intuition>

```text
<the equation exactly as in the source>
```

| Symbol | Meaning | Shape |
| --- | --- | --- |
| <symbol> | <meaning> | <shape or dimension, or empty> |

> [S1] <verbatim quote that defines or uses the equation>

### Equation <next>
...

## Loss Functions
- <loss function>

## Optimization Objective
<the overall objective>

## Notation
| Symbol | Meaning |
| --- | --- |
| <symbol> | <meaning> |

"""
    + _findings("math")
    + "\n",
)

COVERAGE = PromptTemplate(
    name="coverage",
    version="v2",
    system=GROUNDING_RULES,
    user_template=_CONTEXT_BLOCK
    + """
TASK
The excerpts above are the parts of the paper that no other analyst read. A
complete analysis must not lose anything they state. Go through them in order
and record every distinct piece of information a reader of the full paper
would rely on: design choices and their stated reasons, training and inference
settings, hardware and cost, comparisons and complexity arguments, results and
ablation outcomes, limitations, caveats and future work.

- One finding per statement; do not merge unrelated statements.
- Record every numeric setting or value in the Settings and Values table,
  copied character-for-character, with the quote that contains it.
- Skip nothing because it looks minor, but do not repeat the same statement.
- Figure word labels and bibliography fragments carry no claims; ignore them.

Answer with exactly this Markdown template:

## Settings and Values
| Name | Value | Source | Quote |
| --- | --- | --- | --- |
| <what the value is> | <value exactly as written> | [S1] | <verbatim text containing the value> |

"""
    + _findings(
        "contribution | method | training | inference | experiments | results | analysis "
        "| limitations"
    )
    + "\n",
)

VERIFICATION = PromptTemplate(
    name="verification",
    version="v4",
    system=GROUNDING_RULES
    + """

You are acting as the VERIFIER. You did not write the claim you are checking and
you have no stake in it being true. Your job is to be hard to convince.

A claim is NOT supported merely because the evidence is on the same topic or
shares vocabulary with it. Ask specifically:
  - Does the evidence assert this claim, or only something adjacent to it?
  - Do the numbers in the claim appear in the evidence with the same values,
    attached to the same system, dataset, metric and split?
  - Does the evidence apply the claim to the same scope (all datasets vs one)?
  - Does anything in the evidence contradict the claim outright?
  - Does the claim present as this paper's own something the evidence reports
    from a cited work (a source tagged "REPORTS CITED WORK [N]")? That is at
    most partially_supported: the words are in the paper, the finding is not
    the paper's.

Prefer "partially_supported" over "supported" whenever part of the claim is
unevidenced. Prefer "unsupported" over "partially_supported" when the evidence
only shares a topic.""",
    user_template="""\
PAPER METADATA
{metadata}

CLAIM UNDER VERIFICATION
{claim}

CLAIM'S ORIGINAL AGENT: {source_agent}

EVIDENCE RETRIEVED INDEPENDENTLY FROM THE PAPER
Each block is one source, labelled S1, S2, … . Cite a source only by its label.

{context}

DETERMINISTIC NUMERIC CHECK (computed without a model, already reliable)
{numeric_report}

TASK
Decide the claim's status:
  - supported            : the evidence directly asserts the claim.
  - partially_supported  : the evidence supports only part of it.
  - unsupported          : no adequate evidence was found.
  - contradicted         : the evidence conflicts with the claim.
  - inferred             : reasonable given the evidence, not stated.
  - unknown              : verification could not be completed.

Answer with exactly this Markdown template:

## Verdict
- Status: <one status from the list>
- Confidence: <0.0-1.0>

## Explanation
<why, naming the specific mismatch if there is one>

## Supporting Evidence
> [S1] <verbatim quote that supports the claim>

## Contradicting Evidence
> [S2] <verbatim quote that contradicts the claim, or write "None.">
""",
)

PLANNER = PromptTemplate(
    name="planner",
    version="v2",
    system="""\
You plan the analysis of one research paper. You are given only structural facts
about the paper (page count, detected sections, counts of tables, figures and
equations) — not its full text.

Recommend which specialist agents to run and which sections to prioritise. Run
an agent when the paper has content for it; skip it when it does not. Justify each
recommendation in one short sentence. Agent selection is finally decided from the
paper's structure; your recommendations are recorded alongside it. Answer in Markdown only, never JSON, using exactly
the headings of the template.""",
    user_template="""\
PAPER STRUCTURE
{structure}

Answer with exactly this Markdown template:

## Agents
| Agent | Run | Reason |
| --- | --- | --- |
| summary | yes | <why> |
| methodology | yes / no | <why> |
| experiments | yes / no | <why> |
| math | yes / no | <why> |
| table_analysis | yes / no | <why> |

## Priority Sections
- <section name exactly as listed in the structure>
""",
)

PROMPTS: dict[str, PromptTemplate] = {
    p.name: p
    for p in (SUMMARY, METHODOLOGY, EXPERIMENTS, MATH, COVERAGE, VERIFICATION, PLANNER)
}


def get_prompt(name: str) -> PromptTemplate:
    try:
        return PROMPTS[name]
    except KeyError:
        raise KeyError(f"unknown prompt '{name}'; known: {sorted(PROMPTS)}") from None
