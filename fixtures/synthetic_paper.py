"""Generate a synthetic research paper PDF for tests and demos.

The fixture is written rather than committed as a binary so the expected page,
section, table, figure and equation locations stay visible and reviewable in
source. ``EXPECTED`` below is the ground truth the tests and the evaluation
harness assert against.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

TITLE = "Sparse Routing Transformers for Long-Document Question Answering"
AUTHORS = "A. Researcher, B. Scientist, C. Engineer"

#: Ground truth for tests and for the evaluation harness.
EXPECTED = {
    "title": TITLE,
    "page_count": 6,
    "sections": [
        "Abstract",
        "1 Introduction",
        "2 Related Work",
        "3 Method",
        "3.1 Routing Objective",
        "4 Experiments",
        "4.1 Setup",
        "4.2 Main Results",
        "4.3 Ablations",
        "5 Limitations",
        "6 Conclusion",
        "References",
    ],
    "table_pages": [4],
    "figure_pages": [3],
    "equation_pages": [3],
    "expected_claims": [
        "sparse routing reduces attention cost",
        "SRT-Large reaches 81.4 F1 on NarrativeQA",
        "the router is trained with a load-balancing loss",
        "evaluation uses NarrativeQA and QuALITY",
        "gains shrink on documents under 2k tokens",
    ],
    "key_numbers": ["81.4", "76.2", "3.1", "0.01", "2048", "12"],
}


PAGES: list[list[tuple[str, str]]] = [
    # (style, text) — style drives font size/weight so heading detection is exercised.
    [
        ("title", TITLE),
        ("authors", AUTHORS),
        ("authors", "Institute for Applied Sequence Modelling"),
        ("heading", "Abstract"),
        (
            "body",
            "Transformer models applied to long documents spend most of their compute on "
            "attention over irrelevant context. We introduce Sparse Routing Transformers "
            "(SRT), which learn a discrete router that selects a small subset of document "
            "blocks for full attention while the remainder receive a cheap pooled "
            "representation. On NarrativeQA and QuALITY, SRT-Large reaches 81.4 F1 while "
            "reducing attention FLOPs by 3.1x relative to a dense baseline of the same "
            "parameter count. The router is trained jointly with the reader using a "
            "load-balancing loss and a straight-through estimator. We further show that "
            "the gains shrink on documents under 2k tokens, where dense attention is "
            "already inexpensive.",
        ),
        ("heading", "1 Introduction"),
        (
            "body",
            "Long-document question answering requires reasoning over inputs far longer "
            "than the context windows most encoders were trained on. Prior work addresses "
            "this with fixed sparsity patterns, retrieval over passages, or recurrence. "
            "Fixed patterns are cheap but content-agnostic: they cannot spend more compute "
            "on the two or three blocks that actually contain the answer.",
        ),
        (
            "body",
            "We argue that block selection should be learned end to end with the reader. "
            "Our contribution is threefold. First, we propose a routing objective that is "
            "differentiable through a straight-through estimator. Second, we show the "
            "router can be regularised so that block usage stays balanced across a batch. "
            "Third, we report a 3.1x reduction in attention FLOPs at equal accuracy on two "
            "long-document QA benchmarks.",
        ),
    ],
    [
        ("heading", "2 Related Work"),
        (
            "body",
            "Sparse attention. Longformer and BigBird impose fixed local and global "
            "attention patterns. These reduce the quadratic cost to linear but the pattern "
            "is chosen before any input is seen. Reformer uses locality-sensitive hashing "
            "to group similar queries, which is content-dependent but not trained.",
        ),
        (
            "body",
            "Retrieval-augmented readers. A separate line of work retrieves passages with "
            "an external index and reads only the retrieved set. This is effective but "
            "introduces a hard pipeline boundary: retrieval errors cannot be recovered by "
            "the reader, and the retriever is typically trained with a different objective.",
        ),
        (
            "body",
            "Mixture-of-experts routing. Our router is closest in spirit to expert routing "
            "in sparse mixture-of-experts layers, where a gate assigns tokens to experts "
            "and auxiliary losses prevent expert collapse. We adapt the load-balancing "
            "formulation of that literature to block selection rather than expert "
            "selection.",
        ),
    ],
    [
        ("heading", "3 Method"),
        (
            "body",
            "SRT takes a document D segmented into N blocks of 128 tokens and a question q. "
            "A lightweight scorer produces a relevance logit for every block conditioned on "
            "the question. The top-k blocks by logit are passed to the full transformer "
            "reader; the remaining blocks are mean-pooled into a single summary vector that "
            "is appended to the reader input. The output is a span over the selected blocks.",
        ),
        ("caption", "Figure 1: Architecture of the Sparse Routing Transformer. The router scores blocks against the question; only the top-k blocks receive full attention."),
        ("heading2", "3.1 Routing Objective"),
        (
            "body",
            "Let s_i denote the router logit for block i and let m_i be the binary selection "
            "mask. The training objective combines the reader loss with a load-balancing "
            "term:",
        ),
        ("equation", "L = L_read + lambda * sum_i ( f_i * P_i )        (1)"),
        (
            "body",
            "where f_i is the fraction of examples in the batch routing to block position i, "
            "P_i is the mean router probability for that position, and lambda = 0.01 in all "
            "experiments. The mask m is obtained with a straight-through estimator so that "
            "gradients reach the scorer despite the discrete top-k.",
        ),
        ("equation", "m_i = 1[ s_i in topk(s) ] ,   d m_i / d s_i := sigma'(s_i)        (2)"),
        (
            "body",
            "The router score itself is a scaled dot product between the pooled question "
            "representation and the block representation:",
        ),
        ("equation", "s_i = ( W_q q )^T ( W_b b_i ) / sqrt(d)        (3)"),
        (
            "body",
            "We set k = 12 selected blocks for all reported results unless stated otherwise. "
            "Training uses AdamW with a learning rate of 3e-5, a batch size of 32, and a "
            "maximum input length of 2048 tokens per block set.",
        ),
    ],
    [
        ("heading", "4 Experiments"),
        ("heading2", "4.1 Setup"),
        (
            "body",
            "We evaluate on NarrativeQA and QuALITY. NarrativeQA is split into the standard "
            "train, validation and test partitions; we report test F1. QuALITY is evaluated "
            "with accuracy on the hard subset. Baselines are a dense transformer of equal "
            "parameter count, Longformer with a window of 512, and a retrieval-augmented "
            "reader using BM25 over 128-token passages.",
        ),
        ("heading2", "4.2 Main Results"),
        (
            "body",
            "Table 1 reports the main comparison. SRT-Large improves over the dense baseline "
            "by 5.2 F1 on NarrativeQA while using 3.1x fewer attention FLOPs.",
        ),
        ("caption", "Table 1: Main results on long-document QA. FLOPs are attention-only, relative to Dense-Large."),
        ("table", "Model            NarrativeQA F1   QuALITY Acc   Attn FLOPs\n"
                  "Dense-Large              76.2          61.4         1.00\n"
                  "Longformer-Large         77.9          63.0         0.44\n"
                  "BM25 + Reader            74.1          59.8         0.21\n"
                  "SRT-Base                 79.3          65.1         0.38\n"
                  "SRT-Large                81.4          68.7         0.32"),
        ("heading2", "4.3 Ablations"),
        (
            "body",
            "Removing the load-balancing loss reduces NarrativeQA F1 from 81.4 to 78.8 and "
            "causes 70 percent of routed mass to concentrate on the first four block "
            "positions. Replacing the straight-through estimator with a Gumbel relaxation "
            "costs 1.1 F1. Reducing k from 12 to 4 costs 2.9 F1 but halves attention FLOPs "
            "again.",
        ),
    ],
    [
        ("heading", "5 Limitations"),
        (
            "body",
            "Our gains are concentrated on long inputs. On documents under 2k tokens the "
            "dense baseline is already inexpensive and SRT is within 0.3 F1 of it while "
            "adding router overhead. We did not evaluate on languages other than English, "
            "and all experiments use a single random seed per configuration, so the "
            "reported differences are not accompanied by confidence intervals.",
        ),
        (
            "body",
            "The router adds a hyperparameter k that must be tuned per dataset. We observed "
            "that a k that is too small produces unstable training in the first 2000 steps, "
            "which we mitigated with a linear warm-up of the sparsity level.",
        ),
        ("heading", "6 Conclusion"),
        (
            "body",
            "We presented Sparse Routing Transformers, which learn which blocks of a long "
            "document deserve full attention. The approach reaches 81.4 F1 on NarrativeQA "
            "at 3.1x lower attention cost than a dense model of equal size. Future work "
            "includes extending the router to multi-hop questions and to multilingual "
            "corpora.",
        ),
    ],
    [
        ("heading", "References"),
        ("reference", "[1] Beltagy, I., Peters, M., and Cohan, A. Longformer: The long-document transformer. arXiv, 2020."),
        ("reference", "[2] Zaheer, M., Guruganesh, G., and Dubey, A. Big Bird: Transformers for longer sequences. NeurIPS, 2020."),
        ("reference", "[3] Kitaev, N., Kaiser, L., and Levskaya, A. Reformer: The efficient transformer. ICLR, 2020."),
        ("reference", "[4] Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models. JMLR, 2022."),
        ("reference", "[5] Kocisky, T., Schwarz, J., and Blunsom, P. The NarrativeQA reading comprehension challenge. TACL, 2018."),
    ],
]

_STYLES = {
    "title": {"size": 17, "font": "helvetica-bold", "space": 14},
    "authors": {"size": 10, "font": "helvetica", "space": 6},
    "heading": {"size": 13, "font": "helvetica-bold", "space": 10},
    "heading2": {"size": 11.5, "font": "helvetica-bold", "space": 9},
    "body": {"size": 9.5, "font": "helvetica", "space": 9},
    "caption": {"size": 8.8, "font": "helvetica-oblique", "space": 9},
    "equation": {"size": 10, "font": "courier", "space": 11},
    "table": {"size": 8.6, "font": "courier", "space": 9},
    "reference": {"size": 8.6, "font": "helvetica", "space": 7},
}


def build_pdf(output: str | Path) -> Path:
    """Write the synthetic paper to ``output`` and return the path."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open()
    for blocks in PAGES:
        page = doc.new_page(width=595, height=842)  # A4
        cursor = 64.0
        for style_name, text in blocks:
            style = _STYLES[style_name]
            rect = pymupdf.Rect(60, cursor, 535, 800)
            overflow = page.insert_textbox(
                rect,
                text,
                fontsize=style["size"],
                fontname=style["font"],
                align=pymupdf.TEXT_ALIGN_LEFT,
                lineheight=1.25,
            )
            # insert_textbox returns remaining vertical space when it fits.
            used = (rect.height - overflow) if overflow >= 0 else rect.height
            cursor += used + style["space"]
    doc.save(path)
    doc.close()
    return path


if __name__ == "__main__":  # pragma: no cover - developer convenience
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "fixtures/generated/synthetic_paper.pdf"
    print(build_pdf(target))
