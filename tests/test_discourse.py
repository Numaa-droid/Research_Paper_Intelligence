"""Deterministic discourse analysis: attribution, paper kind, review protocol.

These mechanisms decide, without a model, whose statement a quote is and what
kind of paper is being read. Every test here uses generic text, not any one
paper's wording, because the rules must hold for papers not yet seen.
"""

from __future__ import annotations

from paperpilot.discourse import (
    attribution_map,
    heading_numbering_issues,
    profile_paper,
    review_protocol,
    text_attribution,
)
from paperpilot.discourse.attribution import MAX_CARRY, quote_attribution, split_sentences
from paperpilot.domain.enums import CitationQuality, ElementType, FindingStatus
from paperpilot.domain.models import Evidence, Finding, PaperChunk, PaperMetadata
from paperpilot.retrieval.index import PaperIndex
from paperpilot.validation.dedupe import merge_duplicate_findings
from paperpilot.validation.reconcile import reconcile_findings


def _index(chunks: list[tuple[str, str]], *, title: str = "A Study", refs: int = 40) -> PaperIndex:
    """Chunks as (section, text), one page each, plus a numbered reference list."""
    built = [
        PaperChunk(
            chunk_id=f"doc-page-{i}-chunk-01",
            document_id="doc",
            page=i,
            section=section,
            text=text,
        )
        for i, (section, text) in enumerate(chunks, start=1)
    ]
    built.append(
        PaperChunk(
            chunk_id="doc-refs",
            document_id="doc",
            page=len(chunks) + 1,
            section="References",
            element_type=ElementType.REFERENCE,
            text=" ".join(f"[{n}] A. Author, Title {n}." for n in range(1, refs + 1)),
        )
    )
    metadata = PaperMetadata(
        document_id="doc",
        title=title,
        page_count=len(built),
        chunk_count=len(built),
        sections=list(dict.fromkeys(section for section, _ in chunks)),
    )
    return PaperIndex.build(document_id="doc", metadata=metadata, chunks=built)


def _voices(index: PaperIndex) -> list[tuple[str, tuple[str, ...]]]:
    return [(s.voice, s.sources) for s in attribution_map(index).sentences]


# --- sentence splitting ------------------------------------------------------


def test_sentences_keep_et_al_and_figure_references_intact():
    text = "Smith et al. [4] show a gain in Fig. 3 of their paper. We disagree."
    assert split_sentences(text) == [
        "Smith et al. [4] show a gain in Fig. 3 of their paper.",
        "We disagree.",
    ]


# --- attribution -------------------------------------------------------------


def test_a_report_opens_carries_and_ends_at_the_papers_own_voice():
    index = _index(
        [
            (
                "3 Related Work",
                "In [12] the authors trained a classifier on forum posts. "
                "They reported an accuracy of 0.91. "
                "According to them, the gain came from pretraining. "
                "We instead use a rule-based baseline.",
            )
        ]
    )
    assert _voices(index) == [
        ("reported", ("[12]",)),
        ("reported", ("[12]",)),
        ("reported", ("[12]",)),
        ("own", ()),
    ]


def test_a_new_citation_takes_the_floor_from_the_previous_one():
    index = _index(
        [
            (
                "2 Background",
                "According to [3], retrieval helps. In [7] the authors disagree. "
                "Smith et al. [9] show the opposite.",
            )
        ]
    )
    assert [sources for _, sources in _voices(index)] == [("[3]",), ("[7]",), ("[9]",)]


def test_a_report_does_not_carry_into_the_next_section():
    index = _index(
        [
            ("2 Background", "In [5] the authors measured latency."),
            ("3 Method", "The system is built from two modules."),
        ]
    )
    assert _voices(index)[1] == ("neutral", ())


def test_a_report_fades_after_several_sentences_with_nothing_tying_it_back():
    filler = " ".join(f"Sentence number {i} is about something else." for i in range(MAX_CARRY + 2))
    index = _index([("2 Background", f"In [5] the authors measured latency. {filler}")])
    voices = _voices(index)
    assert voices[MAX_CARRY][0] == "reported"
    assert voices[-1][0] == "neutral"


def test_an_inline_citation_without_reporting_wording_stays_the_papers():
    index = _index([("1 Introduction", "Attention models [3] are widely used in translation.")])
    assert _voices(index) == [("neutral", ())]


def test_an_interval_or_an_unknown_reference_number_is_not_a_citation():
    index = _index(
        [("4 Method", "In [0, 1] the scores are normalised. In [99] the authors disagree.")],
        refs=20,
    )
    assert all(voice != "reported" for voice, _ in _voices(index))


def test_author_year_citations_are_recognised():
    index = _index([("2 Background", "In (Lee et al., 2021) the authors propose a parser.")])
    voice, sources = _voices(index)[0]
    assert voice == "reported"
    assert sources == ("Lee et al., 2021",)


def test_a_quote_is_attributed_by_the_sentences_it_was_copied_from():
    index = _index(
        [
            (
                "3 Domains",
                "In [20] the authors used a detector on essays. "
                "The detection rate was between 10% and 40%. "
                "We group these studies by domain.",
            )
        ]
    )
    amap = attribution_map(index)
    chunk = index.chunks[0].chunk_id
    assert quote_attribution("The detection rate was between 10% and 40%.", amap, chunk_id=chunk, page=1) == (
        "cited",
        ("[20]",),
    )
    assert quote_attribution("We group these studies by domain.", amap, chunk_id=chunk, page=1)[0] == "own"


def test_a_phrase_is_attributed_by_its_numbers():
    index = _index(
        [
            ("2 Corpus", "Our search returned 120 records."),
            ("3 Studies", "In [8] the authors compared 327 introductions."),
        ]
    )
    amap = attribution_map(index)
    assert text_attribution("327 published introductions", amap) == ("cited", ("[8]",))
    assert text_attribution("120 records", amap)[0] == "own"


# --- paper kind --------------------------------------------------------------


def _review_body(n: int = 8) -> list[tuple[str, str]]:
    body = [("1 Introduction", "In this article we review applications of the tool.")]
    body += [
        (f"3.{i} Domain {i}", f"In [{i}] the authors applied the tool to task {i}. They found gains.")
        for i in range(1, n + 1)
    ]
    return body


def test_a_titled_review_whose_body_reports_cited_works_is_a_review():
    profile = profile_paper(_index(_review_body(), title="The Tool: A Survey of Applications"))
    assert profile.kind == "review"
    assert any("survey" in s.lower() for s in profile.signals)


def test_an_untitled_paper_that_mostly_reports_cited_works_is_a_review():
    assert profile_paper(_index(_review_body(), title="The Tool in Practice")).kind == "review"


def test_a_paper_that_trains_and_evaluates_in_its_own_voice_is_empirical():
    index = _index(
        [
            ("1 Introduction", "Prior parsers [3] are slow."),
            ("4 Experiments", "We train the model on 10k sentences. We evaluate it on two test sets."),
        ],
        title="A Fast Parser",
    )
    assert profile_paper(index).kind == "empirical"


# --- review protocol ---------------------------------------------------------


def test_the_protocol_checklist_uses_only_the_papers_own_sentences():
    index = _index(
        [
            ("2 Method", "We searched Scopus for the keyword parser between 2019 and 2023."),
            ("3 Studies", "In [4] the authors applied inclusion criteria to 300 records."),
        ]
    )
    found = {item.name: item.found for item in review_protocol(index)}
    assert found["Sources searched named"]
    assert found["Search terms stated"]
    assert found["Time window stated"]
    # Stated only about cited work [4], so not this review's own protocol.
    assert not found["Inclusion/exclusion criteria stated"]
    assert not found["Number of records reported"]


# --- structure ---------------------------------------------------------------


def test_repeated_and_skipped_heading_numbers_are_reported():
    issues = heading_numbering_issues(["1 Intro", "2.1 A", "2.3 B", "2.1 C", "3 End"])
    assert any("2.1 is used twice" in i for i in issues)
    assert any("2.2 is skipped" in i for i in issues)


def test_a_clean_numbering_has_no_issues():
    assert heading_numbering_issues(["1 Intro", "2 Method", "2.1 A", "2.2 B", "3 End"]) == []


# --- reconcile and dedupe ----------------------------------------------------


def _finding(claim: str, quote: str, index: PaperIndex, status=FindingStatus.SUPPORTED) -> Finding:
    chunk = index.chunks[0]
    return Finding(
        claim=claim,
        status=status,
        evidence=[
            Evidence(
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                quote=quote,
                confidence=0.9,
                citation_quality=CitationQuality.EXACT,
            )
        ],
    )


def test_a_claim_crediting_the_paper_with_a_cited_finding_is_capped():
    index = _index([("3 Studies", "In [6] the authors found that tutoring improves scores.")])
    quote = "In [6] the authors found that tutoring improves scores."
    owned = _finding("The paper shows that tutoring improves scores.", quote, index)
    neutral = _finding("Tutoring improves scores, according to a cited study.", quote, index)

    reconcile_findings([owned, neutral], index)

    assert owned.attribution == "cited" and owned.cited_sources == ["[6]"]
    assert owned.status is FindingStatus.PARTIALLY_SUPPORTED
    assert "cited work [6]" in (owned.verification_note or "")
    # Correctly worded, it keeps its status and only carries the attribution.
    assert neutral.status is FindingStatus.SUPPORTED
    assert neutral.attribution == "cited"


def test_a_claim_using_wording_the_paper_never_uses_is_capped_once():
    index = _index([("4 Limits", "We note that the training data is not updated.")])
    finding = _finding(
        "The training data is not updated, limiting use in fast-changing fields.",
        "We note that the training data is not updated.",
        index,
    )
    reconcile_findings([finding], index)
    reconcile_findings([finding], index)  # a retry runs the pass again

    assert finding.status is FindingStatus.PARTIALLY_SUPPORTED
    assert finding.verification_note.count("fast-changing") == 1


def test_duplicate_findings_are_merged_into_the_first():
    index = _index([("3 Studies", "Tutoring improves scores. Tutoring is cheap.")])
    first = _finding("Tutoring improves scores.", "Tutoring improves scores.", index)
    again = _finding("Tutoring improves the scores.", "Tutoring is cheap.", index)
    other = _finding("Tutoring is cheap.", "Tutoring is cheap.", index)

    merged = merge_duplicate_findings([first, again, other])

    assert merged == [first, other]
    assert len(first.evidence) == 2


# --- the report --------------------------------------------------------------


def test_a_review_report_frames_its_sections_as_a_review_and_tags_cited_findings():
    from paperpilot.domain.models import AgentResult, AnalysisPlan
    from paperpilot.reporting.synthesizer import ReportSynthesizer

    index = _index(
        [
            ("2 Corpus", "We searched Scopus for the keyword tool and found 40 papers."),
            *_review_body(),
        ],
        title="The Tool: A Review",
    )
    profile = profile_paper(index)
    chunk = next(c for c in index.chunks if "In [1]" in c.text)
    finding = Finding(
        claim="The tool improves task 1.",
        category="contribution",
        status=FindingStatus.SUPPORTED,
        agent="summary",
        evidence=[
            Evidence(
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                quote="In [1] the authors applied the tool to task 1.",
                citation_quality=CitationQuality.EXACT,
            )
        ],
    )
    reconcile_findings([finding], index)
    report = ReportSynthesizer().synthesize(
        metadata=index.metadata,
        index=index,
        plan=AnalysisPlan(paper_kind=profile.kind, paper_kind_signals=profile.signals),
        agent_results={"summary": AgentResult(agent="summary", findings=[finding])},
        findings=[finding],
        verification_results=[],
    )

    assert "Methodology — how the review was conducted" in report
    assert "Reproducibility checklist — review protocol" in report
    assert "no inclusion or exclusion criteria are stated" in report
    assert "reported from cited work [1]" in report
    # A cited work's finding is never listed as this paper's contribution.
    contributions = report.split("## 3. Contributions")[1].split("## 4.")[0]
    assert "The tool improves task 1." not in contributions
