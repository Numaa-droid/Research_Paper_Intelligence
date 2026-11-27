"""Draw PaperMind's README graphics. Run from the repo root: python assets/make_svgs.py assets

Shared style and helpers come from the AEIDS project's generator (same author)."""
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

BG, INK, MUTED, SOFT = "#060a14", "#e7ecf6", "#8d99b3", "#a9b4cc"
CARD, LINE = "#0d1424", "#94a3b8"
VIOLET, VIOLET_T = "#8b5cf6", "#c4b5fd"
CYAN, CYAN_T = "#22d3ee", "#67e8f9"
TEAL, TEAL_T = "#2dd4bf", "#5eead4"
GREEN, GREEN_T = "#22c55e", "#86efac"
ROSE, ROSE_T = "#f43f5e", "#fda4af"
AMBER, AMBER_T = "#f59e0b", "#fcd34d"
TINT = {CYAN: CYAN_T, TEAL: TEAL_T, VIOLET: VIOLET_T, GREEN: GREEN_T, ROSE: ROSE_T, AMBER: AMBER_T}


def w(text, size, bold=False, mono=False):
    k = 0.61 if mono else (0.57 if bold else 0.53)
    return len(text) * size * k


def t(x, y, s, size=14, fill=INK, weight=None, anchor=None, cls="sans", extra=""):
    a = f' font-size="{size}" fill="{fill}"'
    if weight:
        a += f' font-weight="{weight}"'
    if anchor:
        a += f' text-anchor="{anchor}"'
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}"{a}{extra}>{escape(s)}</text>'


def chip(x, y, s, color, tcolor, size=13, h=30, pad=14, bold=True):
    cw = w(s, size, bold) + 2 * pad
    return (f'<rect x="{x:.1f}" y="{y}" width="{cw:.0f}" height="{h}" rx="{h/2}" fill="{color}" fill-opacity=".13" '
            f'stroke="{color}" stroke-opacity=".5"/>'
            + t(x + cw / 2, y + h / 2 + size * 0.36, s, size, tcolor, "600" if bold else None, "middle")), cw


def head(W, H, title, desc):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">
  <title id="t">{escape(title)}</title>
  <desc id="d">{escape(desc)}</desc>
  <defs>
    <linearGradient id="grad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#a78bfa"/><stop offset=".45" stop-color="#6366f1"/><stop offset="1" stop-color="#22d3ee"/></linearGradient>
    <linearGradient id="gradH" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#a78bfa"/><stop offset=".5" stop-color="#6366f1"/><stop offset="1" stop-color="#22d3ee"/></linearGradient>
    <linearGradient id="gradV" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#a78bfa"/><stop offset="1" stop-color="#6366f1"/></linearGradient>
    <linearGradient id="gradC" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#67e8f9"/><stop offset="1" stop-color="#0891b2"/></linearGradient>
    <radialGradient id="glowV" cx=".08" cy="-.05" r=".7"><stop offset="0" stop-color="#8b5cf6" stop-opacity=".35"/><stop offset="1" stop-color="#8b5cf6" stop-opacity="0"/></radialGradient>
    <radialGradient id="glowC" cx="1" cy=".1" r=".6"><stop offset="0" stop-color="#22d3ee" stop-opacity=".2"/><stop offset="1" stop-color="#22d3ee" stop-opacity="0"/></radialGradient>
    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#94a3b8" stroke-opacity=".07"/></pattern>
    <linearGradient id="fadeGrid" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".9"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
    <mask id="gridMask"><rect width="{W}" height="{H}" fill="url(#fadeGrid)"/></mask>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="14" stdDeviation="16" flood-color="#000" flood-opacity=".5"/></filter>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{CYAN}"/></marker>
    <marker id="arrowV" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{VIOLET_T}"/></marker>
    <marker id="arrowR" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{ROSE}"/></marker>
    <marker id="arrowG" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{GREEN}"/></marker>
    <style>
      .sans{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
      .mono{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}}
      .flow{{stroke-dasharray:6 7;animation:flow 1.1s linear infinite}}
      @keyframes flow{{to{{stroke-dashoffset:-26}}}}
      .pulse{{animation:pulse 2.4s ease-in-out infinite}}
      @keyframes pulse{{0%,100%{{opacity:.45}}50%{{opacity:1}}}}
      .grow{{transform-box:fill-box;transform-origin:50% 100%;animation:grow 1.1s cubic-bezier(.2,.8,.2,1) both}}
      @keyframes grow{{from{{transform:scaleY(0)}}to{{transform:scaleY(1)}}}}
      .pop{{transform-box:fill-box;transform-origin:50% 50%;animation:pop 2.8s ease-in-out infinite}}
      @keyframes pop{{0%,100%{{transform:scale(1)}}50%{{transform:scale(1.08)}}}}
      @media (prefers-reduced-motion:reduce){{.flow,.pulse,.grow,.pop{{animation:none}}}}
    </style>
  </defs>
  <rect width="{W}" height="{H}" rx="24" fill="{BG}"/>
  <rect width="{W}" height="{H}" rx="24" fill="url(#glowV)"/>
  <rect width="{W}" height="{H}" rx="24" fill="url(#glowC)"/>
  <rect width="{W}" height="{H}" rx="24" fill="url(#grid)" mask="url(#gridMask)"/>
  <rect x=".5" y=".5" width="{W-1}" height="{H-1}" rx="24" fill="none" stroke="{LINE}" stroke-opacity=".18"/>
'''


def card(x, y, cw, ch, stroke=LINE, sop=".22", fill=CARD, rx=16, shadow=True):
    f = ' filter="url(#shadow)"' if shadow else ""
    return f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-opacity="{sop}"{f}/>'


def title_block(title, sub, y=70):
    return t(48, y, title, 30, INK, "800", extra=' letter-spacing="-.6"') + t(48, y + 30, sub, 15.5, MUTED)


def icon(kind, cx, cy, c, sw=2.2):
    st = f'fill="none" stroke="{c}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"'
    if kind == "upload":
        return (f'<path d="M{cx-13} {cy-18}h17l9 9v27h-26z" {st}/><path d="M{cx+4} {cy-18}v9h9" {st}/>'
                f'<path d="M{cx-0.5} {cy+12}v-14M{cx-6} {cy+3}l5.5-5.5 5.5 5.5" {st}/>')
    if kind == "parse":
        return (f'<rect x="{cx-17}" y="{cy-18}" width="34" height="36" rx="3" {st}/>'
                f'<path d="M{cx-17} {cy-8}h34M{cx-17} {cy+2}h34M{cx-17} {cy+11}h34M{cx-4} {cy-8}v26" {st}/>')
    if kind == "index":
        return (f'<circle cx="{cx-3}" cy="{cy-3}" r="12" {st}/><path d="M{cx+6} {cy+6}l9 9" {st}/>'
                f'<path d="M{cx-9} {cy-3}h12M{cx-9} {cy+2}h8" {st}/>')
    if kind == "plan":
        return (f'<rect x="{cx-6}" y="{cy-19}" width="12" height="10" rx="2" {st}/>'
                f'<rect x="{cx-20}" y="{cy+9}" width="12" height="10" rx="2" {st}/>'
                f'<rect x="{cx-6}" y="{cy+9}" width="12" height="10" rx="2" {st}/>'
                f'<rect x="{cx+8}" y="{cy+9}" width="12" height="10" rx="2" {st}/>'
                f'<path d="M{cx} {cy-9}v18M{cx} {cy}h-14v9M{cx} {cy}h14v9" {st}/>')
    if kind == "calc":
        return (f'<rect x="{cx-14}" y="{cy-19}" width="28" height="38" rx="4" {st}/>'
                f'<rect x="{cx-8}" y="{cy-13}" width="16" height="7" rx="1.5" {st}/>'
                + "".join(f'<circle cx="{cx-7+dx*7}" cy="{cy+3+dy*8}" r="1.8" fill="{c}"/>'
                          for dx in range(3) for dy in range(2)))
    if kind == "verify":
        return (f'<path d="M{cx} {cy-19}l15 5v11c0 10-6.5 16.5-15 20c-8.5-3.5-15-10-15-20v-11z" {st}/>'
                f'<path d="M{cx-7} {cy}l5 5l9-10" {st}/>')
    if kind == "report":
        return (f'<path d="M{cx-14} {cy-19}h28v38h-28z" {st}/>'
                f'<path d="M{cx-8} {cy-9}l2.5 2.5 4-4M{cx+1} {cy-8}h7M{cx-8} {cy+1}l2.5 2.5 4-4M{cx+1} {cy+2}h7M{cx-8} {cy+12}h16" {st}/>')
    if kind == "hex":
        pts = " ".join(f"{cx+19*math.cos(k*math.pi/3+math.pi/6):.1f},{cy+19*math.sin(k*math.pi/3+math.pi/6):.1f}"
                       for k in range(6))
        return f'<polygon points="{pts}" {st}/><circle cx="{cx}" cy="{cy}" r="5" {st}/>'
    if kind == "user":
        return (f'<circle cx="{cx}" cy="{cy-7}" r="7.5" {st}/><path d="M{cx-14} {cy+17}c0-9 6-14 14-14s14 5 14 14" {st}/>')
    if kind == "db":
        return (f'<ellipse cx="{cx}" cy="{cy-10}" rx="13" ry="5" {st}/>'
                f'<path d="M{cx-13} {cy-10}v20c0 3 6 5 13 5s13-2 13-5v-20M{cx-13} {cy}c0 3 6 5 13 5s13-2 13-5" {st}/>')
    raise ValueError(kind)


def icon_tile(kind, x, y, c, size=56):
    return (f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="14" fill="{c}" fill-opacity=".12" '
            f'stroke="{c}" stroke-opacity=".45"/>' + icon(kind, x + size / 2, y + size / 2, c))


def icon2(kind, cx, cy, c, sw=2.2):
    """Extra icons for PaperMind; everything else comes from icon()."""
    st = f'fill="none" stroke="{c}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"'
    if kind == "chart":
        return (f'<path d="M{cx-17} {cy-18}v36h34" {st}/>'
                f'<path d="M{cx-10} {cy+10}v-8M{cx-2} {cy+10}v-16M{cx+6} {cy+10}v-11M{cx+14} {cy+10}v-22" {st}/>')
    if kind == "sigma":
        return f'<path d="M{cx+13} {cy-16}h-26l14 16l-14 16h26" {st}/>'
    if kind == "paper":
        return (f'<path d="M{cx-14} {cy-19}h20l9 9v29h-29z" {st}/><path d="M{cx+6} {cy-19}v9h9" {st}/>'
                f'<path d="M{cx-8} {cy-4}h16M{cx-8} {cy+3}h16M{cx-8} {cy+10}h10" {st}/>')
    return icon(kind, cx, cy, c, sw)


def tile(kind, x, y, c, size=56):
    return (f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="14" fill="{c}" fill-opacity=".12" '
            f'stroke="{c}" stroke-opacity=".45"/>' + icon2(kind, x + size / 2, y + size / 2, c))


TABLE = [("Dense-Large", "76.2", "61.4"), ("Longformer-Large", "77.9", "63.0"), ("BM25 + Reader", "74.1", "59.8"),
         ("SRT-Base", "79.3", "65.1"), ("SRT-Large", "81.4", "68.7")]


def paper_page(x, y, pw, ph):
    """A white research-paper page: title, section 4.2 and Table 1 with the SRT-Large row highlighted."""
    s = [f'<g filter="url(#shadow)"><rect x="{x}" y="{y}" width="{pw}" height="{ph}" rx="12" fill="#f8fafc"/></g>']
    ix, iw = x + 20, pw - 40
    s.append(f'<rect x="{ix+iw*0.08:.0f}" y="{y+22}" width="{iw*0.84:.0f}" height="9" rx="4.5" fill="#cbd5e1"/>')
    s.append(f'<rect x="{ix+iw*0.22:.0f}" y="{y+37}" width="{iw*0.56:.0f}" height="7" rx="3.5" fill="#e2e8f0"/>')
    for k, frac in enumerate((0.97, 0.92, 0.95, 0.7)):
        s.append(f'<rect x="{ix}" y="{y+60+k*12}" width="{iw*frac:.0f}" height="6" rx="3" fill="#e2e8f0"/>')
    s.append(t(ix, y + 128, "4.2  Main Results", 11.5, "#334155", "800"))
    s.append(t(ix, y + 146, "Table 1: Main results on long-document QA.", 8.6, "#64748b", None, extra=' font-style="italic"'))
    ty = y + 156
    c1, c2 = ix + iw * 0.70, ix + iw
    s.append(f'<rect x="{ix}" y="{ty}" width="{iw}" height="22" fill="#eef2ff"/>')
    s.append(t(ix + 6, ty + 15, "Model", 9.2, "#475569", "700"))
    s.append(t(c1, ty + 15, "NQA F1", 9.2, "#475569", "700", "end"))
    s.append(t(c2 - 6, ty + 15, "QuALITY", 9.2, "#475569", "700", "end"))
    hl = None
    for i, (m, f1, q) in enumerate(TABLE):
        ry = ty + 22 + i * 22
        last = i == len(TABLE) - 1
        if last:
            s.append(f'<rect class="pulse" x="{ix-4}" y="{ry+1}" width="{iw+8}" height="21" rx="4" fill="{CYAN}" '
                     f'fill-opacity=".14" stroke="{CYAN}" stroke-width="1.6" stroke-dasharray="4 3"/>')
            hl = (ix + iw + 4, ry + 12)
        bold = "700" if last else None
        s.append(t(ix + 6, ry + 15, m, 9.2, "#334155", bold))
        s.append(t(c1, ry + 15, f1, 9.2, "#0f172a" if last else "#334155", bold, "end"))
        s.append(t(c2 - 6, ry + 15, q, 9.2, "#334155", bold, "end"))
        s.append(f'<line x1="{ix}" y1="{ry+22}" x2="{ix+iw}" y2="{ry+22}" stroke="#eef2f7"/>')
    py = ty + 22 + len(TABLE) * 22 + 18
    for k, frac in enumerate((0.95, 0.88, 0.6)):
        if py + k * 12 + 6 < y + ph - 26:
            s.append(f'<rect x="{ix}" y="{py+k*12}" width="{iw*frac:.0f}" height="6" rx="3" fill="#e2e8f0"/>')
    s.append(t(ix + iw, y + ph - 12, "p. 4", 9.5, "#94a3b8", None, "end"))
    return "\n".join(s), hl


# =========================================================================== banner
def banner():
    W, H = 1280, 460
    s = [head(W, H, "PaperMind: citation-grounded multi-agent analysis of research papers",
              "A research-paper page whose Table 1 row is cited by a supported claim: the quote was found on the "
              "page and the number matches its evidence.")]
    s.append('<rect x="64" y="64" width="52" height="52" rx="14" fill="url(#grad)"/>')
    s.append(icon2("paper", 90, 91, "#fff", 2.4))
    s.append(t(132, 86, "PaperMind", 17, INK, "700", extra=' letter-spacing=".5"'))
    s.append(t(132, 107, "Citation-grounded analysis of research papers", 14, MUTED))
    s.append(t(64, 190, "Every claim, checked", 52, INK, "800", extra=' letter-spacing="-1.5"'))
    s.append(t(64, 250, "against the page.", 52, "url(#gradH)", "800", extra=' letter-spacing="-1.5"'))
    s.append(t(64, 298, "Specialist agents read one research paper; a verifier", 19, SOFT))
    s.append(t(64, 324, "re-checks every high-impact claim against the PDF.", 19, SOFT))
    x = 64
    for label, c in (("Page-level citations", VIOLET), ("Deterministic checks", CYAN), ("Bounded retries", TEAL)):
        g, cw = chip(x, 358, label, c, TINT[c], 13.5, 34, 16)
        s.append(g)
        x += cw + 12
    page, (hx, hy) = paper_page(672, 66, 262, 330)
    s.append(page)
    cx, cy, cw, ch = 970, 92, 250, 286
    s.append(card(cx, cy, cw, ch))
    s.append(t(cx + 20, cy + 32, "CLAIM", 11, MUTED, "700", extra=' letter-spacing="1.4"'))
    s.append(f'<g class="pop"><rect x="{cx+cw-118}" y="{cy+15}" width="102" height="25" rx="12.5" fill="{GREEN}" '
             f'fill-opacity=".15" stroke="{GREEN}" stroke-opacity=".55"/>'
             + t(cx + cw - 67, cy + 32, "✓ SUPPORTED", 11, GREEN_T, "700", "middle", extra=' letter-spacing=".6"') + "</g>")
    s.append(t(cx + 20, cy + 72, "SRT-Large reaches", 21, INK, "800"))
    s.append(t(cx + 20, cy + 99, "81.4 F1 on NarrativeQA", 19, "url(#gradH)", "800"))
    s.append(f'<line x1="{cx+20}" y1="{cy+118}" x2="{cx+cw-20}" y2="{cy+118}" stroke="{LINE}" stroke-opacity=".2"/>')
    rows = (("QUOTE", "found on the page · exact", VIOLET_T), ("NUMBER", "81.4 = 81.4 in the evidence", CYAN_T),
            ("CITATION", "[p. 4, §4.2, Table 1]", TEAL_T))
    for i, (lab, val, c) in enumerate(rows):
        yy = cy + 144 + i * 46
        s.append(t(cx + 20, yy, lab, 10.5, MUTED, "700", extra=' letter-spacing="1.2"'))
        s.append(t(cx + 20, yy + 20, val, 12.5, c, None, cls="mono"))
    sy = cy + 144 + 2 * 46 + 15
    s.append(f'<path class="flow" d="M{hx+2} {hy} C {hx+40} {hy}, {cx-36} {sy}, {cx-4} {sy}" fill="none" '
             f'stroke="{CYAN}" stroke-width="2" marker-end="url(#arrow)"/>')
    s.append(t(1216, 430, "example from the bundled synthetic test paper", 11, MUTED, None, "end", extra=' font-style="italic"'))
    s.append("</svg>")
    return "\n".join(s)


# =========================================================================== pipeline
def pipeline():
    W, H = 1280, 560
    s = [head(W, H, "How PaperMind analyses a paper",
              "Six steps: ingest and index the PDF, then plan, analyse with specialist agents, verify and report. "
              "Agents whose claims fail to ground are retried a limited number of times.")]
    s.append(title_block("How PaperMind analyses a paper",
                         "The PDF is indexed once, page by page. Then a planned team of agents analyses it, and nothing is cited unchecked."))
    steps = [
        ("upload", "Ingest", ["PDF parsed page by", "page; OCR fallback", "for scanned pages"], CYAN),
        ("index", "Index", ["Page-aware chunks;", "BM25 + dense search", "fused with RRF"], CYAN),
        ("plan", "Plan", ["Planner picks the", "specialists this", "paper needs, and why"], VIOLET_T),
        ("hex", "Analyse", ["Specialists run in", "parallel, each on its", "own targeted evidence"], VIOLET_T),
        ("verify", "Verify", ["Evidence re-retrieved", "independently; code", "checks quotes + numbers"], GREEN),
        ("report", "Report", ["Built from validated", "findings, no model", "call; gaps labelled"], AMBER),
    ]
    x0, cw, gap, y0, ch = 48, 172, 30, 186, 214
    la_end = x0 + 2 * cw + gap
    lb_start = x0 + 2 * (cw + gap)
    s.append(t(x0, 160, "PREPARE THE PAPER · once per PDF", 12, CYAN_T, "700", extra=' letter-spacing="1.3"'))
    s.append(f'<line x1="{x0}" y1="171" x2="{la_end}" y2="171" stroke="{CYAN}" stroke-opacity=".45"/>')
    s.append(t(lb_start, 160, "ANALYSE · a typed state graph of agents", 12, VIOLET_T, "700", extra=' letter-spacing="1.3"'))
    s.append(f'<line x1="{lb_start}" y1="171" x2="{W-48}" y2="171" stroke="{VIOLET}" stroke-opacity=".55"/>')
    for i, (ic, name, lines, c) in enumerate(steps):
        x = x0 + i * (cw + gap)
        s.append(card(x, y0, cw, ch, c, ".35"))
        s.append(f'<circle cx="{x+26}" cy="{y0+28}" r="13" fill="url(#grad)"/>')
        s.append(t(x + 26, y0 + 33, str(i + 1), 13, "#fff", "800", "middle"))
        s.append(tile(ic, x + cw - 72, y0 + 16, c))
        s.append(t(x + 16, y0 + 116, name, 17, INK, "800"))
        for k, ln in enumerate(lines):
            s.append(t(x + 16, y0 + 142 + k * 19, ln, 12.5, SOFT))
        if i < len(steps) - 1:
            ax = x + cw + 3
            s.append(f'<path class="flow" d="M{ax} {y0+ch/2} H{ax+gap-6}" stroke="{CYAN}" stroke-width="2" '
                     f'marker-end="url(#arrow)" fill="none"/>')
    vx = x0 + 4 * (cw + gap) + cw / 2
    ax_ = x0 + 3 * (cw + gap) + cw / 2
    by = y0 + ch + 2
    s.append(f'<path class="flow" d="M{vx} {by} C {vx} {by+62}, {ax_} {by+62}, {ax_} {by+8}" fill="none" '
             f'stroke="{ROSE}" stroke-width="2" marker-end="url(#arrowR)"/>')
    s.append(t((vx + ax_) / 2, by + 76, "claims below the threshold → only those agents retried, with feedback (bounded)",
               12.5, ROSE_T, "600", "middle"))
    s.append(t((x0 + la_end) / 2, by + 40, "a chunk never crosses a page boundary", 12.5, CYAN_T, "600", "middle"))
    yy = H - 52
    s.append(t(48, yy + 19, "BUILT WITH", 11, MUTED, "700", extra=' letter-spacing="1.4"'))
    x = 140
    for lab in ("Python 3.11", "FastAPI", "Pydantic v2", "PyMuPDF", "PP-Structure", "PaddleOCR-VL",
                "BM25 + dense", "OpenAI-compatible LLMs"):
        g, cwid = chip(x, yy, lab, LINE, "#cbd5e1", 12, 28, 12, False)
        s.append(g)
        x += cwid + 10
    s.append("</svg>")
    return "\n".join(s)


# =========================================================================== agents
def agents():
    W, H = 1280, 640
    s = [head(W, H, "The PaperMind agent team",
              "A planner chooses specialist agents (summary, methodology, experiments, and math only when the paper "
              "has mathematics); a verification agent re-checks their claims; a synthesizer writes the report "
              "without a model call.")]
    s.append(title_block("The agent team",
                         "No agent reads the whole PDF, and agents never pass prose to each other: only validated, typed findings."))

    def box(x, y, bw, bh, ic, c, label, name, lines, chip_text=None):
        o = [card(x, y, bw, bh, c, ".4"),
             f'<rect x="{x+14}" y="{y}" width="{bw-28}" height="3" rx="1.5" fill="{c}" fill-opacity=".9"/>',
             tile(ic, x + 20, y + 22, c, 52),
             t(x + 88, y + 41, label, 11, TINT.get(c, c), "700", extra=' letter-spacing="1.4"'),
             t(x + 88, y + 66, name, 20, INK, "800")]
        for k, ln in enumerate(lines):
            o.append(t(x + 20, y + 102 + k * 20, ln, 13.5, SOFT))
        if chip_text:
            g, _ = chip(x + 20, y + bh - 42, chip_text, c, TINT.get(c, c), 12, 26, 11)
            o.append(g)
        return "\n".join(o)

    y1, h1 = 140, 132
    s.append(box(48, y1, 300, h1, "paper", AMBER, "INPUT", "One paper",
                 ["A research-paper PDF,", "indexed page by page."]))
    s.append(box(446, y1, 388, h1, "plan", VIOLET_T, "TEAM LEAD", "Planner",
                 ["Chooses which specialists this paper", "needs, and says why."]))
    s.append(box(932, y1, 300, h1, "report", AMBER, "WRITER", "Synthesizer",
                 ["Renders the report from validated", "findings. No model call."]))
    my = y1 + h1 / 2
    s.append(f'<path class="flow" d="M352 {my} H440" stroke="{CYAN}" stroke-width="2" fill="none" marker-end="url(#arrow)"/>')
    s.append(f'<path class="flow" d="M838 {my} H926" stroke="{GREEN}" stroke-width="2" fill="none" marker-end="url(#arrowG)"/>')
    s.append(t(882, my - 12, "verified findings", 11.5, GREEN_T, "600", "middle"))

    y2, h2 = 352, 178
    bw, gp = 278, 24
    xs = [48 + i * (bw + gp) for i in range(4)]
    specs = [
        ("report", CYAN, "SPECIALIST", "Summary", ["Problem, contribution, data,", "method, results, limitations."], "always"),
        ("plan", VIOLET, "SPECIALIST", "Methodology", ["Architecture, training, inference,", "design decisions, prior work."], "always"),
        ("chart", TEAL, "SPECIALIST", "Experiments", ["Datasets, baselines, metrics,", "ablations, significance."], "when results are found"),
        ("sigma", AMBER, "SPECIALIST", "Math", ["Equations, variables, losses,", "objectives, derivations."], "only if the paper has maths"),
    ]
    for x, (ic, c, lab, name, lines, ct) in zip(xs, specs):
        s.append(box(x, y2, bw, h2, ic, c, lab, name, lines, ct))
    sb = y1 + h1 + 2
    for x, sx in zip(xs, (580, 620, 660, 700)):
        tx = x + bw / 2
        s.append(f'<path class="flow" d="M{sx} {sb} C {sx} {sb+44}, {tx} {y2-44}, {tx} {y2-6}" fill="none" '
                 f'stroke="{VIOLET_T}" stroke-width="2" stroke-opacity=".9" marker-end="url(#arrowV)"/>')
    pill = "targeted evidence ↓   typed findings ↑"
    pw = w(pill, 12, mono=True) + 28
    s.append(f'<rect x="{640-pw/2:.0f}" y="{(sb+y2)/2-15:.0f}" width="{pw:.0f}" height="28" rx="14" fill="{BG}" '
             f'stroke="{VIOLET}" stroke-opacity=".6"/>')
    s.append(t(640, (sb + y2) / 2 + 3.5, pill, 12, VIOLET_T, "600", "middle", cls="mono"))
    vy = H - 76
    for x in xs:
        s.append(f'<path d="M{x+bw/2} {y2+h2+2} V{vy-4}" stroke="{GREEN}" stroke-opacity=".5" stroke-dasharray="2 4"/>')
    s.append(card(48, vy, W - 96, 54, GREEN, ".5", shadow=False, rx=14))
    s.append(icon("verify", 80, vy + 27, GREEN_T, 2))
    s.append(f'<text class="sans" x="112" y="{vy+32}" font-size="13.5" fill="{SOFT}"><tspan font-size="14.5" '
             f'font-weight="700" fill="{INK}">Verification agent:</tspan> re-retrieves evidence for every high-impact '
             f'claim on its own, then code checks pages, quotes and numbers. Failed agents are retried.</text>')
    s.append("</svg>")
    return "\n".join(s)


# =========================================================================== verification
def verification():
    W, H = 1280, 580
    s = [head(W, H, "How a claim earns a citation",
              "A claim is re-checked by code: page in range, quote present, section resolves, numbers match. Quotes "
              "match exactly, after normalising, approximately (labelled), or not at all; an unverifiable quote is "
              "never shown as a citation but kept in the evidence appendix.")]
    s.append(title_block("How a claim earns a citation",
                         "A citation appears only when its quote was found at the cited place. Failures are labelled, never hidden."))
    top = 150
    for x, lab, c in ((48, "1 · CLAIM FROM AN AGENT", CYAN_T), (428, "2 · CODE CHECKS", VIOLET_T),
                      (836, "3 · QUOTE MATCH", GREEN_T)):
        s.append(t(x, top, lab, 12, c, "700", extra=' letter-spacing="1.4"'))
    # claim
    cx, cy, cw, ch = 48, top + 16, 340, 300
    s.append(card(cx, cy, cw, ch, CYAN, ".45"))
    s.append(t(cx + 22, cy + 38, "EXPERIMENTS AGENT", 11, CYAN_T, "700", extra=' letter-spacing="1.3"'))
    s.append(t(cx + 22, cy + 74, "SRT-Large reaches 81.4 F1", 18, INK, "800"))
    s.append(t(cx + 22, cy + 98, "on NarrativeQA.", 18, INK, "800"))
    s.append(f'<rect x="{cx+22}" y="{cy+120}" width="{cw-44}" height="92" rx="10" fill="#111a2e" stroke="{LINE}" stroke-opacity=".18"/>')
    s.append(t(cx + 36, cy + 145, "cites [S3] →", 12.5, MUTED, None, cls="mono"))
    s.append(t(cx + 36, cy + 168, "p. 4 · §4.2 Main Results", 12.5, CYAN_T, None, cls="mono"))
    s.append(t(cx + 36, cy + 191, "Table 1", 12.5, CYAN_T, None, cls="mono"))
    s.append(t(cx + 22, cy + 244, "The verifier fetches its own evidence,", 12.5, MUTED))
    s.append(t(cx + 22, cy + 263, "not the agent's, before it rules.", 12.5, MUTED))
    # checks
    kx, ky, kw, kh = 428, top + 16, 368, 300
    s.append(card(kx, ky, kw, kh, VIOLET, ".45"))
    for i, ck in enumerate(("Page is inside the PDF", "Quote occurs in the cited text", "Section resolves in the index",
                            "Numbers match their evidence", "Fields and confidence are valid")):
        yy = ky + 46 + i * 40
        s.append(f'<circle cx="{kx+33}" cy="{yy-5}" r="11" fill="{GREEN}" fill-opacity=".16" stroke="{GREEN}" stroke-opacity=".6"/>')
        s.append(f'<path d="M{kx+28} {yy-5}l3.5 3.5 6.5-7" fill="none" stroke="{GREEN_T}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>')
        s.append(t(kx + 54, yy, ck, 14, INK))
    s.append(f'<line x1="{kx+22}" y1="{ky+250}" x2="{kx+kw-22}" y2="{ky+250}" stroke="{LINE}" stroke-opacity=".2"/>')
    s.append(t(kx + 22, ky + 276, "Code can only lower the model's verdict, never raise it.", 12.5, VIOLET_T, "600"))
    # tiers
    tx, ty, tw, th = 836, top + 16, 396, 300
    s.append(card(tx, ty, tw, th, GREEN, ".45"))
    tiers = (("Exact", 1.0, GREEN, "cited"), ("Normalised", 0.82, GREEN, "cited"),
             ("Approximate", 0.55, AMBER, "cited, labelled"), ("Unverifiable", 0.12, ROSE, "not cited"))
    for i, (name, frac, c, out) in enumerate(tiers):
        yy = ty + 30 + i * 62
        s.append(t(tx + 22, yy + 14, name, 15, INK, "700"))
        s.append(t(tx + tw - 22, yy + 14, out, 12.5, TINT[c], "600", "end"))
        bw_ = tw - 44
        s.append(f'<rect x="{tx+22}" y="{yy+24}" width="{bw_}" height="10" rx="5" fill="#111a2e" stroke="{LINE}" stroke-opacity=".15"/>')
        s.append(f'<rect x="{tx+22}" y="{yy+24}" width="{bw_*frac:.0f}" height="10" rx="5" fill="{c}" fill-opacity=".85"/>')
    s.append(t(tx + 22, ty + th - 18, "bar = confidence kept after the match", 11.5, MUTED, None, extra=' font-style="italic"'))
    s.append(f'<path class="flow" d="M{cx+cw+4} {cy+150} H{kx-8}" stroke="{CYAN}" stroke-width="2" fill="none" marker-end="url(#arrow)"/>')
    s.append(f'<path class="flow" d="M{kx+kw+4} {ky+150} H{tx-8}" stroke="{VIOLET_T}" stroke-width="2" fill="none" marker-end="url(#arrowV)"/>')
    ry = H - 74
    s.append(f'<rect x="48" y="{ry}" width="{W-96}" height="44" rx="12" fill="{GREEN}" fill-opacity=".08" stroke="{GREEN}" stroke-opacity=".45"/>')
    s.append(t(70, ry + 27.5, "ANNOTATE AND DOWNGRADE", 12, GREEN_T, "800", extra=' letter-spacing="1.2"'))
    s.append(t(292, ry + 27.5, "A citation that fails is never silently deleted: it stays in the evidence appendix, marked unverifiable.",
               13.5, "#d1fae5"))
    s.append("</svg>")
    return "\n".join(s)


for name, fn in (("banner", banner), ("pipeline", pipeline), ("agents", agents), ("verification", verification)):
    (OUT / f"{name}.svg").write_text(fn(), encoding="utf-8")
    print("wrote", name)
