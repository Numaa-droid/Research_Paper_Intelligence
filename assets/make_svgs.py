"""Draw PaperMind's README graphics. Run from the repo root: python assets/make_svgs.py assets

A "paper and ink" look: cream paper, navy ink, serif headings, a highlighter, margin notes.
"""
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

PAPER, SHEET, INK, MUTED, RULE = "#fbf7ef", "#ffffff", "#1f2a44", "#5b6475", "#e7dcc6"
CORAL, CORAL_D = "#d4573a", "#a8412a"
GREEN, GREEN_D = "#2f855a", "#22643f"
MUSTARD, MUSTARD_D = "#c98a12", "#8f6209"
BLUE, BLUE_D = "#2b6cb0", "#1f4f82"
HILITE = "#fde68a"
SOFT = {CORAL: "#fbe4dc", GREEN: "#dcefe3", MUSTARD: "#f8ecc9", BLUE: "#dce8f5", INK: "#e6e8ee"}
DARK = {CORAL: CORAL_D, GREEN: GREEN_D, MUSTARD: MUSTARD_D, BLUE: BLUE_D, INK: INK}


def w(text, size, bold=False, mono=False, serif=False):
    k = 0.61 if mono else (0.55 if serif else (0.57 if bold else 0.53))
    return len(text) * size * k


def t(x, y, s, size=14, fill=INK, weight=None, anchor=None, cls="sans", extra=""):
    a = f' font-size="{size}" fill="{fill}"'
    if weight:
        a += f' font-weight="{weight}"'
    if anchor:
        a += f' text-anchor="{anchor}"'
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}"{a}{extra}>{escape(s)}</text>'


def head(W, H, title, desc):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">
  <title id="t">{escape(title)}</title>
  <desc id="d">{escape(desc)}</desc>
  <defs>
    <pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.1" fill="#c9b98f" fill-opacity=".45"/></pattern>
    <filter id="lift" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="8" stdDeviation="9" flood-color="#5c4a24" flood-opacity=".18"/></filter>
    <marker id="aInk" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{INK}"/></marker>
    <marker id="aCoral" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{CORAL}"/></marker>
    <marker id="aGreen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{GREEN}"/></marker>
    <style>
      .serif{{font-family:Georgia,"Iowan Old Style","Times New Roman",serif}}
      .sans{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
      .mono{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}}
      .ink{{stroke-dasharray:7 6;animation:ink 1.4s linear infinite}}
      @keyframes ink{{to{{stroke-dashoffset:-26}}}}
      .glow{{animation:glow 2.6s ease-in-out infinite}}
      @keyframes glow{{0%,100%{{opacity:.6}}50%{{opacity:1}}}}
      @media (prefers-reduced-motion:reduce){{.ink,.glow{{animation:none}}}}
    </style>
  </defs>
  <rect width="{W}" height="{H}" rx="18" fill="{PAPER}"/>
  <rect width="{W}" height="{H}" rx="18" fill="url(#dots)"/>
  <rect x="1" y="1" width="{W-2}" height="{H-2}" rx="17" fill="none" stroke="{RULE}" stroke-width="2"/>
'''


def heading(title, sub, x=56, y=74):
    return (t(x, y, title, 32, INK, "700", cls="serif", extra=' letter-spacing="-.3"')
            + f'<rect x="{x}" y="{y+12}" width="56" height="4" rx="2" fill="{CORAL}"/>'
            + t(x, y + 42, sub, 15.5, MUTED))


def tag(x, y, s, c, size=13):
    """An outlined label with a coloured bar on the left (not a pill)."""
    tw = w(s, size, True) + 30
    return (f'<rect x="{x}" y="{y}" width="{tw:.0f}" height="32" rx="5" fill="{SHEET}" stroke="{c}" stroke-width="1.4"/>'
            f'<rect x="{x}" y="{y}" width="6" height="32" rx="3" fill="{c}"/>'
            + t(x + 18, y + 21, s, size, DARK[c], "600")), tw


# =========================================================================== banner
TABLE = [("Dense-Large", "76.2", "61.4"), ("Longformer-Large", "77.9", "63.0"), ("BM25 + Reader", "74.1", "59.8"),
         ("SRT-Base", "79.3", "65.1"), ("SRT-Large", "81.4", "68.7")]


def paper_sheet(x, y, pw, ph):
    """A research-paper page: title, section 4.2 and Table 1, with the SRT-Large row highlighted."""
    s = [f'<rect x="{x}" y="{y}" width="{pw}" height="{ph}" rx="4" fill="{SHEET}" stroke="{RULE}" filter="url(#lift)"/>']
    ix, iw = x + 24, pw - 48
    s.append(t(x + pw / 2, y + 34, "Sparse Routing Transformers", 13, INK, "700", "middle", cls="serif"))
    s.append(t(x + pw / 2, y + 51, "for Long-Document Question Answering", 10.5, MUTED, None, "middle", cls="serif"))
    for k, frac in enumerate((1.0, 0.96, 0.99, 0.62)):
        s.append(f'<line x1="{ix}" y1="{y+72+k*11}" x2="{ix+iw*frac:.0f}" y2="{y+72+k*11}" stroke="#d8d2c4" stroke-width="3" stroke-linecap="round"/>')
    s.append(t(ix, y + 132, "4.2  Main Results", 12, INK, "700", cls="serif"))
    s.append(t(ix, y + 150, "Table 1: Main results on long-document QA.", 9, MUTED, None, cls="serif", extra=' font-style="italic"'))
    ty = y + 160
    c1, c2 = ix + iw * 0.72, ix + iw
    s.append(f'<line x1="{ix}" y1="{ty}" x2="{ix+iw}" y2="{ty}" stroke="{INK}" stroke-width="1.2"/>')
    s.append(t(ix + 2, ty + 15, "Model", 9.4, INK, "700", cls="serif"))
    s.append(t(c1, ty + 15, "NQA F1", 9.4, INK, "700", "end", cls="serif"))
    s.append(t(c2 - 2, ty + 15, "QuALITY", 9.4, INK, "700", "end", cls="serif"))
    s.append(f'<line x1="{ix}" y1="{ty+21}" x2="{ix+iw}" y2="{ty+21}" stroke="{INK}" stroke-width=".7"/>')
    anchor = None
    for i, (m, f1, q) in enumerate(TABLE):
        ry = ty + 22 + i * 21
        last = i == len(TABLE) - 1
        if last:
            s.append(f'<rect class="glow" x="{ix-3}" y="{ry+2}" width="{iw+6}" height="18" rx="3" fill="{HILITE}"/>')
            anchor = (ix + iw + 6, ry + 11)
        b = "700" if last else None
        s.append(t(ix + 2, ry + 15, m, 9.4, INK, b, cls="serif"))
        s.append(t(c1, ry + 15, f1, 9.4, INK, b, "end", cls="serif"))
        s.append(t(c2 - 2, ry + 15, q, 9.4, INK, b, "end", cls="serif"))
    end = ty + 22 + len(TABLE) * 21 + 2
    s.append(f'<line x1="{ix}" y1="{end}" x2="{ix+iw}" y2="{end}" stroke="{INK}" stroke-width="1.2"/>')
    for k, frac in enumerate((1.0, 0.9, 0.55)):
        if end + 18 + k * 11 < y + ph - 28:
            s.append(f'<line x1="{ix}" y1="{end+18+k*11}" x2="{ix+iw*frac:.0f}" y2="{end+18+k*11}" stroke="#d8d2c4" stroke-width="3" stroke-linecap="round"/>')
    s.append(t(x + pw / 2, y + ph - 14, "4", 10, MUTED, None, "middle", cls="serif"))
    return "\n".join(s), anchor


def banner():
    W, H = 1280, 460
    s = [head(W, H, "PaperMind: citation-grounded multi-agent analysis of research papers",
              "A research-paper page with its Table 1 row highlighted, and a margin note: the claim that SRT-Large "
              "reaches 81.4 F1 on NarrativeQA is supported, its quote was found and its number matches.")]
    s.append(f'<circle cx="86" cy="80" r="24" fill="{CORAL}"/>')
    s.append(f'<path d="M73 72c5-3 9-3 13 0v17c-4-3-8-3-13 0zM99 72c-5-3-9-3-13 0v17c4-3 8-3 13 0z" fill="none" '
             f'stroke="#fff" stroke-width="2" stroke-linejoin="round"/>')
    s.append(t(122, 76, "PaperMind", 20, INK, "700", cls="serif"))
    s.append(t(122, 97, "Citation-grounded analysis of research papers", 13.5, MUTED))
    s.append(t(64, 186, "Every claim, checked", 46, INK, "700", cls="serif", extra=' letter-spacing="-.8"'))
    hw = len("against the page.") * 46 * 0.5
    s.append(f'<rect x="60" y="222" width="{hw+6:.0f}" height="24" rx="4" fill="{HILITE}"/>')
    s.append(t(64, 242, "against the page.", 46, CORAL_D, "700", cls="serif", extra=' letter-spacing="-.8"'))
    s.append(t(64, 294, "Specialist agents read one research paper. A verifier", 18.5, MUTED))
    s.append(t(64, 320, "re-checks every high-impact claim against the PDF.", 18.5, MUTED))
    x = 64
    for label, c in (("Page-level citations", BLUE), ("Deterministic checks", GREEN), ("Bounded retries", CORAL)):
        g, tw = tag(x, 356, label, c)
        s.append(g)
        x += tw + 12
    px, py, pw, ph = 650, 56, 300, 350
    page, (hx, hy) = paper_sheet(px, py, pw, ph)
    s.append(f'<g transform="rotate(-2 {px+pw/2} {py+ph/2})">{page}</g>')
    nx, ny, nw, nh = 958, 118, 268, 250
    note = [f'<rect x="{nx}" y="{ny}" width="{nw}" height="{nh}" rx="3" fill="#fffdf6" stroke="{RULE}" filter="url(#lift)"/>',
            f'<rect x="{nx+nw/2-34}" y="{ny-9}" width="68" height="18" rx="2" fill="{HILITE}" fill-opacity=".85" '
            f'transform="rotate(-3 {nx+nw/2} {ny})"/>']
    note.append(f'<g class="glow" transform="rotate(-8 {nx+nw-62} {ny+46})">'
                f'<rect x="{nx+nw-118}" y="{ny+30}" width="104" height="32" rx="5" fill="none" stroke="{GREEN}" stroke-width="2.4"/>'
                + t(nx + nw - 66, ny + 52, "SUPPORTED", 13.5, GREEN, "800", "middle", extra=' letter-spacing="1.5"') + "</g>")
    note.append(t(nx + 20, ny + 46, "Claim", 13, MUTED, "700", cls="serif", extra=' font-style="italic"'))
    note.append(t(nx + 20, ny + 92, "SRT-Large reaches", 18, INK, "700", cls="serif"))
    note.append(t(nx + 20, ny + 116, "81.4 F1 on NarrativeQA.", 18, INK, "700", cls="serif"))
    for i, (lab, val, c) in enumerate((("quote", "found on the page", GREEN),
                                       ("number", "81.4 = 81.4", GREEN),
                                       ("cite", "[p. 4, §4.2, Table 1]", BLUE))):
        yy = ny + 156 + i * 28
        note.append(f'<line x1="{nx+16}" y1="{yy+9}" x2="{nx+nw-16}" y2="{yy+9}" stroke="{RULE}"/>')
        note.append(t(nx + 20, yy, lab, 12, MUTED, None, cls="serif", extra=' font-style="italic"'))
        note.append(t(nx + 80, yy, val, 12, DARK[c], "600", cls="mono"))
    s.append(f'<g transform="rotate(1.5 {nx+nw/2} {ny+nh/2})">' + "\n".join(note) + "</g>")
    a = math.radians(-2)
    cx0, cy0 = px + pw / 2, py + ph / 2
    rx = cx0 + (hx - cx0) * math.cos(a) - (hy - cy0) * math.sin(a)
    ry = cy0 + (hx - cx0) * math.sin(a) + (hy - cy0) * math.cos(a)
    s.append(f'<path class="ink" d="M{rx:.0f} {ry:.0f} C {rx+24:.0f} {ry+10:.0f}, {nx-30} {ny+170}, {nx+10} {ny+182}" '
             f'fill="none" stroke="{CORAL}" stroke-width="2.2" stroke-linecap="round" marker-end="url(#aCoral)"/>')
    s.append(t(1216, 440, "example from the bundled synthetic test paper", 11, MUTED, None, "end", cls="serif",
               extra=' font-style="italic"'))
    s.append("</svg>")
    return "\n".join(s)


# =========================================================================== pipeline (a metro line)
def pipeline():
    W, H = 1280, 540
    s = [head(W, H, "How PaperMind analyses a paper",
              "A line with six stations: ingest and index the PDF once, then plan, analyse with specialist agents, "
              "verify and report. Agents whose claims fail to ground loop back for a limited number of retries.")]
    s.append(heading("How PaperMind analyses a paper",
                     "The PDF is indexed once, page by page. Then a planned team of agents analyses it; nothing is cited unchecked."))
    xs = [140 + i * 200 for i in range(6)]
    ly = 268
    steps = [("Ingest", ["PDF parsed page by", "page; OCR fallback", "for scanned pages"], BLUE),
             ("Index", ["page-aware chunks;", "BM25 + dense search", "fused with RRF"], BLUE),
             ("Plan", ["the planner picks", "the specialists this", "paper needs, and why"], CORAL),
             ("Analyse", ["specialists run in", "parallel, each on its", "own targeted evidence"], CORAL),
             ("Verify", ["evidence re-retrieved", "independently; code", "checks quotes, numbers"], GREEN),
             ("Report", ["built from validated", "findings, no model", "call; gaps labelled"], MUSTARD)]
    for x0, x1, lab, c in ((xs[0] - 50, xs[1] + 50, "PREPARE THE PAPER  ·  once per PDF", BLUE),
                           (xs[2] - 50, xs[5] + 50, "ANALYSE  ·  a typed state graph of agents", CORAL)):
        s.append(f'<path d="M{x0} 176 v-8 H{x1} v8" fill="none" stroke="{c}" stroke-width="1.6"/>')
        s.append(t((x0 + x1) / 2, 158, lab, 12, DARK[c], "700", "middle", extra=' letter-spacing="1.6"'))
    s.append(f'<line x1="{xs[0]-60}" y1="{ly}" x2="{xs[1]+100}" y2="{ly}" stroke="{BLUE}" stroke-width="8" stroke-linecap="round"/>')
    s.append(f'<line x1="{xs[1]+100}" y1="{ly}" x2="{xs[5]+60}" y2="{ly}" stroke="{CORAL}" stroke-width="8" stroke-linecap="round"/>')
    s.append(f'<line class="ink" x1="{xs[0]-60}" y1="{ly}" x2="{xs[5]+60}" y2="{ly}" stroke="#fff" stroke-width="2" stroke-opacity=".8"/>')
    for i, (x, (name, lines, c)) in enumerate(zip(xs, steps)):
        s.append(t(x, ly - 44, name, 21, INK, "700", "middle", cls="serif"))
        s.append(f'<circle cx="{x}" cy="{ly}" r="24" fill="{SHEET}" stroke="{c}" stroke-width="5"/>')
        s.append(t(x, ly + 7, str(i + 1), 20, DARK[c], "700", "middle", cls="serif"))
        for k, ln in enumerate(lines):
            s.append(t(x, ly + 60 + k * 19, ln, 13.5, MUTED, None, "middle"))
    by = ly + 124
    s.append(f'<path class="ink" d="M{xs[4]} {by} C {xs[4]} {by+56}, {xs[3]} {by+56}, {xs[3]} {by+6}" fill="none" '
             f'stroke="{CORAL}" stroke-width="2.2" marker-end="url(#aCoral)"/>')
    s.append(t((xs[3] + xs[4]) / 2, by + 74, "claims below the threshold: only those agents re-run, with feedback (bounded)",
               13, CORAL_D, "600", "middle", cls="serif", extra=' font-style="italic"'))
    s.append(t((xs[0] + xs[1]) / 2, by + 30, "a chunk never crosses a page boundary", 13, BLUE_D, "600", "middle",
               cls="serif", extra=' font-style="italic"'))
    s.append(f'<line x1="56" y1="{H-58}" x2="{W-56}" y2="{H-58}" stroke="{RULE}" stroke-width="1.5"/>')
    s.append(t(56, H - 30, "Built with", 13, INK, "700", cls="serif", extra=' font-style="italic"'))
    s.append(t(140, H - 30, "Python 3.11 · FastAPI · Pydantic v2 · PyMuPDF · PP-Structure · PaddleOCR-VL · "
               "BM25 + dense retrieval · OpenAI-compatible LLMs", 13, MUTED, None, cls="mono"))
    s.append("</svg>")
    return "\n".join(s)


# =========================================================================== agents (hub, tabs, merge)
def agents():
    W, H = 1280, 600
    s = [head(W, H, "The PaperMind agent team",
              "The planner fans a paper out to specialist agents: summary, methodology, experiments, and math only "
              "when the paper has mathematics. Their findings merge into the verification agent, then the synthesizer.")]
    s.append(heading("The agent team",
                     "No agent reads the whole PDF, and agents never pass prose to each other: only validated, typed findings."))
    cy = 362
    dx, dy = 60, cy - 70
    s.append(f'<path d="M{dx} {dy} h92 l26 26 v114 h-118 z" fill="{SHEET}" stroke="{INK}" stroke-width="1.6" filter="url(#lift)"/>')
    s.append(f'<path d="M{dx+92} {dy} v26 h26" fill="none" stroke="{INK}" stroke-width="1.6"/>')
    for k in range(5):
        s.append(f'<line x1="{dx+16}" y1="{dy+46+k*16}" x2="{dx+(100 if k < 4 else 62)}" y2="{dy+46+k*16}" stroke="#d8d2c4" stroke-width="3" stroke-linecap="round"/>')
    s.append(t(dx + 59, dy + 168, "one paper", 15, INK, "700", "middle", cls="serif", extra=' font-style="italic"'))
    hx, hr = 330, 82
    s.append(f'<path class="ink" d="M{dx+124} {cy} H{hx-hr-14}" stroke="{INK}" stroke-width="2" marker-end="url(#aInk)"/>')
    s.append(f'<circle cx="{hx}" cy="{cy}" r="{hr+10}" fill="none" stroke="{CORAL}" stroke-width="1.4" stroke-dasharray="3 5"/>')
    s.append(f'<circle cx="{hx}" cy="{cy}" r="{hr}" fill="{INK}"/>')
    s.append(t(hx, cy - 6, "Planner", 22, "#fff", "700", "middle", cls="serif"))
    s.append(t(hx, cy + 16, "picks the specialists", 12, "#cfd5e2", None, "middle"))
    s.append(t(hx, cy + 32, "and says why", 12, "#cfd5e2", None, "middle"))
    tx, tw, th, gap = 520, 400, 78, 22
    specs = [("S", "Summary", "problem, contribution, data, results, limits", "ALWAYS", BLUE),
             ("M", "Methodology", "architecture, training, inference, design", "ALWAYS", CORAL),
             ("E", "Experiments", "datasets, baselines, metrics, ablations", "WHEN RESULTS ARE FOUND", GREEN),
             ("Σ", "Math", "equations, variables, losses, derivations", "ONLY IF THE PAPER HAS MATHS", MUSTARD)]
    total = 4 * th + 3 * gap
    top = cy - total / 2
    mids = []
    for i, (ch, name, desc, when, c) in enumerate(specs):
        y = top + i * (th + gap)
        mids.append(y + th / 2)
        tab_w = w(when, 9.5, True) + len(when) * 0.5 + 22
        s.append(f'<path d="M{tx+14} {y} v-12 q0 -4 4 -4 h{tab_w:.0f} q4 0 4 4 v12 z" fill="{SOFT[c]}" stroke="{c}" stroke-width="1.4"/>')
        s.append(t(tx + 25, y - 4, when, 9.5, DARK[c], "700", extra=' letter-spacing=".5"'))
        s.append(f'<rect x="{tx}" y="{y}" width="{tw}" height="{th}" rx="8" fill="{SHEET}" stroke="{c}" stroke-width="1.6" filter="url(#lift)"/>')
        s.append(f'<circle cx="{tx+38}" cy="{y+th/2}" r="20" fill="{SOFT[c]}" stroke="{c}" stroke-width="1.6"/>')
        s.append(t(tx + 38, y + th / 2 + 7, ch, 19, DARK[c], "700", "middle", cls="serif"))
        s.append(t(tx + 72, y + th / 2 - 4, name, 18, INK, "700", cls="serif"))
        s.append(t(tx + 72, y + th / 2 + 17, desc, 12.5, MUTED))
        s.append(f'<path class="ink" d="M{hx+hr+12} {cy} C {hx+hr+70} {cy}, {tx-70} {y+th/2}, {tx-8} {y+th/2}" '
                 f'fill="none" stroke="{c}" stroke-width="2" marker-end="url(#aInk)"/>')
    vx, vy, vw, vh = 1010, cy - 128, 222, 120
    s.append(f'<rect x="{vx}" y="{vy}" width="{vw}" height="{vh}" rx="10" fill="{SOFT[GREEN]}" stroke="{GREEN}" stroke-width="1.8"/>')
    s.append(f'<path d="M{vx+34} {vy+20}l16 6v12c0 11-7 18-16 22c-9-4-16-11-16-22v-12z" fill="{SHEET}" stroke="{GREEN}" stroke-width="2"/>'
             f'<path d="M{vx+27} {vy+39}l5 5 9-10" fill="none" stroke="{GREEN}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>')
    s.append(t(vx + 62, vy + 38, "Verification", 18, GREEN_D, "700", cls="serif"))
    s.append(t(vx + 62, vy + 57, "agent", 13, GREEN_D))
    s.append(t(vx + 18, vy + 88, "fetches the evidence itself;", 12.5, INK))
    s.append(t(vx + 18, vy + 106, "code checks quotes + numbers", 12.5, INK))
    for m in mids:
        s.append(f'<path d="M{tx+tw+4} {m} C {tx+tw+50} {m}, {vx-50} {vy+vh/2}, {vx-6} {vy+vh/2}" fill="none" '
                 f'stroke="{GREEN}" stroke-width="1.6" stroke-opacity=".75" marker-end="url(#aGreen)"/>')
    sy = vy + vh + 58
    s.append(f'<path class="ink" d="M{vx+vw/2} {vy+vh+4} V{sy-6}" stroke="{GREEN}" stroke-width="2" marker-end="url(#aGreen)"/>')
    s.append(t(vx + vw / 2 + 10, vy + vh + 34, "verified findings", 12, GREEN_D, "600", cls="serif", extra=' font-style="italic"'))
    s.append(f'<rect x="{vx}" y="{sy}" width="{vw}" height="86" rx="10" fill="{SOFT[MUSTARD]}" stroke="{MUSTARD}" stroke-width="1.8"/>')
    s.append(t(vx + 18, sy + 32, "Synthesizer", 18, MUSTARD_D, "700", cls="serif"))
    s.append(t(vx + 18, sy + 55, "writes the report from validated", 12.5, INK))
    s.append(t(vx + 18, sy + 73, "findings. No model call.", 12.5, INK))
    s.append(f'<path d="M{vx+30} {vy-6} C {vx-10} {vy-60}, {tx+tw+10} {top-70}, {tx+tw-30} {top-22}" fill="none" '
             f'stroke="{CORAL}" stroke-width="1.8" stroke-dasharray="5 5" marker-end="url(#aCoral)"/>')
    s.append(t(vx + 40, vy - 40, "failed to ground: retried", 12.5, CORAL_D, "600", cls="serif", extra=' font-style="italic"'))
    s.append("</svg>")
    return "\n".join(s)


# =========================================================================== verification (slip, ledger, staircase)
def verification():
    W, H = 1280, 560
    s = [head(W, H, "How a claim earns a citation",
              "A claim from an agent is re-checked by code: page in range, quote present, section resolves, numbers "
              "match. Quotes match exactly, after normalising, approximately (labelled), or not at all; an "
              "unverifiable quote is never shown as a citation but kept in the evidence appendix.")]
    s.append(heading("How a claim earns a citation",
                     "A citation appears only when its quote was found at the cited place. Failures are labelled, never hidden."))
    top = 164
    cx, cy, cw, ch = 56, top, 330, 270
    s.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="4" fill="{SHEET}" stroke="{RULE}" filter="url(#lift)" '
             f'transform="rotate(-1.5 {cx+cw/2} {cy+ch/2})"/>')
    s.append(t(cx + 22, cy + 40, "1 · AN AGENT CLAIMS", 12.5, BLUE_D, "700", extra=' letter-spacing="1"'))
    s.append(t(cx + 18, cy + 108, "“", 70, BLUE, "700", cls="serif", extra=' fill-opacity=".35"'))
    s.append(t(cx + 56, cy + 92, "SRT-Large reaches", 20, INK, "700", cls="serif"))
    s.append(t(cx + 56, cy + 118, "81.4 F1 on NarrativeQA.”", 20, INK, "700", cls="serif"))
    s.append(f'<line x1="{cx+22}" y1="{cy+146}" x2="{cx+cw-22}" y2="{cy+146}" stroke="{RULE}"/>')
    s.append(t(cx + 22, cy + 172, "cites [S3] → p. 4, §4.2, Table 1", 12.5, BLUE_D, "600", cls="mono"))
    s.append(t(cx + 22, cy + 212, "The verifier then fetches its own", 13.5, MUTED, None, cls="serif", extra=' font-style="italic"'))
    s.append(t(cx + 22, cy + 232, "evidence, not the agent's.", 13.5, MUTED, None, cls="serif", extra=' font-style="italic"'))
    lx, ly, lw, lh = 450, top, 370, 270
    s.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" rx="4" fill="#fffdf6" stroke="{RULE}" filter="url(#lift)"/>')
    s.append(f'<line x1="{lx+44}" y1="{ly}" x2="{lx+44}" y2="{ly+lh}" stroke="{CORAL}" stroke-opacity=".45"/>')
    s.append(t(lx + 60, ly + 36, "2 · CHECKED BY CODE", 12.5, GREEN_D, "700", extra=' letter-spacing="1"'))
    for i, ck in enumerate(("page is inside the PDF", "quote occurs in the cited text", "section resolves in the index",
                            "numbers match their evidence", "fields and confidence are valid")):
        yy = ly + 80 + i * 34
        s.append(f'<line x1="{lx}" y1="{yy+12}" x2="{lx+lw}" y2="{yy+12}" stroke="{BLUE}" stroke-opacity=".22"/>')
        s.append(f'<rect x="{lx+60}" y="{yy-12}" width="17" height="17" rx="3" fill="{SHEET}" stroke="{GREEN}" stroke-width="1.8"/>')
        s.append(f'<path d="M{lx+63} {yy-4}l4 4 8-10" fill="none" stroke="{GREEN}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>')
        s.append(t(lx + 88, yy + 2, ck, 14.5, INK, None, cls="serif"))
    s.append(t(lx + 60, ly + lh - 20, "Code can lower the model's verdict, never raise it.", 12.5, GREEN_D, "600"))
    sx0, base, sw = 884, top + 250, 88
    tiers = (("exact", "cited", GREEN, 182), ("normalised", "cited", GREEN, 142),
             ("approximate", "labelled", MUSTARD, 96), ("unverifiable", "not cited", CORAL, 40))
    s.append(t(sx0, top + 4, "3 · HOW WELL THE QUOTE MATCHED", 12.5, MUSTARD_D, "700", extra=' letter-spacing="1"'))
    for i, (name, out, c, hgt) in enumerate(tiers):
        x = sx0 + i * sw
        s.append(f'<rect x="{x}" y="{base-hgt}" width="{sw-6}" height="{hgt}" rx="3" fill="{SOFT[c]}" stroke="{c}" stroke-width="1.6"/>')
        s.append(t(x + (sw - 6) / 2, base - hgt - 24, name, 13, INK, "700", "middle", cls="serif"))
        s.append(t(x + (sw - 6) / 2, base - hgt - 8, out, 11.5, DARK[c], "600", "middle"))
    s.append(f'<line x1="{sx0-6}" y1="{base}" x2="{sx0+4*sw}" y2="{base}" stroke="{INK}" stroke-width="1.6"/>')
    s.append(t(sx0, base + 22, "step height = confidence kept", 12, MUTED, None, cls="serif", extra=' font-style="italic"'))
    s.append(f'<path class="ink" d="M{cx+cw+8} {cy+130} H{lx-12}" stroke="{INK}" stroke-width="2" marker-end="url(#aInk)"/>')
    s.append(f'<path class="ink" d="M{lx+lw+8} {ly+130} H{sx0-16}" stroke="{INK}" stroke-width="2" marker-end="url(#aInk)"/>')
    by = H - 72
    s.append(f'<g transform="rotate(-3 180 {by+20})"><rect x="60" y="{by}" width="244" height="40" rx="4" fill="none" '
             f'stroke="{CORAL}" stroke-width="2.4"/>' + t(182, by + 26, "ANNOTATE & DOWNGRADE", 14, CORAL, "800", "middle",
                                                          extra=' letter-spacing="1.6"') + "</g>")
    s.append(t(330, by + 26, "A citation that fails is never silently deleted: it stays in the evidence appendix, marked unverifiable.",
               15, INK, None, cls="serif"))
    s.append("</svg>")
    return "\n".join(s)


for name, fn in (("banner", banner), ("pipeline", pipeline), ("agents", agents), ("verification", verification)):
    (OUT / f"{name}.svg").write_text(fn(), encoding="utf-8")
    print("wrote", name)
