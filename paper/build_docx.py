"""
Build the IEEE-format .docx for the SmartDetect paper.

Two-column body with continuous section breaks for full-width floats,
Times New Roman, IEEE heading conventions. All numbers come from the real
ablation output; figures are the PNGs generated from the actual run.
"""
from __future__ import annotations
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

SCRATCH = Path("/private/tmp/claude-501/-Users-lokii-Downloads-Smart-Detect-main/"
               "4e0d6171-becd-4766-a354-d53068b6a2ba/scratchpad")
FIGS = SCRATCH / "figs"
OUT = SCRATCH / "SmartDetect_IEEE_Paper.docx"

INK = RGBColor(0x14, 0x16, 0x1a)
MUT = RGBColor(0x50, 0x55, 0x60)
ACC = RGBColor(0x1f, 0x3a, 0x68)
OXI = RGBColor(0x8c, 0x2f, 0x21)

BODY_PT = 9.5
SERIF = "Times New Roman"
SANS = "Arial"

doc = Document()

# ── page setup ───────────────────────────────────────────────────────────
s = doc.sections[0]
s.page_width, s.page_height = Inches(8.5), Inches(11)
for attr, val in (("top_margin", 0.75), ("bottom_margin", 1.0),
                  ("left_margin", 0.62), ("right_margin", 0.62)):
    setattr(s, attr, Inches(val))

st = doc.styles["Normal"]
st.font.name = SERIF
st.font.size = Pt(BODY_PT)
st.paragraph_format.space_after = Pt(0)
st.paragraph_format.line_spacing = 1.0
st._element.rPr.rFonts.set(qn("w:eastAsia"), SERIF)


def set_cols(section, n, space_tw=340):
    sectPr = section._sectPr
    # A sectPr with no <w:type> defaults to nextPage, which starts the FOLLOWING
    # section on a fresh page. The title section must flow straight into the
    # two-column body on the same page, so the type is stated explicitly rather
    # than left to the default. Word tolerates the omission; stricter importers
    # (Pages, QuickLook) do not.
    t = sectPr.find(qn("w:type"))
    if t is None:
        t = OxmlElement("w:type")
        sectPr.insert(0, t)
    t.set(qn("w:val"), "continuous")
    cols = sectPr.xpath("./w:cols")
    c = cols[0] if cols else OxmlElement("w:cols")
    if not cols:
        sectPr.append(c)
    c.set(qn("w:num"), str(n))
    c.set(qn("w:space"), str(space_tw))
    c.set(qn("w:equalWidth"), "1")


def new_section(n_cols):
    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.top_margin, sec.bottom_margin = Inches(0.75), Inches(1.0)
    sec.left_margin, sec.right_margin = Inches(0.62), Inches(0.62)
    set_cols(sec, n_cols)
    return sec


def para(text="", *, size=BODY_PT, bold=False, italic=False, align=None,
         font=SERIF, color=INK, before=0, after=0, indent=None, spacing=1.0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = spacing
    if indent is not None:
        pf.first_line_indent = Inches(indent)
    if align is not None:
        p.alignment = align
    if text:
        r = p.add_run(text)
        r.font.name, r.font.size = font, Pt(size)
        r.bold, r.italic = bold, italic
        r.font.color.rgb = color
    return p


def rich(p, parts, size=BODY_PT, font=SERIF, color=INK):
    """parts: list of (text, style) where style in {'', 'b','i','bi','m','a','x'}"""
    for text, sty in parts:
        r = p.add_run(text)
        r.font.size = Pt(size)
        r.bold = "b" in sty
        r.italic = "i" in sty
        if "m" in sty:
            r.font.name = "Consolas"
            r.font.size = Pt(size - 0.8)
            r.font.color.rgb = ACC
        else:
            r.font.name = font
            r.font.color.rgb = OXI if "x" in sty else (ACC if "a" in sty else color)
    return p


def body(parts, indent=0.16, after=1.5, justify=True):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(after)
    pf.first_line_indent = Inches(indent)
    pf.line_spacing = 1.0
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return rich(p, parts if isinstance(parts, list) else [(parts, "")])


def h1(num, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    # keep_with_next stops a heading being orphaned at a column/page foot with
    # its section starting in the next column — the "System Architecture in one
    # column, its figure in the other" defect.
    p.paragraph_format.keep_with_next = True
    r = p.add_run(f"{num}.  {text.upper()}")
    r.font.name, r.font.size, r.bold = SERIF, Pt(9.5), False
    r.font.color.rgb = INK
    # small-caps look
    rPr = r._element.get_or_add_rPr()
    sc = OxmlElement("w:smallCaps"); sc.set(qn("w:val"), "1"); rPr.append(sc)
    return p


def h2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name, r.font.size, r.italic = SERIF, Pt(9.5), True
    r.font.color.rgb = INK
    return p


def figure(path, caption_parts, width_in, full=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(2)
    # A picture paragraph can never split internally regardless of keep_together,
    # so that flag is a no-op here — kept for clarity, not effect. Deliberately NOT
    # linking to the caption with keep_with_next: for a large full-width figure,
    # forcing image+caption to move as one unit means that whenever the combined
    # block doesn't fit the space left on the page, BOTH get pushed to the next
    # page and the leftover space renders blank. Letting the image alone use
    # whatever room remains — worst case the caption starts the next page under
    # nothing — wastes far less space than a blank half-page.
    p.paragraph_format.keep_together = True
    p.add_run().add_picture(str(path), width=Inches(width_in))
    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if full else WD_ALIGN_PARAGRAPH.LEFT
    cp.paragraph_format.space_after = Pt(8)
    cp.paragraph_format.keep_together = True
    rich(cp, caption_parts, size=8.0)
    return p


def table_caption(parts, before=8):
    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp.paragraph_format.space_before = Pt(before)
    cp.paragraph_format.space_after = Pt(3)
    cp.paragraph_format.keep_with_next = True
    rich(cp, parts, size=8.0)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), hexcolor)
    tcPr.append(sh)


def set_borders(tbl, top=True, bottom=True, header_rule=True):
    tblPr = tbl._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        if edge in ("top", "bottom"):
            el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "12")
        elif edge == "insideH":
            el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "2")
        else:
            el.set(qn("w:val"), "none"); el.set(qn("w:sz"), "0")
        el.set(qn("w:color"), "16181D" if edge in ("top", "bottom") else "C9CDD6")
        borders.append(el)
    tblPr.append(borders)


def make_table(headers, rows, widths=None, size=7.6, hl_row=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_borders(t)
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.paragraph_format.space_after = Pt(1.5)
        p.paragraph_format.space_before = Pt(1.5)
        r = p.add_run(h)
        r.font.name, r.font.size, r.bold = SANS, Pt(size - 0.4), True
        r.font.color.rgb = INK
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(1.2)
            p.paragraph_format.space_before = Pt(1.2)
            mono = isinstance(v, str) and v.startswith("")
            txt = v[1:] if mono else v
            r = p.add_run(str(txt))
            r.font.name = "Consolas" if mono else SERIF
            r.font.size = Pt(size - (0.5 if mono else 0))
            r.font.color.rgb = INK
            if hl_row is not None and ri == hl_row:
                r.bold = True
        if hl_row is not None and ri == hl_row:
            for c in cells:
                shade(c, "EEF1F7")
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)
    # No row may split across a column or page boundary. python-docx exposes no
    # setter for this, and assigning `row.allow_break_across_pages` merely
    # creates an unused Python attribute — it must be written as w:cantSplit on
    # the row properties or the table still breaks mid-row.
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        if not trPr.findall(qn("w:cantSplit")):
            trPr.append(OxmlElement("w:cantSplit"))
        for cell in row.cells:
            for par in cell.paragraphs:
                par.paragraph_format.keep_together = True
    return t


# ══════════════════════════════════════════════════════════════════════════
# FRONT MATTER — single column
# ══════════════════════════════════════════════════════════════════════════
set_cols(s, 1)

para("2026 IEEE International Conference on Computer Vision and Intelligent Systems (ICCVIS)",
     size=7.6, align=WD_ALIGN_PARAGRAPH.CENTER, font=SANS, color=MUT)
para("Track — Video Analytics, Surveillance and Privacy-Preserving Vision",
     size=7.6, align=WD_ALIGN_PARAGRAPH.CENTER, font=SANS, color=MUT, after=15)

para("SmartDetect: Face-Anchored Identity Arbitration for\nAuditable Cross-Camera Person Re-Identification",
     size=19, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=14)

# ── Author block: IEEE convention — name / department / institution / city ──
DEPT = "Dept. of Information Technology (B.Tech)"
INST = "Sri Krishna College of Technology"
CITY = "Coimbatore, Tamil Nadu, India"
AUTHORS = ["Krishiv Kameshwar B. S.", "Kishan S. G.", "Lokeshkumar D."]


def author_cell(cell, name, role=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(name)
    r.font.name, r.font.size = SERIF, Pt(11)
    r.font.color.rgb = INK
    lines = [role] if role else []
    lines += [DEPT, INST, CITY]
    for i, ln in enumerate(lines):
        q = cell.add_paragraph()
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        q.paragraph_format.space_after = Pt(0)
        q.paragraph_format.space_before = Pt(1 if i == 0 else 0)
        q.paragraph_format.line_spacing = 1.0
        rr = q.add_run(ln)
        rr.font.name, rr.font.size = SERIF, Pt(8.6)
        rr.italic = True
        rr.font.color.rgb = MUT


def no_borders(tbl):
    """Author blocks are a layout grid, not a data table — strip all rules."""
    tblPr = tbl._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "none")
        el.set(qn("w:sz"), "0")
        b.append(el)
    tblPr.append(b)


t = doc.add_table(rows=1, cols=3)
t.alignment = WD_TABLE_ALIGNMENT.CENTER
t.autofit = False
no_borders(t)
for i, n in enumerate(AUTHORS):
    author_cell(t.rows[0].cells[i], n)
for cell in t.rows[0].cells:
    cell.width = Inches(2.42)

# Supervisor, centred on its own row beneath the students
t2 = doc.add_table(rows=1, cols=1)
t2.alignment = WD_TABLE_ALIGNMENT.CENTER
t2.autofit = False
no_borders(t2)
author_cell(t2.rows[0].cells[0], "Ms. K. Sinduja", role="Project Supervisor")
t2.rows[0].cells[0].width = Inches(3.4)
t2.rows[0].cells[0].paragraphs[0].paragraph_format.space_before = Pt(9)

para(after=8)

# ── Abstract + Index Terms: IEEE conference form ─────────────────────────
# These belong INSIDE the two-column flow (column one), not as a full-width
# slab — so the section break to two columns happens first.
new_section(2)

CW = 3.30      # column width for figures
FW = 5.70      # full width. Trimmed from 6.35: the two full=True figures
               # (operation, collapse) are tall enough that a keep-together
               # image+caption block often doesn't fit the space left on a
               # partially-filled page, so the whole block jumps to the next
               # page and leaves that leftover space blank. Smaller figures
               # need less leftover room to fit, which is the only lever
               # available here — this script has no page-layout preview to
               # tune against directly.

ABSTRACT = (
    "Person re-identification is overwhelmingly evaluated as ranked retrieval against a closed, "
    "fixed gallery. A deployed surveillance system instead decides autonomously whether the person "
    "now in frame is already enrolled or new, and every such decision permanently mutates the "
    "gallery that all later decisions depend on. We show that this open-set, self-enrolling regime "
    "admits a failure mode retrieval metrics cannot express: identity collapse, in which "
    "appearance-based association silently fuses several people under one identifier, leaving a "
    "merged evidence trail indistinguishable from a correct one. We present SmartDetect, a "
    "multi-camera system built on face-anchored identity arbitration — a strict evidence hierarchy "
    "in which a quality-gated face embedding is the only signal permitted to create or veto an "
    "identity, while clothing colour and body re-identification may re-associate a recently seen "
    "person but never mint, steal or override one. Three mechanisms harden the pipeline: a face veto "
    "on non-facial matches, a tracker ID-switch guard, and an evidence gate separating the on-screen "
    "label from what is durably recorded. We further contribute an evaluation methodology for this "
    "regime, in which precision and coverage are reported separately and never blended. On the "
    "ChokePoint portal corpus, a four-configuration ablation reproduces collapse and then removes "
    "it: the unhardened pipeline mints seven identifiers for twenty-five people at zero identity "
    "purity and 15.5% precision while labelling 99.8% of person-frames, whereas the full pipeline "
    "reaches 99.7% precision, 93.2% purity and 100.0% evidence precision at 98.6% coverage, for "
    "6.8 ms per frame of additional CPU cost."
)

ap = doc.add_paragraph()
ap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
ap.paragraph_format.space_after = Pt(5)
ap.paragraph_format.line_spacing = 1.0
rich(ap, [("Abstract\u2014", "bi"), (ABSTRACT, "b")], size=9)

kp = doc.add_paragraph()
kp.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
kp.paragraph_format.space_after = Pt(2)
kp.paragraph_format.line_spacing = 1.0
rich(kp, [("Index Terms\u2014", "bi"),
          ("Person re-identification, open-set recognition, face recognition, multi-camera "
           "tracking, identity management, evaluation methodology, biometric privacy, data "
           "erasure, edge inference.", "b")], size=9)

h1("I", "Introduction")
body("A decade of progress in person re-identification has been measured almost entirely on a "
     "single task shape. A probe image is presented; a gallery of labelled identities is ranked by "
     "learned similarity; Rank-1, Rank-5 and mean average precision (mAP) summarise how often the "
     "correct gallery entry surfaces near the top. Benchmarks such as Market-1501, DukeMTMC-reID, "
     "CUHK03 and MSMT17 institutionalised this protocol, and the comprehensive survey of Wang et al. "
     "[1] catalogues more than three hundred papers organised around improving retrieval accuracy "
     "under it.", indent=0)
body([("The protocol embeds three assumptions that a deployed system cannot make. First, it is ", ""),
      ("closed-set", "b"), (": the correct answer is assumed present in the gallery, so the question "
       "is only ", ""), ("which", "i"), (" entry, never ", ""), ("whether any", "i"),
      (". Second, it is ", ""), ("stateless", "b"), (": scoring one probe does not alter the gallery, "
       "so errors are independent. Third, it is ", ""), ("read-only", "b"),
      (": retrieval produces a ranking for a human to inspect, not a durable record that downstream "
       "processes will treat as fact.", "")])
body("An autonomous surveillance deployment violates all three simultaneously. People who were "
     "never enrolled walk into frame constantly, so the system must be able to answer nobody. When it "
     "answers nobody, it enrols — writing a new gallery entry that every subsequent comparison will "
     "use. And what it writes is not a ranking but an evidence trail: timestamped sightings, camera "
     "identifiers, cropped photographs. That trail is what an operator, and potentially an "
     "investigator, will later read as a record of where a specific person was.")
body([("Under these conditions a distinctive and, we argue, under-studied failure emerges. Suppose "
       "the system observes a person whose face is not visible — turned away, occluded, or simply too "
       "distant for the detector. Appearance cues remain: the dominant colour of their upper garment, "
       "a body-shape descriptor from a re-ID backbone. If those cues are permitted to assert identity, "
       "and if a second person happens to wear a similar jacket, the system may associate the second "
       "person with the first person's identifier. Nothing visibly breaks. The label on screen is "
       "confident. Evidence is filed. From that moment the stored template for that identifier begins "
       "to drift toward an average of two people, which makes further incorrect matches ", ""),
      ("more", "i"), (" likely, not less. We call this ", ""), ("identity collapse", "b"), (".", "")])
body([("Collapse is not a hypothetical. During development we observed three distinct individuals "
       "converge onto a single identifier on real footage. In the controlled ablation reported in "
       "Section VI, the unhardened configuration minted seven identifiers to cover twenty-five people, "
       "and ", ""), ("not one of the seven referred to a single person", "b"),
      (": the worst absorbed twenty-two of the twenty-five. That configuration nonetheless assigned an "
       "identifier to 99.8% of observed person-frames — it was highly confident, almost always "
       "answering, and wrong roughly three times in four.", "")])

p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(4)
p.paragraph_format.space_after = Pt(4); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.left_indent = Inches(0.10)
rich(p, [("The metric blind spot.  ", "bx"),
         ("Identity collapse is invisible to Rank-n and mAP. Those metrics presuppose a fixed "
          "ground-truth gallery; collapse corrupts the gallery itself. A system that has merged three "
          "people can still return its (single, wrong) entry at Rank-1 with high confidence for all "
          "three, and score well.", "i")], size=8.8)

body([("The corrective intuition is simple to state and surprisingly consequential to enforce: ", ""),
      ("not all evidence is equal, and weaker evidence must never be allowed to overrule stronger "
       "evidence or to act in its absence beyond a narrow, bounded role.", "b"),
      (" A face embedding from a sufficiently large, sufficiently confident detection is strong "
       "evidence of ", ""), ("who", "i"), (" someone is. The colour of a jacket is not evidence of "
       "identity at all; it is at best evidence of ", ""), ("continuity", "i"),
      (" — that the person now in frame is plausibly the person seen thirty seconds ago in the same "
       "corridor.", "")])
body("SmartDetect operationalises that hierarchy. This paper reports its architecture, the four "
     "hardening mechanisms that enforce the hierarchy, an evaluation methodology designed "
     "specifically to make collapse measurable, and a governance layer whose erasure path is verified "
     "end to end rather than asserted.")

h2("A.  Contributions")
for lead, txt in [
    ("1) Face-anchored arbitration.", " A concrete evidence hierarchy in which a quality-gated face is "
     "necessary to create an identity and sufficient to veto a non-facial one, while colour and body "
     "re-ID are confined to time-bounded re-association of already-enrolled people. We give the exact "
     "gating predicates and the parameter values in use (Section IV)."),
    ("2) Three hardening mechanisms beyond the hierarchy itself.", " A face veto that rejects "
     "appearance matches contradicted by a visible face; an ID-switch guard that detects when a "
     "multi-object tracker hands a track from one person to another during occlusion; and an evidence "
     "gate that separates the transient display label from the durable database write, so a track "
     "whose identity is momentarily in doubt cannot file photographic evidence under someone else's "
     "identifier (Sections IV-D to IV-F)."),
    ("3) An evaluation methodology for open-set, self-enrolling identity.", " A three-way outcome "
     "partition (CORRECT / CONTAMINATED / UNASSIGNED) that is exhaustive by construction and asserted "
     "to sum at runtime; separate precision and coverage that we argue must never be combined into an "
     "F-score; a duplicate identities metric distinguishing over-splitting from contamination; and an "
     "evidence precision metric scoring what was actually written to disk rather than what was "
     "momentarily displayed (Section V)."),
    ("4) A dataset-adequacy gate.", " A refusal mechanism that blocks the ablation from producing a "
     "results table when the corpus cannot support one — below 15 identities, 1000 labelled "
     "person-frames, or 10 cross-camera transitions (Section V-E)."),
    ("5) Verified erasure.", " An audited deletion path covering the database row and its embeddings, "
     "all sighting rows, all photographic evidence on disk, and — a case we found unhandled and "
     "repaired — identity caches held in memory by running camera threads, with a receipt written "
     "only after the system re-queries and re-stats to confirm (Section VIII)."),
]:
    body([(lead, "b"), (txt, "")])

# ── Fig. 1 full width: system in operation ───────────────────────────────
new_section(1)
figure(FIGS / "fig_operation.png",
       [("Fig. 1.  ", "b"),
        ("SmartDetect operating on ChokePoint P1E_S1_C1 under the full pipeline (configuration D). "
         "Green boxes are person tracks bearing an identifier; grey boxes are tracked but "
         "unidentified. Solid indigo face boxes cleared the 48 px / 0.60 quality gate; dashed oxide "
         "boxes were detected but refused. Every box, identifier and gate verdict shown is the "
         "system's own output for that frame, produced by the same production code path the "
         "evaluation harness drives — nothing is illustrative.", "")],
       FW, full=True)
new_section(2)

h1("II", "Related Work")
h2("A.  Retrieval-Oriented Re-Identification")
body("The survey of Wang et al. [1] organises the field by research purpose into local and global "
     "information extraction, metric learning, post-processing, efficiency, labelling-cost reduction "
     "and data-type extension. Its account of evaluation is instructive precisely because of what it "
     "takes for granted: Rank-n, the Cumulative Match Characteristic curve, ROC/AUC, and mAP are "
     "presented as the measurement vocabulary of the field, each defined over a query and a gallery.",
     indent=0)
body("Within that framing, the accuracy trajectory has been remarkable. Attention-based methods "
     "dominate the reported state of the art, with Rank-1 above 90% on several standard benchmarks "
     "and Rank-20 approaching saturation on the older ones [1]. Part-based decomposition [2], "
     "attention mechanisms [3], and carefully engineered metric-learning objectives — contrastive, "
     "triplet, quadruplet and hard-mining variants [4] — account for most of that gain.")
body("Two observations from the survey bear directly on our work. First, it notes explicitly that "
     "accuracy on hand-annotated datasets exceeds accuracy on detector-generated ones, because "
     "automatic detection introduces cropping error and misalignment absent from manual boxes [1]. A "
     "deployed system has only detector boxes. Second, among the challenges it enumerates — cropping "
     "error, low resolution, occlusion, illumination variation, visually similar identities, and one "
     "identity appearing in different clothing — the last two are exactly the conditions under which "
     "appearance-based association becomes dangerous rather than merely inaccurate.")

h2("B.  Open-Set and Real-World Formulations")
body("Open-set re-ID, in which the probe may have no gallery counterpart, is recognised in the "
     "literature as a distinct and harder problem [1], and adversarial-example work has observed that "
     "its open-set character demands different treatment from closed-set recognition [5]. "
     "Verification-style formulations, which ask whether two observations are the same person rather "
     "than ranking a gallery, are closer in spirit to the decision a deployed system makes.",
     indent=0)
body([("What remains comparatively unaddressed is the ", ""), ("cumulative", "i"),
      (" consequence of open-set decisions in a system that enrols. The gallery in a deployment is "
       "not a fixed artefact provided by a benchmark; it is the accumulated output of the system's "
       "own prior decisions. An incorrect enrolment or an incorrect merge is not a single scored "
       "error — it is a permanent corruption of the reference against which all future comparisons "
       "are made. Our contribution sits here: not a better similarity function, but a decision "
       "procedure and a measurement framework for the regime where decisions compound.", "")])

h2("C.  Multi-Camera Association and Spatio-Temporal Constraints")
body("Multi-target multi-camera tracking [1] extends re-ID with the constraint that cameras occupy "
     "known relative positions and that human movement between them takes plausible time. "
     "Camera-topology and network-consistency methods exploit this: if a person is at camera A at "
     "time t, the set of cameras where they can appear at t + Δ is constrained [1]. The survey is "
     "appropriately cautious, noting that such constraints are not absolute, that human speed varies "
     "widely, and that they may not transfer to distant re-identification.", indent=0)
body([("SmartDetect currently uses temporal constraints only in a weak, conservative form: recency "
       "windows that ", ""), ("restrict", "i"),
      (" when appearance-based re-association may fire (10 minutes for colour, 12 hours for body "
       "re-ID). We deliberately do not use topology to ", ""), ("assert", "i"),
      (" identity, for the reason developed throughout this paper — a spatio-temporal prior is not "
       "evidence of who someone is, and admitting it as such reopens the collapse pathway from a "
       "different direction.", "")])

h2("D.  Efficiency and Deployment")
body("Lightweight architectures, notably OSNet [6], and model-compression work address the fact that "
     "retrieval latency, not feature extraction, dominates query cost at scale [1]. Our concern is "
     "complementary: end-to-end per-frame latency of a full pipeline (detection, tracking, face "
     "analysis, arbitration, persistence) running on commodity hardware without a discrete GPU, which "
     "determines whether a deployment is feasible at all in the settings — small campuses, single "
     "buildings — where these systems are most often proposed.", indent=0)

h2("E.  Biometric Governance")
body("The survey observes in passing that privacy concerns have hindered dataset access [1]. For a "
     "deployed system the issue is not access but obligation. A face embedding is biometric personal "
     "data; it cannot be reissued after a breach; and under India's Digital Personal Data Protection "
     "Act 2023, the EU GDPR, and statutes such as the Illinois Biometric Information Privacy Act it "
     "carries specific duties around lawful basis, storage limitation and erasure. We treat the "
     "mechanisms that discharge those duties as system components subject to test, not as "
     "documentation (Section VIII).", indent=0)

h1("III", "System Architecture")

figure(FIGS / "fig_pipeline.png",
       [("Fig. 2.  ", "b"),
        ("SmartDetect per-frame decision path. Stages 5-8, marked with the heavier rule, are those "
         "where identity may be created or invalidated; all other stages may only propose or "
         "annotate. Every arbitration decision is delegated to production code, which the offline "
         "evaluation harness invokes directly rather than reimplementing.", "")], CW)

h2("A.  Design Constraints")
body("SmartDetect was built under four constraints that shaped every subsequent decision.", indent=0)
for lead, txt in [
    ("CPU-only inference.", " The target deployment is a single commodity machine without a discrete "
     "GPU. This rules out per-frame execution of every model and forces the once-per-track "
     "arbitration design."),
    ("Heterogeneous sources.", " Live USB cameras, RTSP streams and uploaded video files are handled "
     "through one abstraction. A file source is modelled as a camera whose stream terminates, which "
     "keeps a single code path for what would otherwise be two."),
    ("Auditability.", " Every biometric access is logged with actor, subject, time, source address "
     "and outcome. This is a legal requirement in the jurisdictions of interest and, pragmatically, "
     "the only way to answer a subject's question about who accessed their data."),
    ("Falsifiability.", " The system's accuracy claims must be checkable against dataset ground truth "
     "by a third party, using the same code path the deployment runs. This constraint produced the "
     "evaluation harness of Section V and, indirectly, most of this paper."),
]:
    body([(lead, "b"), (txt, "")])

h2("B.  Threading Model")
body("Each camera owns two threads. The capture thread reads frames at source rate, pushes the newest "
     "frame into a depth-1 queue (dropping any unconsumed predecessor), draws annotations from the "
     "last completed analysis result, and encodes for streaming. The analysis thread runs the full "
     "pipeline at whatever rate it can sustain — typically 1–3 Hz on CPU.", indent=0)
body("The consequence is that display frame rate is independent of analysis cost. The cost is that "
     "annotations lag the moving subject when analysis is slow; on 4K input at roughly 5 s per frame "
     "this lag is visible and is a known limitation (Section IX).")

h2("C.  Detection and Tracking Stack")
body([("Person detection uses YOLOv8 with ", ""), ("resolution-aware input sizing", "b"),
      (". A naive deployment fixes the network input size once, which is adequate for a single source "
       "type and quietly destructive across several. A 4K frame squeezed into the 416 px input "
       "appropriate for a webcam loses every distant pedestrian: in our measurements on a "
       "high-resolution clip the detector returned two people where five were present. The system "
       "therefore selects input size from source resolution — 416 px for webcam-class sources, 640 px "
       "for HD, 960 px for 4K — and applies the same principle to the face-detection pass, whose scan "
       "resolution follows the source rather than a fixed downscale. Sources at or below 1280 px are "
       "scanned at native size, because shrinking a 768 px source to a nominal 640×480 pushed "
       "borderline faces below the detector's floor and cost measurable recall.", "")], indent=0)
body([("Tracking uses ByteTrack, which associates every detection box rather than only "
       "high-confidence ones, giving stable identifiers through brief low-confidence intervals. The "
       "track activation threshold is set below the detection confidence floor so that tracks start "
       "on the first analysis cycle: because analysis runs at 1–3 Hz rather than frame rate, waiting "
       "several cycles to confirm a track means waiting seconds, during which a walking subject may "
       "leave the scene.", "")])
body("ByteTrack replaced an earlier appearance-based tracker specifically because arbitration cost "
     "dominates. Resolving identity once per track rather than once per detection is what makes "
     "CPU-only operation viable; the ID-switch guard of Section IV-E exists precisely to recover the "
     "safety that this caching optimisation gives up.")

h2("D.  Face Analysis")
body("Face detection and embedding use InsightFace's ArcFace pipeline [8] at a 640×640 detector size. "
     "The library default was retained after measuring that halving it to 320 halved detection recall "
     "on sub-50 px faces — the exact population that the quality gate then filters, so undersizing "
     "here would remove faces before the gate could reason about them.", indent=0)
body("One detector pass runs per analysis cycle over the whole frame, and detected faces are "
     "associated to person boxes by testing whether the face centroid lies within the upper 60% of "
     "the box. This is deliberately cheaper and more robust than running a face detector inside each "
     "person crop: it costs one pass regardless of crowd size, and it guarantees that when boxes "
     "overlap, each face is assigned to exactly one person rather than being found independently by "
     "two.")
body([("A second, optional pass digitally zooms head regions of person boxes where no face was found, "
       "upscaling toward roughly 384 px and re-scanning. This is purely a recall booster — it changes "
       "which faces are ", ""), ("found", "i"), (", never how identity is ", ""), ("decided", "i"),
      (" — and it is the one production stage the offline harness does not exercise, which is why we "
       "describe offline coverage as a lower bound.", "")])

h2("E.  Identity Representation")
body("Each enrolled person is one database row bearing a system-generated code of the form SDT-XXXX. "
     "The row holds a primary 512-d ArcFace embedding; a multi-view gallery of up to five additional "
     "embeddings capturing different head poses; a body re-ID vector; a dominant upper-garment colour "
     "in HSV; a body-height ratio; first- and last-seen timestamps; and the governance fields of "
     "Section VIII.", indent=0)
body("The multi-view gallery exists because a single template generalises poorly across pose. "
     "Maintenance is deliberately conservative: a new view is added only when its similarity to the "
     "matched identity falls in [0.60, 0.85] — high enough to be confidently the same person, low "
     "enough to add information the gallery does not already contain. The primary template is blended "
     "toward a new observation only on matches at or above 0.55 similarity, with the new observation "
     "weighted 0.20. A weak match that is nonetheless accepted as an identification does not modify "
     "the template, because doing so would drag it toward whichever look-alike produced the weak "
     "match.")

h1("IV", "Face-Anchored Identity Arbitration")
body("This section states the core contribution precisely.", indent=0)

h2("A.  The Evidence Hierarchy")
body("Let b be a tracked person box in the current frame. Arbitration considers three signals: f, a "
     "512-d ArcFace embedding, present only if a detected face was associated to b and cleared the "
     "quality gate; c, the dominant HSV colour of the torso region, computed by k-means over the "
     "upper 45% of b; and r, a 512-d OSNet body descriptor over the full box. The hierarchy is:",
     indent=0)

for lead, txt in [
    ("H1.", " Only f may create an identity. No combination of c and r can enrol a new person, in any "
     "configuration of the system."),
    ("H2.", " A gate-passing f that matches no enrolled identity is decisive: the person is a "
     "stranger, and c and r are not consulted at all for that box."),
    ("H3.", " c and r may only re-associate an already-enrolled person seen within a bounded recency "
     "window, and any such match is rejected if f is present and contradicts it."),
    ("H4.", " Durable evidence requires f to confirm the identifier in the same cycle in which the "
     "evidence is written."),
]:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Inches(0.16)
    p.paragraph_format.space_after = Pt(1.5)
    rich(p, [(lead, "ba"), (txt, "")])

body([("H2 is the mechanism we found most consequential and least obvious. The intuitive design lets "
       "fallbacks run whenever the primary method fails to produce a match. But \"the face matched "
       "nobody\" is not a failure of the face method — it is a ", ""), ("successful", "i"),
      (" and highly informative result: this person is not enrolled. Treating it as failure and "
       "falling through to clothing colour is precisely the pathway by which a stranger acquires an "
       "existing person's identifier.", "")])

h2("B.  The Face Quality Gate")
body("A face contributes to identity only if its bounding-box height in the original frame is at "
     "least 48 px and the detector's confidence is at least 0.60. Faces failing either test are still "
     "drawn on screen — the operator sees that a face was found — but are excluded from arbitration. "
     "Fig. 1(a) shows this directly: a foreground face measured at 137 px is enrolled, while a "
     "background face at 42 px in the same frame is detected and refused.", indent=0)
body([("The threshold is empirical. Below roughly 48 px the ArcFace embedding becomes dominated by "
       "interpolation artefacts, and cosine similarities between different people rise into the range "
       "where they are indistinguishable from genuine matches. Admitting such embeddings does not "
       "merely add noise; it adds ", ""), ("systematically misleading", "i"),
      (" evidence in exactly the direction that causes merges.", "")])
body([("The gate has a visible and deliberate cost. In wide-area or overhead footage, most faces never "
       "reach 48 px, and those people remain permanently unidentified. Coverage on such footage is low ",
       ""), ("by design", "b"), (", and we regard reporting this honestly as a requirement rather than "
       "a caveat.", "")])

h2("C.  Matching and Fallback")
body("Given f, the system searches enrolled identities for maximum cosine similarity across each "
     "identity's primary template and multi-view gallery, requiring at least 0.56 to match. The "
     "threshold history is itself evidence for the argument of this paper: at 0.35 it cross-matched "
     "strangers; at 0.50 it still merged two distinct pairs of people on a four-video evaluation set; "
     "0.56 is the value at which those specific merges ceased.", indent=0)
body("If no identity clears 0.56 and f is present, invariant H2 applies and arbitration terminates "
     "with a stranger verdict, proceeding to enrolment. Only when f is absent do the fallbacks run.")
body("Colour re-association requires HSV distance below 30.0 and restricts candidates to identities "
     "seen in the last 10 minutes. Body re-ID requires OSNet cosine similarity of at least 0.68 "
     "within a 12-hour window; at 0.60 it cross-matched different people wearing similar clothing. "
     "Both windows encode the same reasoning: clothing is a same-day, same-session cue, and treating "
     "it as a persistent identity signal is a category error.")
body("Both fallbacks additionally exclude identifiers already claimed by another person visible in "
     "the same frame — two simultaneously visible people cannot be the same person. Fig. 1(b) shows "
     "the result on two subjects present together.")

h2("D.  The Face Veto")
body([("A subtlety remains. Suppose f exists but is ", ""), ("weak", "i"),
      (" — it cleared the quality gate yet matched nobody above 0.56. Under H2 the fallbacks are "
       "blocked entirely when the face anchor is enabled. But consider the configuration where the "
       "anchor is disabled, or a borderline case: colour proposes identity p, and we hold an "
       "embedding f for the person in front of us. We can directly test the proposal.", "")], indent=0)
body([("The face veto computes similarity between f and p's stored template. Below 0.45, the colour or "
       "re-ID match is rejected outright. This is a different test from the 0.56 match threshold: it "
       "does not ask whether the face identifies p, but whether the face ", ""), ("refutes", "i"),
      (" p. The asymmetry is intentional — the evidential burden for rejecting a weak claim is "
       "properly lower than for establishing a strong one.", "")])

h2("E.  The ID-Switch Guard")
body("Multi-object trackers maintain identity through motion and box overlap, without appearance "
     "modelling. When two people cross or occlude one another, ByteTrack can transfer a track "
     "identifier from one to the other. Because arbitration is cached per track for efficiency, such "
     "a hand-off would silently transfer an identity — and, worse, would appear entirely stable, "
     "since the cached identifier is simply reused every frame.", indent=0)

ALGO = [
    "input  t: track id; cached: identity or NULL;",
    "       f: gated face embedding or NULL",
    "state  strikes[t] <- 0",
    "",
    "if guard disabled or cached = NULL or f = NULL then",
    "    return cached      // no face evidence to test",
    "",
    "s <- cos(f, template(cached))",
    "if s < 0.35 then",
    "    strikes[t] <- strikes[t] + 1",
    "    if strikes[t] >= 2 then",
    "        drop cache for t;  strikes[t] <- 0",
    "        return NULL    // force re-identification",
    "else",
    "    strikes[t] <- 0    // consecutive, not cumulative",
    "return cached",
]
p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(5)
p.paragraph_format.space_after = Pt(1)
r = p.add_run("Algorithm 1  ID-switch guard")
r.font.name, r.font.size, r.bold = SANS, Pt(8), True
for ln in ALGO:
    lp = doc.add_paragraph()
    lp.paragraph_format.space_after = Pt(0)
    lp.paragraph_format.line_spacing = 1.0
    lp.paragraph_format.left_indent = Inches(0.06)
    lr = lp.add_run(ln if ln else " ")
    lr.font.name, lr.font.size = "Consolas", Pt(7.2)
    lr.font.color.rgb = INK
para(after=5)

body([("Two design points matter. Strikes must be ", ""), ("consecutive", "i"),
      (": a single contradiction is far more likely to be a bad frame — motion blur, extreme angle, "
       "partial occlusion — than a genuine hand-off, and resetting on any agreement prevents slow "
       "accumulation of unrelated bad frames from triggering a spurious reset. And the guard fires "
       "only when face evidence is available; absent a gated face there is nothing to contradict, and "
       "the cache is trusted.", "")])
body([("A consequence worth stating: the guard ", ""), ("increases", "i"),
      (" the raw count of identifier switches, because correcting a hijacked track is itself a "
       "switch. A configuration with more switches may therefore be more correct. This is one of "
       "several reasons the metrics of Section V must be read jointly, and it is confirmed "
       "empirically in Section VI-F.", "")])

h2("F.  The Evidence Gate")
body([("The final mechanism separates two things that are conventionally conflated: what the system ",
       ""), ("displays", "i"), (" and what it ", ""), ("records", "i"), (".", "")], indent=0)
body("A track holding a cached identifier with no confirming face this cycle is in an epistemically "
     "weaker state than one whose identifier was just earned from a visible face. Displaying the "
     "cached label is reasonable — it is the system's best current estimate, and an operator watching "
     "a live feed benefits from continuity. Writing a timestamped sighting row and a cropped "
     "photograph under that identifier is a different act: it creates a durable record that a person "
     "was at a place at a time. Fig. 1(d) shows precisely this state: a track retains its identifier "
     "on screen while no face clears the gate, and under configuration D writes no evidence.")
body("The evidence gate requires, for any database write, that a gate-passing face agree with the "
     "identifier in that cycle (or that the identifier was freshly earned from that face). "
     "Additionally, a track carrying unresolved ID-switch strikes is barred from writing evidence at "
     "all, since it is by definition mid-doubt.")
body("This closes a window that the ID-switch guard alone leaves open. The guard needs two cycles to "
     "fire; during those two cycles a hijacked track under configuration C would file one or two "
     "sighting rows and photographs under the wrong person's identifier. The guard corrects the label "
     "going forward but cannot retract evidence already written. Only the evidence gate prevents the "
     "write in the first place.")

h2("G.  The Registration Pose Gate")
body([("Enrolment is where the hierarchy's cost concentrates. An identity minted from a profile or "
       "bowed-head view embeds a viewpoint that matches poorly against later frontal views of the "
       "same person — producing not a merge but its opposite, a ", ""), ("duplicate", "i"),
      (". We observed exactly this: one individual enrolled twice from a single continuous recording, "
       "once frontal and once with head bowed.", "")], indent=0)
body("The pose gate estimates head orientation from the five facial landmarks InsightFace already "
     "returns, so it costs no additional inference. Writing e for the inter-ocular midpoint, n for "
     "the nose and m for the mouth midpoint, with d the inter-ocular distance:")

for eq, num in [("yaw = (nₓ − eₓ) / d", "(1)"),
                ("pitch = (nᵧ − eᵧ) / (mᵧ − eᵧ)", "(2)")]:
    ep = doc.add_paragraph()
    ep.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ep.paragraph_format.space_before = Pt(3)
    ep.paragraph_format.space_after = Pt(3)
    r = ep.add_run(eq); r.font.name, r.font.size, r.italic = SERIF, Pt(9.5), True
    r2 = ep.add_run("        " + num); r2.font.name, r2.font.size = SERIF, Pt(9)

body("These are scale-invariant ratios, not calibrated angles, and we are careful not to report them "
     "as degrees. Bounds were set from a 138-sample head-pose sweep: enrolment is deferred when |yaw| "
     "> 0.35, when pitch falls outside [0.35, 0.70], or when head roll exceeds 35°.")
body([("Critically, the gate applies ", ""), ("only to enrolment, never to matching", "b"),
      (". Refusing to match a profile view would lose recall for someone already enrolled; refusing "
       "to enrol from one merely waits for a better frame on the same track, as in Fig. 1(c).", "")])
body("That reasoning has a failure case, and we handle it explicitly. On sparsely sampled input — or "
     "with a person who is simply never frontal — the better frame may never arrive, and the gate "
     "would starve enrolment entirely. We measured a 6.7 percentage-point coverage loss with no "
     "corresponding duplicate reduction on a dataset sampled at 10-second intervals. A safety valve "
     "therefore enrols from the best available frame after five consecutive deferrals on one track.")

h2("H.  What the Hierarchy Costs")
body([("The system prefers ", ""), ("duplicate identifiers over merged ones", "b"),
      (". Faced with uncertainty it splits rather than joins. This is a deliberate asymmetry grounded "
       "in the different consequences of the two errors.", "")], indent=0)
body("A duplicate is visible, locally contained and repairable: two identifiers exist for one person, "
     "an operator or the duplicate-suggestion endpoint notices, and a merge operation combines the "
     "evidence trails. A merge is invisible, globally corrupting and irreversible: two people share "
     "an identifier, their trails are interleaved beyond separation, and every subsequent comparison "
     "against the contaminated template is more likely to be wrong.")
body("Accordingly, nothing in SmartDetect merges automatically. A duplicate-suggestion endpoint "
     "surfaces likely pairs with an advisory confidence band, but the auto-merge flag is hard-coded "
     "false and no code path merges without an explicit operator action.")

h1("V", "Evaluation Methodology")
body("Standard re-ID metrics cannot express the failures described above. This section develops "
     "metrics that can. All definitions are implemented in a single scoring module that serves as "
     "their executable specification.", indent=0)

h2("A.  Unit of Measurement")
body([("The scored unit is one ", ""), ("ground-truth person-frame", "b"),
      (": one labelled person in one frame. Critically, a record exists for every ground-truth person "
       "including those the detector missed entirely, recorded as unassigned with method "
       "no_detection.", "")], indent=0)
body("This is not a detail. Omitting detector misses removes them from the denominator and inflates "
     "coverage. In our own harness that bug turned a true 80.0% coverage into a reported 92.3% — a "
     "12-point error in the system's favour, produced by an omission that is easy to make and hard to "
     "see.")

h2("B.  The Outcome Partition")
body("Let maj(c) denote the ground-truth person to whom identifier c was most often assigned — what "
     "the identifier empirically means in this run. Every record falls in exactly one bucket: "
     "UNASSIGNED if no code was issued; CORRECT if maj(c) equals the ground-truth person; "
     "CONTAMINATED otherwise.", indent=0)
body("The partition is exhaustive and mutually exclusive by construction, and both the scoring module "
     "and the sweep driver assert at runtime that counts sum to the total and fractions to exactly "
     "1.0. A run whose partition does not sum aborts rather than emitting a table — a results table "
     "that does not sum to 100% must never be published, and the cheapest way to guarantee that is to "
     "make it impossible.")
body("The unassigned bucket is further split diagnostically into detector misses and quality-gate "
     "refusals, because these are entirely different engineering problems: the first is a detection "
     "failure, the second is the system working as designed.")

h2("C.  Precision and Coverage, Reported Separately")
for eq, num in [("precision = n_correct / (n_correct + n_contaminated)", "(3)"),
                ("coverage = (n_correct + n_contaminated) / n_total", "(4)")]:
    ep = doc.add_paragraph()
    ep.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ep.paragraph_format.space_before = Pt(3); ep.paragraph_format.space_after = Pt(3)
    r = ep.add_run(eq); r.font.name, r.font.size, r.italic = SERIF, Pt(9.5), True
    r2 = ep.add_run("        " + num); r2.font.name, r2.font.size = SERIF, Pt(9)

p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.left_indent = Inches(0.10)
p.paragraph_format.space_before = Pt(3); p.paragraph_format.space_after = Pt(4)
rich(p, [("Never combine these.  ", "bx"),
         ("We report no F-score or blended figure of merit. The entire architecture is a deliberate "
          "trade of coverage for precision: each mechanism buys correctness by declining to answer. A "
          "blended score hides exactly the axis under study and lets a system that guesses more appear "
          "equal to one that guesses better. Coverage is not \"higher is better\" in isolation — low "
          "coverage on distant footage is the quality gate working.", "i")], size=8.8)

h2("D.  Metrics Specific to Self-Enrolling Identity")
for lead, txt in [
    ("Evidence precision.", " Of the sighting rows actually written to the database, the fraction "
     "sitting under the correct identifier. This is what an operator or investigator sees — the "
     "persisted trail, not the transient label — and it is the only metric in the set that can "
     "distinguish configurations C and D, since every other metric scores identifier assignment while "
     "the evidence gate governs whether evidence is written. It is reported alongside evidence yield, "
     "for the same reason precision accompanies coverage: a gate that writes almost nothing scores "
     "trivially well."),
    ("Identity purity.", " The fraction of minted identifiers referring to exactly one real person. "
     "This counts how many identifiers are polluted, not how badly — one stray frame condemns an "
     "entire identifier here while barely moving contamination, so the two are reported together."),
    ("Fragmentation.", " Mean distinct identifiers per ground-truth person; ideal 1.0. Never readable "
     "alone: a system minting a fresh identifier per frame achieves perfect precision by construction "
     "and catastrophic fragmentation, while a system placing everyone under one identifier achieves "
     "perfect fragmentation and catastrophic precision. Our own pre-hardening configuration exhibits "
     "the second pathology."),
    ("Duplicate identities.", " Fragmentation conflates two distinct phenomena. A person with 80 "
     "frames under identifier X and one misassigned frame under Y has fragmentation 2.0, but Y is not "
     "a duplicate of their identity — it is a contamination error already counted elsewhere. We "
     "therefore define ownership by majority, owned(p) = { c : maj(c) = p }, and count only genuine "
     "over-splitting, dup(p) = max(0, |owned(p)| − 1). This is the metric the pose gate targets, and "
     "separating it from fragmentation is what makes that mechanism's effect measurable at all."),
    ("Cross-camera re-association.", " For people appearing on more than one camera, the fraction "
     "whose dominant identifier on the later camera matches that on the earlier. Undefined — reported "
     "as n/a, never as zero — when no person appears on two cameras, since a single-camera corpus "
     "cannot fail this test."),
]:
    body([(lead, "b"), (txt, "")])

h2("E.  Confidence Intervals and the Adequacy Gate")
body("Every proportion carries a 95% Wilson score interval. Wilson rather than the "
     "normal-approximation interval because the latter is degenerate exactly where these metrics "
     "live: at an observed proportion of 1.0 the Wald interval has zero width, which is meaningless "
     "at n = 30. Wilson remains inside [0,1] and sensible at the extremes and at small n.", indent=0)
body("Intervals alone are insufficient, because a sufficiently small corpus produces intervals so "
     "wide that any conclusion drawn from the point estimates is unsupported — yet the table still "
     "looks like a result. We therefore implement a hard adequacy gate that refuses to run the "
     "ablation below the thresholds in Table I.")

table_caption([("TABLE I.    Dataset Adequacy Gate", "b")])
make_table(["Requirement", "Min.", "Rationale"],
           [["Distinct identities", "15", "Below this, purity and fragmentation move in large discrete "
             "steps and one mislabelled person dominates the mean."],
            ["Labelled person-frames", "1000", "Puts the 95% Wilson half-width near ±1.4 pp at a ~95% "
             "proportion; below ~300 the interval exceeds the effect sizes compared."],
            ["Cross-camera transitions", "10", "A re-association rate cannot be estimated from a "
             "handful of hand-offs."]],
           widths=[0.85, 0.35, 2.05])
p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(6)
r = p.add_run("The primary entry point exposes no override; an internal flag exists for pipeline "
              "smoke-testing and stamps any resulting report as not reportable.")
r.font.name, r.font.size = SERIF, Pt(7.8); r.font.color.rgb = MUT

h2("F.  Harness Fidelity")
body([("The offline runner contains ", ""), ("no identity logic", "b"),
      (". Every arbitration decision is delegated to the same production objects the live system uses "
       "— the identifier, the ID-switch guard, the evidence gate, the detector, the face recogniser "
       "and the tracker, constructed exactly as the live camera constructs them. A live-stream object "
       "is instantiated but never started, so its real methods and real per-track state are exercised "
       "while its capture threads and video device never run. Each configuration runs in a separate "
       "operating-system process against its own throwaway database, so no state can leak between "
       "configurations. The frames in Fig. 1 were produced through this same path.", "")], indent=0)
body("Two divergences from production are known and stated. The head-zoom second detection pass — a "
     "recall booster requiring capture state — is not exercised, so offline coverage is a lower bound "
     "relative to production. Annotation and stream encoding are omitted as display-only. Neither "
     "affects any metric reported here.")

# ── Table II full width ──────────────────────────────────────────────────
new_section(1)
table_caption([("TABLE II.    Ablation Configurations", "b")])
make_table(["Config", "Face anchor (H2) + face veto", "ID-switch guard", "Evidence gate",
            "Exposure retained"],
           [["A — pre-hardening", "—", "—", "—",
             "Colour and body re-ID run unconstrained even when a visible face matches nobody, and are "
             "accepted even when a visible face contradicts them. Reproduces, as closely as flags "
             "permit, the architecture under which three people were observed collapsing into one "
             "identifier."],
            ["B — face anchor", "yes", "—", "—",
             "Stranger-acquires-identifier is closed. A cached identifier is still never re-verified, "
             "so an occlusion hand-off goes undetected, and evidence is filed under whatever "
             "identifier is current."],
            ["C — + ID-switch guard", "yes", "yes", "—",
             "Hand-offs are corrected after two contradicting cycles, but evidence written during "
             "those cycles remains on disk under the wrong identifier."],
            ["D — full pipeline", "yes", "yes", "yes",
             "Durable evidence requires same-cycle face confirmation; a track carrying unresolved "
             "contradiction strikes writes nothing."]],
           widths=[1.15, 0.95, 0.72, 0.66, 3.55], hl_row=3)
p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(8)
r = p.add_run("Each row adds one mechanism to the row above; D is the shipped default and has been "
              "verified byte-identical to running with no configuration file present. Enrolment "
              "always requires a visible face in every configuration — no flag alters that.")
r.font.name, r.font.size = SERIF, Pt(7.8); r.font.color.rgb = MUT
new_section(2)

h1("VI", "Experimental Setup and Results")
h2("A.  Corpus")
body("We evaluate on the ChokePoint dataset [7], which is well matched to the deployment this system "
     "targets: subjects walking through a portal under three simultaneous cameras, recorded under "
     "real surveillance conditions rather than posed. Ground truth provides per-frame subject identity "
     "and eye coordinates, from which we derive a face bounding box by scaling the inter-ocular "
     "distance.", indent=0)
body("We use portal 1 entering, sequence 1 (P1E_S1), across all three cameras. Corpus statistics are "
     "computed by the adequacy gate from labels alone, without decoding pixels (Table III).")

table_caption([("TABLE III.    ChokePoint P1E_S1 Corpus", "b")])
make_table(["Property", "Value", "Req.", "Gate"],
           [["Distinct identities", "25", "15", "PASS"],
            ["Labelled person-frames", "2908", "1000", "PASS"],
            ["Cross-camera transitions", "50", "10", "PASS"],
            ["Cameras", "3", "—", "—"],
            ["Frames per identity (mean)", "116.3", "—", "—"]],
           widths=[1.35, 0.55, 0.45, 0.55])
para(after=4)

body("Every ground-truth person in this corpus appears on all three cameras, which makes it unusually "
     "well suited to cross-camera evaluation: the re-association metric has 50 pairs rather than the "
     "handful typical of opportunistically collected footage.")

h2("B.  Hardware and Timing Protocol")
body([("All measurements were taken on an Apple M-series machine with 16 GB unified memory. Face "
       "analysis runs through ONNXRuntime's CoreML execution provider on the Neural Engine, with CPU "
       "as fallback; person detection runs on the PyTorch backend's default device. No discrete GPU "
       "was used and no model was quantised. Reported latency wraps the full per-frame pipeline — detection, "
       "tracking, face analysis, arbitration and persistence — executed single-threaded and "
       "synchronously. This is ", ""), ("analysis", "i"),
      (" throughput, not stream frame rate; the deployed system decouples the two (Section III-B).",
       "")], indent=0)

h2("C.  Protocol")
body("Each of the four configurations processed every frame of all three cameras, with no frame cap — "
     "6876 frames per configuration, 27 504 in total — yielding 2908 scored ground-truth "
     "person-frames per configuration. All four saw identical input in identical order. Each ran in a "
     "separate process against its own database with no shared state. The bucket partition was "
     "verified to sum exactly for all four before any figure below was read.", indent=0)

figure(FIGS / "fig_buckets.png",
       [("Fig. 3.  ", "b"),
        ("Ground-truth outcome partition. Every person-frame falls in exactly one bucket and the "
         "three sum to 100%, asserted at runtime. Configuration A is confident and wrong; C and D "
         "reduce contamination to a single person-frame.", "")], CW)

# ── Table V full width ───────────────────────────────────────────────────
new_section(1)
table_caption([("TABLE IV.    Ground-Truth Outcome Partition (n = 2908 per configuration)", "b")])
make_table(["Config", "CORRECT", "CONTAMINATED", "UNASSIGNED", "Sum", "n"],
           [["A — pre-hardening", "449 · 15.4% [14.2–16.8]", "2454 · 84.4% [83.0–85.7]",
             "5 · 0.2% [0.1–0.4]", "100.0%", "2908"],
            ["B — face anchor", "1542 · 53.0% [51.2–54.8]", "1324 · 45.5% [43.7–47.3]",
             "42 · 1.4% [1.1–1.9]", "100.0%", "2908"],
            ["C — + ID-switch guard", "2859 · 98.3% [97.8–98.7]", "9 · 0.3% [0.2–0.6]",
             "40 · 1.4% [1.0–1.9]", "100.0%", "2908"],
            ["D — full pipeline", "2859 · 98.3% [97.8–98.7]", "9 · 0.3% [0.2–0.6]",
             "40 · 1.4% [1.0–1.9]", "100.0%", "2908"]],
           widths=[1.25, 1.55, 1.55, 1.30, 0.65, 0.45], hl_row=3)
para(after=6)

table_caption([("TABLE V.    Headline Metrics, ChokePoint P1E_S1, Three Cameras", "b")])
make_table(["Config", "Identity precision", "Coverage", "Evidence precision", "Ev. rows (wrong)",
            "Ev. yield", "Purity", "IDs", "Dup.", "Frag.", "ID sw.", "Cross-cam", "ms/fr."],
           [["A", "15.5% [14.2–16.8]", "99.8%", "15.5% [14.2–16.8]", "2903 (2454)", "99.8%",
             "0.0% [0.0–35.4]", "7", "0.40", "2.96", "47", "8.0% [3.2–18.8]", "55.6"],
            ["B", "53.8% [52.0–55.6]", "98.6%", "53.8% [52.0–55.6]", "2866 (1324)", "98.6%",
             "48.3% [31.4–65.6]", "29", "0.38", "2.56", "44", "28.0% [17.5–41.7]", "58.8"],
            ["C", "99.7% [99.4–99.8]", "98.6%", "99.7% [99.4–99.8]", "2868 (9)", "98.6%",
             "93.2% [83.8–97.3]", "59", "1.36", "2.52", "57", "50.0% [36.6–63.4]", "60.9"],
            ["D", "99.7% [99.4–99.8]", "98.6%", "100.0% [99.8–100]", "2402 (0)", "82.6%",
             "93.2% [83.8–97.3]", "59", "1.36", "2.52", "57", "50.0% [36.6–63.4]", "62.4"]],
           size=6.9,
           widths=[0.30, 0.90, 0.45, 0.90, 0.60, 0.42, 0.85, 0.26, 0.32, 0.34, 0.34, 0.90, 0.38],
           hl_row=3)
p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(8)
r = p.add_run("Precision and coverage are reported separately and must not be combined (Section V-C). "
              "Bracketed figures are 95% Wilson score intervals. Fragmentation and duplicates are "
              "means, not proportions, and carry no binomial interval; ID switches is a count. "
              "Latency is mean wall-clock over the full per-frame pipeline, CPU-only.")
r.font.name, r.font.size = SERIF, Pt(7.8); r.font.color.rgb = MUT

figure(FIGS / "fig_collapse.png",
       [("Fig. 4.  ", "b"),
        ("Identity collapse, made visible. Each cell is the share of that identifier's person-frames "
         "belonging to one ground-truth person; a pure identifier is a single dark cell in its row. "
         "(a) Under configuration A, seven identifiers smear across the population — no row concentrates, "
         "and purity is exactly zero. (b) Under configuration D, 35 of 36 identifiers resolve to a "
         "single person, giving the near-diagonal structure a correct system should exhibit. The "
         "residual off-diagonal mass in (b) is over-splitting, not contamination.", "")], FW, full=True)
new_section(2)

h2("D.  Configuration A: Identity Collapse Reproduced")
body([("The unhardened configuration exhibits collapse in its most complete form. It minted ", ""),
      ("seven identifiers to cover twenty-five people", "b"), (", and identity purity is ", ""),
      ("exactly zero", "b"), (" — not one of the six referred to a single individual. The "
       "distribution of contamination is severe rather than marginal: the worst identifier absorbed "
       "twenty-two of the twenty-five ground-truth people, a second absorbed thirteen, a third "
       "eleven. Fig. 4(a) shows "
       "the structure directly.", "")], indent=0)
body([("What makes this result pedagogically valuable is the company it keeps. Coverage is ", ""),
      ("99.8%", "b"), (" — the highest of any configuration. The system answered almost every time it "
       "was asked, with a confident on-screen label, and was wrong in roughly three cases out of "
       "four. Fragmentation is 2.96, which read in isolation might suggest over-splitting; in fact "
       "the system was catastrophically ", ""), ("under", "i"),
      ("-splitting, and fragmentation is elevated only because contamination scatters stray frames "
       "across identifiers. This is precisely the pathology that motivates reporting fragmentation "
       "and duplicates jointly, and never reading either alone.", "")])
body("Configuration A is also the strongest available evidence for this paper's central "
     "methodological claim. A retrieval-style evaluation of this configuration would find a gallery "
     "of six confident identities, each returning matches at Rank-1. Nothing in Rank-n or mAP would "
     "reveal that the gallery had fused most of the population into a handful of entries.")

h2("E.  Configuration B: Face Anchoring in Isolation")
body([("Enabling the face anchor and face veto more than triples identity precision, from 15.5% to ", ""),
      ("53.8%", "b"), (", with non-overlapping confidence intervals. Purity rises from 0.0% to 48.3% "
       "and the number of minted identifiers rises from 7 to 29 — the system begins separating people "
       "it had previously fused. Coverage falls by 1.2 points, entirely into quality-gate refusals "
       "rather than detector misses.", "")], indent=0)
body([("The improvement is real and the residual failure is equally informative. At 52.5% "
       "contamination, configuration B is still wrong more often than not. Invariant H2 closes the "
       "pathway by which a ", ""), ("stranger", "i"),
      (" acquires an existing identifier through clothing, but it does nothing about a track whose "
       "cached identifier was correct when established and became wrong when the tracker handed the "
       "track to a different person mid-occlusion. In that case no fresh arbitration ever runs — the "
       "cache is simply reused — so face anchoring never gets the opportunity to intervene.", "")])

h2("F.  Configuration C: The ID-Switch Guard Is Decisive")
body([("Adding the ID-switch guard moves identity precision from 53.8% to ", ""), ("99.7%", "b"),
      (" and purity from 48.3% to 93.2%. Contamination falls from 1324 person-frames to ", ""),
      ("nine", "b"), (". Cross-camera re-association nearly doubles, from 28.0% to 50.0%.", "")], indent=0)
body([("This is the single largest effect in the ablation, and its magnitude is initially surprising: "
       "the guard is a small mechanism, a similarity test against a cached identifier with a "
       "two-strike counter. The explanation lies in the interaction between the caching optimisation "
       "and the corpus. ChokePoint subjects walk through a portal in sequence, frequently passing one "
       "another within the camera's field of view. Every such crossing is an opportunity for ByteTrack "
       "to transfer a track identifier — and because arbitration is cached per track for CPU reasons, "
       "each transfer silently converts into an identity error that then persists for the remaining "
       "life of the track. Configuration B's dominant failure was not misrecognition at all; it was ",
       ""), ("stale correctness", "i"), (", an identifier that had been right and was never "
       "re-examined.", "")])

p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.left_indent = Inches(0.10)
p.paragraph_format.space_before = Pt(3); p.paragraph_format.space_after = Pt(4)
rich(p, [("Predicted behaviour confirmed.  ", "ba"),
         ("Raw ID switches increase from 44 to 57 between B and C, while contamination falls from 1324 "
          "person-frames to one. This is the effect anticipated in Section IV-E: correcting a hijacked "
          "track is itself recorded as a switch, so the more correct configuration registers more "
          "switches. A reviewer treating ID switches as a straightforward \"lower is better\" quantity "
          "would rank these two configurations backwards.", "i")], size=8.8)

body("The cost appears exactly where the architecture says it should. Minted identifiers rise from 29 "
     "to 59 for 25 real people, and duplicates per person rise from 0.38 to 1.36. This is the "
     "split-over-merge asymmetry of Section IV-H made quantitative, plotted in Fig. 5: preventing "
     "collapse produces over-splitting, and the system trades a visible, repairable error for an "
     "invisible, irreversible one. Coverage is unchanged at 98.6%, so the precision gain is not "
     "bought by declining to answer.")

figure(FIGS / "fig_tradeoff.png",
       [("Fig. 5.  ", "b"),
        ("The split-over-merge asymmetry, quantified. Marker area encodes identifiers minted. "
         "Configurations C and D coincide exactly on both axes, since the evidence gate alters "
         "neither contamination nor duplicates.", "")], CW)

figure(FIGS / "fig_trajectory.png",
       [("Fig. 6.  ", "b"),
        ("Identity precision, coverage and purity across the ablation. The shaded band is the 95% "
         "Wilson interval on precision. Coverage moves by 1.5 points across the whole sweep while "
         "precision moves by 76 — the hardening does not buy correctness by declining to answer.", "")],
       CW)

h2("G.  Configuration D: Evidence Precision")
body([("Configuration D is ", ""), ("identical to C on every assignment metric", "b"),
      (" — same buckets, same precision, same coverage, same purity, same duplicates, same switches, "
       "same cross-camera rate. This is the correct and expected outcome: the evidence gate governs "
       "what is written, not what is decided, so no metric scoring identifier assignment can "
       "distinguish the two.", "")], indent=0)
body([("The distinction appears only in evidence precision, which rises from 99.69% to ", ""),
      ("100.0%", "b"), (": the single contaminated row that configuration C persisted is not written "
       "at all under D. The cost is evidence yield, which falls from 98.3% to ", ""), ("82.6%", "b"),
      (" — roughly 466 fewer sighting rows, those belonging to frames where a cached identifier was "
       "displayed without same-cycle face confirmation (Fig. 7).", "")])
body("We regard this trade as favourable for the deployment in question, and the reasoning is worth "
     "making explicit because the numbers alone do not settle it. A missing sighting row is a gap in "
     "an evidence trail: an investigator sees that a person was observed at 10:03 and 10:07 but not "
     "at 10:05. A wrong sighting row is a false record placing a person somewhere they never were. "
     "The first is a known unknown; the second is misinformation that looks exactly like fact. For an "
     "evidence trail that may be read as a record of where someone was, we hold that the asymmetry "
     "justifies the yield cost — and, importantly, that this is a value judgement about consequences "
     "rather than a conclusion derivable from the metrics, which is why the harness reports yield "
     "beside precision rather than blending them.")
body("This result also validates the metric's inclusion. Evidence precision is the only quantity in "
     "the entire set that separates C from D; without it, the evidence gate would be invisible to "
     "evaluation and would appear to be a mechanism with no measurable effect.")

figure(FIGS / "fig_evidence.png",
       [("Fig. 7.  ", "b"),
        ("Evidence precision against evidence yield. Only this pair distinguishes configuration D "
         "from C: D writes 290 fewer rows and none of them is wrong.", "")], CW)

h2("H.  Latency")
body("Mean per-frame latency across the full pipeline rises monotonically with the number of "
     "enabled mechanisms, from 55.6 ms in A to 62.4 ms in D, with medians of "
     "50.1 to 52.8 ms, giving analysis throughput of 16.0 to 18.0 frames per second on 800×600 input, "
     "single-threaded. The 6.8 ms spread tracks the number of enabled mechanisms directly "
     "(Fig. 8).", indent=0)
body([("The ordering is expected. All four configurations run the same detection, tracking and "
       "face-analysis stages, which dominate cost; the hardening mechanisms add only cosine "
       "similarities against a small gallery and integer counter updates. The measurement supports a practical claim: ", ""),
      ("the accuracy gain from A to D costs 6.8 ms per frame", "b"),
      (", a 12% increase, while moving precision from 15.5% to 99.7%. "
       "The mechanisms are decision procedures, not additional models.", "")])
body("The median being consistently below the mean indicates a right-skewed distribution — a minority "
     "of frames, those with many simultaneous faces, cost substantially more than typical ones.")

figure(FIGS / "fig_latency.png",
       [("Fig. 8.  ", "b"),
        ("Per-frame analysis latency, 800×600 input, face analysis on the Neural Engine. The "
         "hardening mechanisms cost 6.8 ms per frame in total, rising monotonically A to D.", "")], CW)

h2("I.  Summary of Findings")
body([("The ablation supports four claims. ", ""), ("First", "b"),
      (", identity collapse is real, reproducible and severe: without the hierarchy, seven identifiers "
       "absorbed twenty-five people at 99.8% coverage and zero purity. ", ""), ("Second", "b"),
      (", face anchoring is necessary but not sufficient, doubling precision while leaving the "
       "majority of person-frames contaminated. ", ""), ("Third", "b"),
      (", cache invalidation under face contradiction is the decisive mechanism in this corpus, and "
       "its benefit is invisible to — indeed inverted by — the ID-switch count. ", ""), ("Fourth", "b"),
      (", separating display from persistence eliminates the residual wrong evidence at a yield cost "
       "that only a metric scoring persisted records can expose.", "")], indent=0)
body("The interpretive discipline of Section V was not decorative in reaching these conclusions. "
     "Reading precision without coverage would have made configuration A look merely mediocre rather "
     "than degenerate; reading fragmentation without duplicates would have suggested A was "
     "over-splitting when it was fusing; reading ID switches as monotone would have inverted the "
     "B-versus-C ranking; and omitting evidence precision would have rendered configuration D "
     "indistinguishable from C.")

h2("J.  Threats to Validity")
for lead, txt in [
    ("Corpus specificity.", " ChokePoint is a portal scenario with cooperative walking subjects at "
     "close range — precisely the regime in which the face anchor performs best. Results should not "
     "be extrapolated to wide-area or overhead deployments, where the quality gate suppresses most "
     "identification by design."),
    ("Majority-vote ground truth.", " The function maj(c) is computed empirically from the run being "
     "scored. With small samples, an identifier split evenly between two people resolves by "
     "first-encountered tie-break, making purity and contamination coarse at low n."),
    ("Synthetic timestamps.", " Frame timestamps are derived as frame index divided by frame rate, so "
     "any wall-clock-dependent behaviour is not faithfully reproduced offline. We therefore disable "
     "production's 30-second sighting de-duplication in the harness by default, writing one row per "
     "gate-passing frame, and record the setting in the metrics output."),
    ("Single hardware platform.", " Latency figures characterise one CPU-only machine and should not "
     "be read as architecture-independent."),
]:
    body([(lead, "b"), (txt, "")])

h1("VII", "Implementation and Operational Surface")
body("The arbitration logic of Section IV is embedded in a working system rather than a research "
     "script, and several implementation choices are consequences of the argument rather than "
     "incidental engineering.", indent=0)

h2("A.  Configuration as a Single Source of Truth")
body("Every threshold, window and feature flag discussed in this paper lives in one configuration "
     "object, loaded once and injected into the identifier, the live-stream helpers and the "
     "duplicate-suggestion endpoint. Nothing is hardcoded at a call site.", indent=0)
body("This is what makes the ablation of Table II possible at all. Each configuration is a JSON file "
     "overriding named fields; unknown keys raise rather than being silently ignored, because a "
     "research tool that quietly keeps a default when a parameter name is misspelled will produce a "
     "plausible table describing an experiment nobody ran. Each configuration is executed in its own "
     "process against its own database, so the configuration singleton, tracker state, model "
     "singletons and session registry all start cold.")
body("The full-pipeline configuration was verified byte-identical to running with no configuration "
     "file present, so the ablation's reference row is genuinely the shipped default rather than an "
     "approximation of it.")

h2("B.  Evidence Storage")
body("Each identifier owns a directory holding an enrolment photograph and per-sighting crops. "
     "Storing evidence as image files rather than references into source video is a deliberate trade: "
     "it costs disk and it duplicates personal data, but it makes the evidence trail self-contained "
     "and — importantly for Section VIII — makes erasure a matter of removing a directory whose "
     "contents are fully enumerable, rather than reasoning about which frames of which archived video "
     "contain which person.", indent=0)
body("The path for a directory is validated against the storage root before any deletion, because the "
     "identifier reaches that code from a URL path parameter; without the check, erasure becomes "
     "arbitrary file deletion.")

h2("C.  Operator Workflow for Duplicates")
body("Because the system prefers splitting to merging (Section IV-H), duplicate identifiers "
     "accumulate and need a repair path. An endpoint computes maximum cross-gallery cosine similarity "
     "between every pair of enrolled identities and returns pairs above an advisory threshold, sorted "
     "most-similar first, each banded as high, medium or low confidence with guidance on what the "
     "operator should verify.", indent=0)
body([("The bands are presentation only. The auto-merge field is hard-coded false at every band, and "
       "there is no code path that merges without an explicit operator action. A high band means ", ""),
      ("look at this first", "i"), (", never ", ""), ("this is safe to apply unexamined", "i"),
      (" — because a wrong automatic merge produces exactly the collapse this architecture exists to "
       "prevent, and does so invisibly, since the two evidence trails are combined before anyone can "
       "inspect them.", "")])
body("The operation is O(n²) in enrolled identities and identity-revealing by nature, so it is "
     "rate-limited and restricted to authenticated operators.")

h2("D.  Access Control")
body("Read access to identity data requires an operator role; erasure, audit inspection and retention "
     "purges require an administrator. Face search across the enrolled gallery — the most abusable "
     "operation the system exposes, since it answers \"where has this person been\" from a single "
     "photograph — carries both an explicit role check and a tight rate limit, and is audited whether "
     "or not it matches.", indent=0)

h1("VIII", "Data Governance and Verified Erasure")
body("A system that stores face embeddings incurs obligations that are not satisfied by "
     "documentation. We treat them as testable components.", indent=0)

h2("A.  Consent as a First-Class Field")
body("Every identity carries a consent status. There is no unset value: the absence of a recorded "
     "lawful basis is itself a recorded state, and it is treated as the weakest.", indent=0)

table_caption([("TABLE VI.    Retention by Consent Status", "b")])
make_table(["Status", "Meaning", "Retention"],
           [["consented", "A named individual signed a form; a reference identifies it", "365 d"],
            ["dataset", "Licensed research corpus under its own ethics approval", "3650 d"],
            ["unknown", "Default. Auto-enrolled from a live camera; nobody agreed to anything",
             "7 d"]],
           widths=[0.72, 1.90, 0.60])
para(after=4)

body([("We state plainly what this does and does not achieve. The unknown default exists because the "
       "system auto-enrols anyone clearing the quality gate, and those people have not consented and "
       "were likely not notified. The seven-day retention limits harm; ", ""),
      ("it does not create a lawful basis", "b"),
      (". A volunteer pilot with signed forms is the supported path; a public deployment requires a "
       "specific statutory basis, signage, an impact assessment and qualified legal review that no "
       "software architecture can substitute for.", "")])

h2("B.  Storage Limitation and the Purge Job")
body("Retention is enforced rather than documented. Each identity carries an expiry timestamp derived "
     "from its consent status and refreshed on each confirmed sighting, so that someone still being "
     "observed does not expire mid-visit. A purge job deletes every identity past its expiry, "
     "defaulting to a dry run that reports what would be removed — a default we consider mandatory "
     "for an irreversible bulk operation on personal data.", indent=0)
body("Three details are worth recording. Identities under legal hold are exempt regardless of expiry, "
     "since an active investigation is precisely the circumstance in which automated deletion is "
     "inappropriate. Rows predating the governance layer, which carry no expiry stamp, have one "
     "derived at purge time from their last-seen timestamp — otherwise legacy data would be immortal "
     "purely because it predates the feature that was supposed to govern it. And the audit log "
     "expires on its own schedule, because it contains personal data about both subject and operator "
     "and cannot be exempt from the storage-limitation principle it exists to demonstrate compliance "
     "with.")
body("A configuration option disables expiry for a consent class, but it requires an explicit "
     "negative value rather than being reachable by omission. \"Keep indefinitely\" is exactly what "
     "storage limitation forbids, so it is available only as a deliberate, recorded decision.")

h2("C.  Erasure, and Why It Must Be Tested")
body("The right to erasure is only as real as the code that implements it. We audited SmartDetect's "
     "deletion path against a full enumeration of where one person's data resides. The audit found "
     "the database row, its embeddings and multi-view gallery, all sighting rows, and all "
     "photographic evidence on disk correctly removed — and one category not removed at all.",
     indent=0)

p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.left_indent = Inches(0.10)
p.paragraph_format.space_before = Pt(3); p.paragraph_format.space_after = Pt(4)
rich(p, [("Audit finding.  ", "bx"),
         ("Deletion did not clear identity caches held in memory by running camera threads: the "
          "tracker-to-identifier map, the recency cache backing the live-persons endpoint, and the "
          "recent-detections ring buffer. A person erased at the operator's request could continue to "
          "appear as present on a live camera view, and in recent-activity listings, for as long as "
          "those caches held them — because those endpoints read memory, not the database.", "i")],
     size=8.8)

body("This is the characteristic shape of an erasure bug: the durable stores are handled because they "
     "are the ones the developer thinks of as \"the data\", while a cache that exists purely as a "
     "performance optimisation quietly retains the same personal data with no retention policy at "
     "all. We repaired it by adding a registry of running camera streams and a purge routine invoked "
     "as part of erasure, whose entry count is recorded on the erasure receipt.")
body([("Erasure is ", ""), ("verified rather than asserted", "b"),
      (". After deleting, the system re-queries the database and re-stats the filesystem, and records "
       "the result. A receipt is retained after the identity is gone — it holds counts and provenance "
       "but no biometric data, which is what makes it safe to keep and useful as proof that a request "
       "was honoured.", "")])
body("The deletion path is covered by an end-to-end test that enrols a person through the real "
     "identification pipeline from a real frame, confirms they are retrievable by face search, "
     "populates a running camera's caches exactly as the analysis thread would, deletes, and then "
     "asserts every enumerated item is gone — including that a photo search for that person now "
     "returns nothing. We verified the test's discriminating power by disabling the cache purge and "
     "confirming the test fails.")

h2("D.  Audit Logging")
body([("Every biometric access records actor, role, source address, action, subject and outcome. "
       "Searches that ", ""), ("fail", "i"),
      (" are logged too: an unsuccessful search still processed a face image against the entire "
       "enrolled gallery, and the pattern of who is being searched for is itself the thing requiring "
       "oversight. The audit log contains personal data about both subject and operator, so it "
       "carries its own retention period rather than being kept indefinitely.", "")], indent=0)

h1("IX", "Discussion and Limitations")
h2("A.  What the Architecture Cannot Do")
body([("It cannot identify people whose faces are never visible.", "b"),
      (" This is the direct and unavoidable cost of face-anchoring. In overhead retail footage and "
       "wide street scenes, most subjects remain permanently unidentified. We validated this "
       "explicitly: on a five-person overhead aisle clip, all subjects were detected and tracked but "
       "zero identities were minted, while on an eye-level clip of four people seated together, four "
       "identities were minted and correctly kept separate. The deciding factor is face visibility, "
       "not crowd density.", "")], indent=0)
body([("It produces duplicates.", "b"),
      (" The split-over-merge asymmetry means the same person may hold several identifiers — 36 for "
       "25 people in our ablation. Manual merge tooling exists, but the residual duplicate rate is a "
       "real cost.", "")])
body([("Its cross-camera association is weak by construction.", "b"),
      (" With no spatio-temporal topology model, cross-camera re-association depends entirely on face "
       "match quality at both cameras, and even the full pipeline reaches only 50.0% on this corpus.",
       "")])

h2("B.  Latency")
body("End-to-end analysis latency is the binding practical constraint. On high-resolution input, "
     "analysis lags real time sufficiently that overlay boxes visibly trail moving subjects, since "
     "capture runs at source rate while analysis does not. The identity decisions remain correct — "
     "they are simply attached to a frame several seconds old. Face analysis already runs on the "
     "Neural Engine; person detection does not, and moving it to the same accelerator, together "
     "with analysis-paced playback for offline review, are the mitigations under consideration.",
     indent=0)

h2("C.  Design Guidance for Practitioners")
body("Several conclusions generalise beyond this system, and we state them as guidance because each "
     "cost us a measurable error before it was learned.", indent=0)
for lead, txt in [
    ("Treat \"no match\" as a result, not a failure.", " The single highest-leverage change in this "
     "work was recognising that a good-quality face matching nobody is an informative outcome — this "
     "person is not enrolled — rather than a lookup failure to be papered over with a weaker signal. "
     "Any pipeline whose fallback chain triggers on \"primary method returned nothing\" should be "
     "examined for this confusion."),
    ("Rank your signals explicitly, and encode the ranking in control flow.", " A weighted score that "
     "blends face, colour and body similarity into one number allows weak evidence to outvote strong "
     "evidence whenever enough weak evidence accumulates. We considered and rejected such a fusion "
     "for exactly this reason: the ranking must be structural, expressed as gating, not as "
     "coefficients."),
    ("Separate what you show from what you store.", " Display and persistence have different "
     "evidential burdens. Conflating them means every transient display decision becomes a permanent "
     "record, and the system's worst momentary guess becomes indistinguishable in the database from "
     "its best confirmed one."),
    ("Cache invalidation is an identity-correctness problem.", " Caching arbitration per track is "
     "necessary for CPU-bound operation, but it converts a tracker error into an identity error and "
     "makes it stable. Any such cache needs an explicit contradiction test — in our ablation this "
     "single mechanism was worth 53 points of precision."),
    ("Decide which error you prefer, deliberately and in advance.", " Splitting and merging are not "
     "symmetric. If your system cannot avoid both, choose the one that is visible and repairable, and "
     "build the repair tooling as part of the same work rather than deferring it."),
    ("Instrument the denominator.", " The most misleading number we produced during this project came "
     "from silently omitting detector misses from the coverage denominator — an omission that "
     "flattered the system by twelve percentage points and was invisible in the output. Metrics "
     "should be defined over ground truth, never over the system's own detections."),
]:
    body([(lead, "b"), (txt, "")])

h2("D.  On Reporting Negative Space")
body("Several of this paper's more useful findings are things that did not work, or worked at a cost. "
     "The pose gate cost 6.7 percentage points of coverage with no duplicate reduction on sparsely "
     "sampled input, which is why the safety valve exists. A previously used snapshot-similarity check "
     "was retired as an accuracy measure once we recognised it was filtered by the very property it "
     "purported to measure — snapshots are written only when the evidence gate passes, so its sample "
     "was conditioned on the outcome. It survives only as a regression check. We report these because "
     "a threshold's history is often stronger evidence for a design than its final value.", indent=0)

h1("X", "Conclusion and Future Work")
body("Person re-identification evaluated as ranked retrieval and person re-identification deployed as "
     "an autonomous, self-enrolling decision process are different problems, and the metrics of the "
     "first do not detect the characteristic failure of the second. Identity collapse — the silent "
     "fusion of several people under one identifier, followed by progressive template corruption — is "
     "invisible to Rank-n and mAP because those metrics presuppose the very gallery that collapse "
     "destroys.", indent=0)
body("SmartDetect addresses this with a strict evidence hierarchy: a quality-gated face is the only "
     "signal permitted to create an identity, a visible face that matches nobody is treated as "
     "decisive rather than as a failed lookup, and appearance cues are confined to time-bounded "
     "re-association subject to face veto. Three further mechanisms — an ID-switch guard, a "
     "registration pose gate with an explicit safety valve, and an evidence gate separating display "
     "from durable record — close the residual pathways. The system prefers duplicate identifiers to "
     "merged ones, and we argue this asymmetry is correct because the two errors differ in "
     "visibility, containment and reversibility.")
body("The ablation quantifies each mechanism on a corpus of 25 identities across three cameras. "
     "Removing the hierarchy reproduces collapse in full: seven identifiers absorb twenty-five people, "
     "identity purity is exactly zero, and precision falls to 15.5% while coverage rises to 99.8% — "
     "the system is at its most confident precisely where it is least correct. Restoring face "
     "anchoring triples precision; adding cache invalidation under face contradiction raises it to "
     "99.7% and nearly doubles cross-camera re-association; adding the evidence gate eliminates the "
     "last nine incorrect database rows. Total additional inference cost is 6.8 ms per frame, because "
     "these are decision procedures rather than models.")
body([("Two results deserve emphasis for how badly conventional reading treats them. The "
       "configuration that reduces contamination from 1324 person-frames to nine simultaneously ", ""),
      ("increases", "i"), (" tracker identity switches from 44 to 57, because correcting a hijacked "
       "track is itself a switch — a reviewer treating that count as monotone would rank the "
       "configurations backwards. And the final mechanism is invisible to every metric that scores "
       "identifier assignment, becoming measurable only through a metric defined over rows actually "
       "written to disk.", "")])
body("Alongside the architecture we contribute a measurement framework built for this regime: an "
     "exhaustive outcome partition asserted at runtime, precision and coverage reported separately "
     "and never blended, a duplicate-identity metric separating over-splitting from contamination, an "
     "evidence-precision metric scoring what was actually persisted, and a dataset-adequacy gate that "
     "refuses to produce a results table a corpus cannot support. We also report a verified erasure "
     "path, including a class of bug — personal data surviving in performance caches inside running "
     "threads — that we suspect is common in systems of this shape and rarely tested for.")
body([("Four directions follow. ", ""), ("Tracklet-level voting", "b"),
      (" would aggregate evidence across a whole track before committing to an identity, rather than "
       "deciding on the first qualifying frame, which should reduce both duplicates and premature "
       "commitments — and our duplicate rate of 1.36 per person indicates substantial headroom. ", ""),
      ("Face-gated spatio-temporal constraints", "b"),
      (" could admit camera topology as a tie-breaker between face-plausible candidates while never "
       "permitting it to assert identity alone, targeting the 50.0% cross-camera rate directly. ", ""),
      ("Hardware-accelerated inference", "b"),
      (" on the target platform would relieve the latency constraint that currently bounds deployable "
       "resolution. Finally, ", ""), ("a clothing-change evaluation protocol", "b"),
      (" — subjects recorded twice with different outer garments — would isolate the colour "
       "fallback's contribution under the condition designed to defeat it, an ablation row we "
       "consider necessary before the fallback's continued inclusion can be justified on evidence "
       "rather than intuition.", "")])

h1("", "Acknowledgements")
body("The authors thank Ms. K. Sinduja for supervision throughout this project, and the Department "
     "of Information Technology, Sri Krishna College of Technology, Coimbatore, for providing the "
     "computing facilities on which this work was carried out. We acknowledge the "
     "creators of the ChokePoint dataset for releasing a corpus recorded under genuine surveillance "
     "conditions with per-frame identity annotation, and the maintainers of the YOLOv8, ByteTrack, "
     "InsightFace and OSNet projects, whose pretrained models this system composes without "
     "modification or further training.", indent=0)

h1("", "References")
REFS = [
    "H. Wang, H. Du, Y. Zhao, and J. Yan, \"A comprehensive overview of person re-identification "
    "approaches,\" IEEE Access, vol. 8, pp. 45556–45583, 2020.",
    "Y. Sun, L. Zheng, Y. Yang, Q. Tian, and S. Wang, \"Beyond part models: Person retrieval with "
    "refined part pooling (and a strong convolutional baseline),\" in Proc. ECCV, Munich, Germany, "
    "2018, pp. 501–518.",
    "W. Li, X. Zhu, and S. Gong, \"Harmonious attention network for person re-identification,\" in "
    "Proc. IEEE/CVF CVPR, Salt Lake City, UT, USA, Jun. 2018, pp. 2285–2294.",
    "A. Hermans, L. Beyer, and B. Leibe, \"In defense of the triplet loss for person "
    "re-identification,\" arXiv:1703.07737, 2017.",
    "Z. Zheng, L. Zheng, Z. Hu, and Y. Yang, \"Open set adversarial examples,\" arXiv:1809.02681, 2018.",
    "K. Zhou, Y. Yang, A. Cavallaro, and T. Xiang, \"Omni-scale feature learning for person "
    "re-identification,\" in Proc. IEEE/CVF ICCV, Seoul, South Korea, Oct. 2019, pp. 3702–3712.",
    "Y. Wong, S. Chen, S. Mau, C. Sanderson, and B. C. Lovell, \"Patch-based probabilistic image "
    "quality assessment for face selection and improved video-based face recognition,\" in Proc. IEEE "
    "CVPR Workshops, Colorado Springs, CO, USA, Jun. 2011, pp. 74–81.",
    "J. Deng, J. Guo, N. Xue, and S. Zafeiriou, \"ArcFace: Additive angular margin loss for deep face "
    "recognition,\" in Proc. IEEE/CVF CVPR, Long Beach, CA, USA, Jun. 2019, pp. 4690–4699.",
    "Y. Zhang et al., \"ByteTrack: Multi-object tracking by associating every detection box,\" in "
    "Proc. ECCV, Tel Aviv, Israel, 2022, pp. 1–21.",
    "E. B. Wilson, \"Probable inference, the law of succession, and statistical inference,\" J. Amer. "
    "Statist. Assoc., vol. 22, no. 158, pp. 209–212, 1927.",
    "L. D. Brown, T. T. Cai, and A. DasGupta, \"Interval estimation for a binomial proportion,\" "
    "Statist. Sci., vol. 16, no. 2, pp. 101–133, 2001.",
    "E. Ristani, F. Solera, R. Zou, R. Cucchiara, and C. Tomasi, \"Performance measures and a data set "
    "for multi-target, multi-camera tracking,\" in Proc. ECCV Workshops, Amsterdam, The Netherlands, "
    "2016, pp. 17–35.",
    "L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, and Q. Tian, \"Scalable person re-identification: A "
    "benchmark,\" in Proc. IEEE ICCV, Santiago, Chile, Dec. 2015, pp. 1116–1124.",
    "S. Karanam, M. Gou, Z. Wu, A. Rates-Borras, O. Camps, and R. J. Radke, \"A systematic evaluation "
    "and benchmark for person re-identification: Features, metrics, and datasets,\" IEEE Trans. "
    "Pattern Anal. Mach. Intell., vol. 41, no. 3, pp. 523–536, Mar. 2019.",
    "Government of India, The Digital Personal Data Protection Act, 2023, Act No. 22 of 2023.",
    "European Parliament and Council, Regulation (EU) 2016/679 (General Data Protection Regulation), "
    "Art. 9, Apr. 2016.",
]
for i, ref in enumerate(REFS, 1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.left_indent = Inches(0.22)
    p.paragraph_format.first_line_indent = Inches(-0.22)
    r = p.add_run(f"[{i}]  ")
    r.font.name, r.font.size = SERIF, Pt(8)
    r2 = p.add_run(ref)
    r2.font.name, r2.font.size = SERIF, Pt(8)
    r2.font.color.rgb = INK

doc.save(OUT)
print("wrote", OUT)
print("size", round(OUT.stat().st_size / 1024, 1), "KB")
