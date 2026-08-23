"""Compose the multi-panel 'system in operation' figure from real run frames."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FIGS = Path("/private/tmp/claude-501/-Users-lokii-Downloads-Smart-Detect-main/"
            "4e0d6171-becd-4766-a354-d53068b6a2ba/scratchpad/figs")

PANELS = [
    ("run_0272.png", "(a)  quality gate refuses the distant face",
     "foreground face 137 px passes the 48 px gate and is enrolled as SDT-0001; "
     "the background face at 42 px is detected but refused, so that person stays \"Detecting…\""),
    ("run_0296.png", "(b)  two identities held apart in one frame",
     "both subjects clear the gate and receive separate identifiers — no merge, "
     "despite simultaneous presence and similar dress"),
    ("run_0420.png", "(c)  enrolment from a clean frontal frame",
     "the pose gate admits this view; method new_registration mints SDT-0004"),
    ("run_0500.png", "(d)  identity retained with no visible face",
     "no face clears the gate this cycle, yet the track keeps SDT-0004 on screen — "
     "under config D no evidence row is written until a face reconfirms"),
]

W = 1100          # panel width
CAPH = 66         # caption strip height
PAD = 14
BG = (255, 255, 255)
INK = (22, 24, 29)
MUT = (90, 96, 108)

try:
    fb = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 25)
    fr = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 21)
except OSError:
    fb = fr = ImageFont.load_default()


def wrap(draw, text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= maxw:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


tiles = []
for fn, title, sub in PANELS:
    p = FIGS / fn
    if not p.is_file():
        print("skip missing", fn); continue
    im = Image.open(p).convert("RGB")
    im = im.resize((W, int(im.height * W / im.width)), Image.LANCZOS)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    sublines = wrap(probe, sub, fr, W - 8)
    caph = 34 + len(sublines) * 26 + 8
    tile = Image.new("RGB", (W, im.height + caph), BG)
    tile.paste(im, (0, 0))
    d = ImageDraw.Draw(tile)
    y = im.height + 8
    d.text((2, y), title, font=fb, fill=INK)
    y += 30
    for ln in sublines:
        d.text((2, y), ln, font=fr, fill=MUT)
        y += 26
    tiles.append(tile)

cols = 2
rows = (len(tiles) + cols - 1) // cols
cw = max(t.width for t in tiles)
rh = [max(t.height for t in tiles[r*cols:(r+1)*cols]) for r in range(rows)]
canvas = Image.new("RGB", (cols * cw + (cols + 1) * PAD,
                           sum(rh) + (rows + 1) * PAD), BG)
for i, t in enumerate(tiles):
    r, c = divmod(i, cols)
    x = PAD + c * (cw + PAD)
    y = PAD + sum(rh[:r]) + r * PAD
    canvas.paste(t, (x, y))

out = FIGS / "fig_operation.png"
canvas.save(out, dpi=(300, 300))
print("wrote", out, canvas.size)
