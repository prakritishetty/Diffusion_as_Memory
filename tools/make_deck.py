"""Generate the team-meeting deck.

A plain-language explainer: what we changed, why, what the data already says, and what to run
next. Numbers are the measured ones from `runs/`, not placeholders.

    python -m tools.make_deck --out Diffusion_as_Memory_update.pptx
"""

from __future__ import annotations

import argparse
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

INK = RGBColor(0x1A, 0x1D, 0x24)
MUTED = RGBColor(0x5F, 0x6B, 0x7A)
ACCENT = RGBColor(0x1F, 0x6F, 0xD4)
GOOD = RGBColor(0x1B, 0x7F, 0x4B)
BAD = RGBColor(0xB3, 0x2D, 0x2D)
WARN = RGBColor(0xB5, 0x72, 0x0D)
BG = RGBColor(0xFF, 0xFF, 0xFF)
BAND = RGBColor(0xF2, 0xF5, 0xF9)

W, H = Inches(13.333), Inches(7.5)


def _blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = BG
    return s


def _tb(slide, x, y, w, h, text, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT,
        space_after=6, line=1.15):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    lines = text if isinstance(text, list) else [text]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        indent = 0
        while isinstance(ln, str) and ln.startswith("\t"):
            indent += 1
            ln = ln[1:]
        p.text = ln
        p.alignment = align
        p.level = indent
        p.space_after = Pt(space_after)
        p.line_spacing = line
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.color.rgb = color
            r.font.name = "Segoe UI"
    return box


def _rule(slide, y, x=Inches(0.7), w=Inches(11.9), color=ACCENT, h=Pt(3)):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, Emu(int(h)))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def _band(slide, y, h, color=BAND):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), y, W, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    slide.shapes._spTree.remove(sh._element)
    slide.shapes._spTree.insert(2, sh._element)
    return sh


def title_slide(prs, title, subtitle, footer):
    s = _blank(prs)
    _band(s, Inches(0), Inches(2.6))
    _tb(s, Inches(0.9), Inches(0.75), Inches(11.5), Inches(1.2), title, 40, True, INK)
    _rule(s, Inches(1.85), Inches(0.9), Inches(2.2))
    _tb(s, Inches(0.9), Inches(2.05), Inches(11.5), Inches(0.6), subtitle, 19, False, MUTED)
    _tb(s, Inches(0.9), Inches(3.2), Inches(11.5), Inches(3.2), footer, 18, False, INK,
        space_after=12)
    return s


def section(prs, kicker, title):
    s = _blank(prs)
    _band(s, Inches(2.4), Inches(2.0))
    _tb(s, Inches(0.9), Inches(2.65), Inches(11.5), Inches(0.4), kicker, 15, True, ACCENT)
    _tb(s, Inches(0.9), Inches(3.05), Inches(11.5), Inches(1.0), title, 34, True, INK)
    return s


def bullets(prs, title, items, note=None, size=18):
    s = _blank(prs)
    _tb(s, Inches(0.7), Inches(0.45), Inches(11.9), Inches(0.7), title, 28, True, INK)
    _rule(s, Inches(1.15), Inches(0.7), Inches(1.6))
    _tb(s, Inches(0.75), Inches(1.5), Inches(11.8), Inches(4.7), items, size, space_after=10)
    if note:
        _tb(s, Inches(0.75), Inches(6.35), Inches(11.8), Inches(0.8), note, 15, False, MUTED)
    return s


def table(prs, title, headers, rows, note=None, col_w=None, highlight=None, size=14):
    s = _blank(prs)
    _tb(s, Inches(0.7), Inches(0.45), Inches(11.9), Inches(0.7), title, 28, True, INK)
    _rule(s, Inches(1.15), Inches(0.7), Inches(1.6))

    nrow, ncol = len(rows) + 1, len(headers)
    height = Inches(min(4.9, 0.42 * nrow + 0.2))
    shape = s.shapes.add_table(nrow, ncol, Inches(0.7), Inches(1.5), Inches(11.9), height)
    tbl = shape.table
    if col_w:
        total = sum(col_w)
        for i, frac in enumerate(col_w):
            tbl.columns[i].width = Emu(int(Inches(11.9) * frac / total))

    for j, h in enumerate(headers):
        c = tbl.cell(0, j)
        c.text = h
        c.fill.solid()
        c.fill.fore_color.rgb = INK
        for p in c.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
            for r in p.runs:
                r.font.size = Pt(size)
                r.font.bold = True
                r.font.color.rgb = BG
                r.font.name = "Segoe UI"

    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.text = str(val)
            c.fill.solid()
            c.fill.fore_color.rgb = BG if i % 2 else BAND
            colour = INK
            if highlight:
                colour = highlight(i - 1, j, val) or INK
            for p in c.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
                for r in p.runs:
                    r.font.size = Pt(size)
                    r.font.color.rgb = colour
                    r.font.bold = colour in (GOOD, BAD)
                    r.font.name = "Segoe UI"
    if note:
        _tb(s, Inches(0.7), Inches(1.55) + height, Inches(11.9), Inches(1.0), note, 15,
            False, MUTED)
    return s


def two_col(prs, title, left_title, left, right_title, right, note=None):
    s = _blank(prs)
    _tb(s, Inches(0.7), Inches(0.45), Inches(11.9), Inches(0.7), title, 28, True, INK)
    _rule(s, Inches(1.15), Inches(0.7), Inches(1.6))
    _tb(s, Inches(0.75), Inches(1.45), Inches(5.6), Inches(0.4), left_title, 17, True, BAD)
    _tb(s, Inches(0.75), Inches(1.95), Inches(5.6), Inches(4.2), left, 16, space_after=9)
    _tb(s, Inches(6.95), Inches(1.45), Inches(5.6), Inches(0.4), right_title, 17, True, GOOD)
    _tb(s, Inches(6.95), Inches(1.95), Inches(5.6), Inches(4.2), right, 16, space_after=9)
    if note:
        _tb(s, Inches(0.75), Inches(6.4), Inches(11.8), Inches(0.7), note, 15, False, MUTED)
    return s


# ---------------------------------------------------------------------------------------------
def build(out_path: str) -> str:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # 1 ----------------------------------------------------------------------------------------
    title_slide(
        prs,
        "Diffusion as Memory — where we are, what changed",
        "Rebuild of the pipeline, plus results that already change the plan",
        [
            "The idea in one line: store a memory once, then read it back at "
            "any level of detail — and stop as soon as you have enough.",
            "",
            "This deck: what was broken, what the data already told us, what we "
            "rebuilt, and what to run next.",
        ],
    )

    # 2 ----------------------------------------------------------------------------------------
    bullets(prs, "The idea, in plain language", [
        "A memory is stored as two things: a GIST (the gene­ral point) and DETAILS "
        "(names, numbers, dates).",
        "",
        "Time passing = adding noise to the details. The gist is left untouched.",
        "",
        "Remembering = removing that noise step by step, using the gist as the cue. "
        "Stop early and you get a true-but-vague answer; keep going and the specifics come back.",
        "",
        "Why anyone should care: normal retrieval is a switch — you get the whole stored "
        "passage or nothing. We want a dial.",
        "",
        "And a dial is only useful if turning it down never makes you WRONG, only less specific.",
    ], note="That last line is the whole claim. Everything we measure is aimed at it.")

    # 3 ----------------------------------------------------------------------------------------
    section(prs, "PART 1", "What we found when we audited the old code")

    # 4 ----------------------------------------------------------------------------------------
    table(
        prs, "Four defects that make the current results untrustworthy",
        ["#", "What was wrong", "Why it matters"],
        [
            ["1", "Validation scored the model while showing it the answer",
             "The reported BERTScore of 0.932 is inflated — the decoder was fed the correct "
             "words one at a time"],
            ["2", "The 'different but related' training example was the SAME sentence",
             "The gist vector had nothing to learn, so it was decoration"],
            ["3", "Stage 3 trained on 1 denoising step, but would run hundreds at test time",
             "Trained on inputs it would never actually see"],
            ["4", "The original sentence was never a training target",
             "The model could literally never be asked for full detail"],
        ],
        note="12 more issues found, including: a third of sentences silently cut off at 64 tokens; "
             "the projection module could not look at more than one slot at a time; three config "
             "files disagreed on the latent size. Full list in RESEARCH_PLAN.md §0.",
        col_w=[0.5, 4.5, 7],
    )

    # 5 ----------------------------------------------------------------------------------------
    bullets(prs, "The supervision bug, explained without maths", [
        "We have ~1000 'noise levels' and about 5 'detail levels' per sentence.",
        "",
        "The old rule mapped noise level to detail level by dividing by 100 — which assumes "
        "10 detail levels, not 5.",
        "",
        "Result: roughly 6 out of every 10 training steps asked for the SAME most-abstract "
        "answer, whatever the noise level was.",
        "",
        "So the model was effectively taught two settings — 'slightly vague' and 'very vague' — "
        "and nothing in between. The in-between is the entire product.",
        "",
        "Fix: every detail level now owns an equal slice of the noise range, computed per "
        "sentence, and we deliberately sample each level equally often.",
    ], note="Measured: old rule 51% of steps on the last rung; new rule 17% (= 1/6, which is "
            "exactly even). Verified by an automated test.")

    # 6 ----------------------------------------------------------------------------------------
    section(prs, "PART 2", "Results we already have — no GPU needed")

    # 7 ----------------------------------------------------------------------------------------
    def hl(i, j, v):
        if j == 0:
            return None
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        if j == 1:
            return GOOD if x > 0.9 else (BAD if x < 0.5 else WARN)
        if j == 2:
            return BAD if x < 0.15 else (GOOD if x > 0.6 else WARN)
        if j == 4:
            return GOOD if x > 0.8 else (BAD if x < 0.2 else WARN)
        return None

    table(
        prs, "GATE 1: how good is our training data, really?",
        ["Ladder source", "Detail falls\nsmoothly", "Never adds\nanything new",
         "Makes things\nup", "Complete\nsentences"],
        [
            ["Gold ladders (GPT-written)", "0.92", "0.05", "0.24", "0.86"],
            ["Gold ladders (our new rule-based set)", "1.00", "0.03", "0.09", "0.99"],
            ["Just truncating the sentence", "0.93", "0.73", "0.02", "0.08"],
        ],
        note="Read this as: our best model so far scored 0.325 on column 1, and the data's ceiling "
             "is 0.92 — so this is a MODELLING problem, not a data problem. Green light to train.",
        col_w=[4.4, 1.9, 1.9, 1.9, 1.9], highlight=hl,
    )

    # 8 ----------------------------------------------------------------------------------------
    bullets(prs, "But the same table killed our headline claim", [
        "We wanted to claim: 'each vaguer answer never adds anything the detailed one "
        "didn't say' — so stopping early is safe.",
        "",
        "Problem 1 — our own gold data only does that 5% of the time. GPT rewrites "
        "sentences rather than deleting from them, so new words appear constantly.",
        "",
        "Problem 2 — plain TRUNCATION scores 0.73 on it. Cutting a sentence in half is "
        "trivially 'never adds anything new'. Scissors beat us.",
        "",
        "So that property alone is not a contribution. But look at the last column: "
        "truncation produces a complete sentence only 8% of the time.",
        "",
        "New headline metric — USABLE: a rung counts only if it is a complete sentence AND "
        "adds nothing new AND invents nothing.",
    ], note="Catching this now, from data, is much cheaper than catching it in review.")

    # 9 ----------------------------------------------------------------------------------------
    table(
        prs, "The scoreboard the model has to beat",
        ["Approach", "USABLE score", "What it tells us"],
        [
            ["Truncate the sentence", "0.08", "Trivially consistent, but produces fragments"],
            ["GPT-written ladders (our training data)", "0.33", "The natural-language target"],
            ["Rule-based ladders (new, built this week)", "0.56", "Deletion-based, so "
             "consistency holds by construction"],
            ["Our model", "to be measured", "Must clear 0.33 to be interesting"],
        ],
        note="USABLE = complete sentence AND adds nothing specific that the finer rung lacked AND "
             "invents nothing. This is the number to put on the first slide of the paper.",
        col_w=[5, 2.4, 5.6],
    )

    # 10 ---------------------------------------------------------------------------------------
    section(prs, "PART 3", "What we rebuilt, and why")

    # 11 ---------------------------------------------------------------------------------------
    two_col(
        prs, "The memory itself: gist vs details",
        "Before",
        ["The gist and the details were both a single linear layer on the same vector.",
         "",
         "So they were not really different things — any separation had to come from the "
         "loss, and the only loss was 'rebuild the sentence'.",
         "",
         "Nothing ever checked that the gist contained the gist."],
        "Now",
        ["Gist: an attention layer picks WHICH parts matter, then a small network compresses "
         "them. Length-normalised so it behaves as a stable cue.",
         "",
         "Details: slots can talk to each other, and the output is scale-controlled.",
         "",
         "Three new training signals: write the summary from the gist alone; put the gist "
         "near an off-the-shelf sentence embedding; keep gist and details from duplicating "
         "each other."],
        note="The second of those is the bridge to question answering: if the gist lives in a "
             "standard search space, a QUESTION lands next to it for free.",
    )

    # 12 ---------------------------------------------------------------------------------------
    bullets(prs, "The most interesting change: give forgetting a floor", [
        "Standard diffusion destroys everything — at maximum noise you have pure static.",
        "",
        "So our 'most abstract' rung was anchored on nothing, which is exactly where the old "
        "runs produced gibberish.",
        "",
        "Change: we add noise to (details MINUS gist) instead of to everything.",
        "",
        "Now maximum noise = the gist plus static, not static. Details dissolve; the gist "
        "survives — by construction, not by hope.",
        "",
        "It is about fifteen lines of code and no new mathematics: ordinary diffusion on a "
        "shifted variable.",
    ], note="This is the change worth writing a paper around. It makes the memory metaphor "
            "structurally true instead of aspirational.")

    # 13 ---------------------------------------------------------------------------------------
    table(
        prs, "Everything else that changed",
        ["Part", "Change", "Why"],
        [
            ["Data", "Ladder now starts at the original sentence; sentence limit 64 → 128",
             "Full detail was unreachable; a third of sentences were being cut"],
            ["Data", "Contrastive partner = another rung of the same ladder",
             "Free, and it IS the claim: the gist must not change as detail fades"],
            ["Encoder", "t5-small → flan-t5-base; separate frozen encoder for the gist",
             "Bigger win than more parameters; makes question-as-cue work later"],
            ["Projection", "Turn on attention; let it read the gist directly; give it its own loss",
             "It was a per-slot MLP with no target of its own"],
            ["Decoder", "Told which level of detail to produce, via a 'level token'",
             "It had to GUESS whether a short answer was intended or a failure"],
            ["Diffusion", "Predict the clean latent directly; multi-step only",
             "Removes the divide-by-almost-zero that broke high-noise readout"],
            ["Evaluation", "Count facts by type instead of embedding similarity",
             "BERTScore cannot tell Paris from London"],
        ],
        col_w=[1.7, 5.3, 5], size=13,
    )

    # 14 ---------------------------------------------------------------------------------------
    section(prs, "PART 4", "How we know it works before spending a penny on GPUs")

    # 15 ---------------------------------------------------------------------------------------
    bullets(prs, "Everything is testable on a laptop", [
        "133 automated tests, all passing, ~50 seconds, no GPU and no internet.",
        "",
        "The whole 3-stage pipeline runs end to end on CPU in 37 seconds using a "
        "tiny stand-in model: python -m tools.smoke",
        "",
        "Tests specifically check the things that went wrong last time:",
        "\tdoes every component actually receive gradient?",
        "\tdoes the frozen part stay frozen?",
        "\tis stage 3 fed exactly what inference will feed it?",
        "\tdoes a saved checkpoint reload to identical outputs?",
        "\tis the same trajectory reproducible twice?",
        "",
        "Two bugs were caught this way already — one where the 'random noise' was "
        "accidentally identical to the data.",
    ], note="What this cannot tell us is whether the idea works. It tells us that when we get "
            "GPUs, 'press go' is a scheduling problem, not a debugging one.")

    # 16 ---------------------------------------------------------------------------------------
    section(prs, "PART 5", "What to run, in what order")

    # 17 ---------------------------------------------------------------------------------------
    table(
        prs, "The experiments, and the command for each",
        ["#", "Experiment", "Command", "GPU"],
        [
            ["0", "Is the latent at the right scale?",
             "python -m tools.diagnose_v0_scale --p0 <ckpt>", "10 min"],
            ["1", "Data ceiling  (DONE)", "python -m evaluation.ceiling", "none"],
            ["2", "Does a plain model already do this?",
             "python -m baselines.level_token", "~1 h"],
            ["3", "Train the autoencoder", "python -m training.train_p0 --preset full", "2-3 h"],
            ["4", "Train the denoiser", "python -m training.train_p1 --p0-checkpoint ...",
             "<0.5 h"],
            ["5", "Train the projection + reader",
             "python -m training.train_p2 --p0-... --p1-...", "~2 h"],
            ["6", "Score it", "python -m evaluation.evaluate --p0 ... --p1 ... --p2 ...",
             "~1 h"],
            ["7", "Ablations (cue, slots, brackets)", "python -m ablations.runner --only ...",
             "~15 h"],
        ],
        note="Experiment 2 is a decision point: if a plain 'level 3: <sentence>' model matches us "
             "on USABLE, the diffusion machinery is not justified on this task and we pivot to "
             "the question-answering setting instead.",
        col_w=[0.5, 4.2, 5.8, 1.5], size=13,
    )

    # 18 ---------------------------------------------------------------------------------------
    bullets(prs, "Three moments where we stop and reconsider", [
        "GATE 0 — is the stored vector the right size? If it is far from 1, forgetting will "
        "happen all at once instead of gradually. Ten minutes, and it may explain the old "
        "'cliff' behaviour entirely.",
        "",
        "GATE 1 — DONE. Data ceiling is 0.92 against our 0.325, so the problem is ours, "
        "not the data's. Proceed.",
        "",
        "GATE 2 — does the simple baseline already win? One GPU-hour. If yes, we change "
        "the claim rather than spend a month finding out.",
        "",
        "GATE 3 — shuffle the gist between examples. If recall barely changes, the gist is "
        "decorative and the whole cue story is dead. Twelve minutes.",
    ], note="Each gate is cheap and each can save weeks. Gate 3 is the one that could most "
            "embarrass us in review, so we run it first among the trained ones.")

    # 19 ---------------------------------------------------------------------------------------
    table(
        prs, "What it costs",
        ["Item", "Amount", "Notes"],
        [
            ["Total GPU time", "150-220 hours", "Everything: main runs, baselines, ablations, "
             "failures"],
            ["On a cheap cloud GPU (T4, spot)", "$25 - $35", "Fits in 16 GB; no A100 needed"],
            ["On a cheap cloud GPU (T4, on-demand)", "$85 - $120", "Safer, no eviction"],
            ["API credits", "$0 - $25", "Paraphrasing and rule-based data are local and free"],
            ["Free option", "$0", "UMass Unity, or Azure for Students (~$100)"],
        ],
        note="Money is not the constraint — GPU access and our own time are. Note: Azure gives "
             "you ZERO GPU quota by default and approval can take a day, so request it first.",
        col_w=[4.5, 2.5, 6],
    )

    # 20 ---------------------------------------------------------------------------------------
    bullets(prs, "What I need from the room", [
        "1.  Do we restart training from scratch with the bigger encoder? "
        "(My view: yes — the old checkpoints are invalid anyway because of the "
        "sentence-length bug.)",
        "",
        "2.  Do we keep MSR sentences, or move to a real long-conversation benchmark "
        "(LOCOMO / LongMemEval) for the memory story?",
        "",
        "3.  Is 'gist-anchored forgetting' a headline experiment or a footnote? "
        "(My view: headline — it is the most original thing here.)",
        "",
        "4.  Who takes which workstream? Data + metrics, model, diffusion, and "
        "downstream QA can all proceed in parallel.",
        "",
        "5.  Do we have GPU access sorted — Unity, Azure credits, or do we pay?",
    ], note="Full technical detail, including all 16 defects and the reasoning behind every "
            "change, is in RESEARCH_PLAN.md in the repo.")

    # 21 ---------------------------------------------------------------------------------------
    s = _blank(prs)
    _band(s, Inches(2.2), Inches(2.4))
    _tb(s, Inches(0.9), Inches(2.5), Inches(11.5), Inches(0.5), "WHERE WE ARE", 15, True, ACCENT)
    _tb(s, Inches(0.9), Inches(2.95), Inches(11.5), Inches(1.6), [
        "The pipeline is rebuilt, tested and runnable.",
        "The data says the problem is ours to fix, not the data's.",
    ], 26, True, INK, space_after=6)
    _tb(s, Inches(0.9), Inches(5.0), Inches(11.5), Inches(1.6), [
        "Next: Gate 0 (10 min) → Gate 2 baseline (1 h) → train.",
        "Everything else is already written and waiting for a GPU.",
    ], 18, False, MUTED, space_after=8)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    prs.save(out_path)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the team-meeting deck")
    ap.add_argument("--out", default="Diffusion_as_Memory_update.pptx")
    args = ap.parse_args(argv)
    path = build(args.out)
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    main()
