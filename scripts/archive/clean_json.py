import json
import re
from pathlib import Path

INP = Path("data/raw/train_with_summaries_gemma_better_prompt_part2.json")
OUT = Path("data/processed/train_part2_clean.json")

REPL = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
}

def clean_text(s: str) -> str:
    if s is None:
        return ""
    for k, v in REPL.items():
        s = s.replace(k, v)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(INP.read_text(encoding="utf-8"))
    changed = 0

    for r in data:
        before_x = r.get("x", "")
        r["x"] = clean_text(before_x)

        # handle summary key variants
        if "y" in r:
            before_y = r.get("y", "")
            r["y"] = clean_text(before_y)
        if "summary" in r:
            before_s = r.get("summary", "")
            r["summary"] = clean_text(before_s)

        # xt is optional for now, but clean it anyway
        if isinstance(r.get("xt"), list):
            r["xt"] = [clean_text(t) for t in r["xt"] if isinstance(t, str)]

        if before_x != r["x"]:
            changed += 1

    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", OUT)
    print("Total records", len(data))
    print("Records where x changed", changed)

if __name__ == "__main__":
    main()
