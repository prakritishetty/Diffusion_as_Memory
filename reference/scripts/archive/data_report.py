import json
import re
from collections import Counter

# PATH = "data/raw/train_with_summaries_gemma_better_prompt_part2.json"
PATH = "data/processed/train_part2_clean.json"

def clean(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def has_artifacts(s: str) -> bool:
    return any(tok in s for tok in ["-LRB-", "-RRB-", "-LSB-", "-RSB-"])

def main():
    with open(PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = Counter()
    bad_ids = {
        "empty_x": [],
        "empty_y": [],
        "no_content_y": [],
        "artifact_x": [],
        "artifact_y": [],
        "xt_empty": [],
        "xt_has_artifact": [],
    }

    for r in data:
        rid = str(r.get("id", ""))
        x = clean(r.get("x", ""))
        y = clean(r.get("y", ""))  # some files use 'summary' instead, we will check later if needed
        xt = r.get("xt", None)

        if not x:
            stats["empty_x"] += 1
            bad_ids["empty_x"].append(rid)

        if not y:
            stats["empty_y"] += 1
            bad_ids["empty_y"].append(rid)
        elif y.lower() in {"no content.", "no content", "none", "n/a"}:
            stats["no_content_y"] += 1
            bad_ids["no_content_y"].append(rid)

        if has_artifacts(x):
            stats["artifact_x"] += 1
            bad_ids["artifact_x"].append(rid)

        if has_artifacts(y):
            stats["artifact_y"] += 1
            bad_ids["artifact_y"].append(rid)

        if not isinstance(xt, list) or len(xt) == 0:
            stats["xt_empty"] += 1
            bad_ids["xt_empty"].append(rid)
        else:
            if any(has_artifacts(clean(t)) for t in xt if isinstance(t, str)):
                stats["xt_has_artifact"] += 1
                bad_ids["xt_has_artifact"].append(rid)

    print("Total records:", len(data))
    for k, v in stats.most_common():
        print(f"{k}: {v}")

    # Print a small sample of bad ids so you can report to team
    print("\nExamples of bad ids (first 10):")
    for k, ids in bad_ids.items():
        if ids:
            print(k, ids[:10])

if __name__ == "__main__":
    main()
