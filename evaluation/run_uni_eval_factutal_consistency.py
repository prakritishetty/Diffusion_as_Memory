import sys
import json
import os
import re
import importlib
import importlib.util

# Prefer local clone inside this repo; fallback to shared path if needed.
LOCAL_UNIEVAL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "UniEval")
)
SHARED_UNIEVAL_PATH = "/project/pi_dagarwal_umass_edu/project_3/shared/UniEval"


def _resolve_unieval_path():
    if os.path.isdir(LOCAL_UNIEVAL_PATH):
        return LOCAL_UNIEVAL_PATH
    if os.path.isdir(SHARED_UNIEVAL_PATH):
        return SHARED_UNIEVAL_PATH
    raise FileNotFoundError(
        f"UniEval path not found. Tried: {LOCAL_UNIEVAL_PATH}, {SHARED_UNIEVAL_PATH}"
    )


def _simple_sent_tokenize(text, language="english"):
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _normalize_unieval_inputs(src_list, output_list):
    """Normalize inputs so UniEval sentence splitting never receives empty claims."""
    if len(src_list) != len(output_list):
        raise ValueError(
            f"src_list and output_list must have same length: {len(src_list)} != {len(output_list)}"
        )

    normalized_src = []
    normalized_out = []
    empty_count = 0

    for src, out in zip(src_list, output_list):
        src_text = "" if src is None else str(src)
        out_text = "" if out is None else str(out).strip()

        # UniEval FactEvaluator divides by number of tokenized claim sentences.
        # Ensure at least one minimal sentence for blank outputs.
        if not out_text:
            out_text = "."
            empty_count += 1

        normalized_src.append(src_text)
        normalized_out.append(out_text)

    return normalized_src, normalized_out, empty_count


def _ensure_nltk_sentence_tokenizer():
    try:
        import nltk
    except Exception:
        return

    local_nltk_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".nltk_data")
    )
    os.makedirs(local_nltk_dir, exist_ok=True)
    if local_nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_dir)

    def _has_resource():
        for resource in ("tokenizers/punkt_tab/english", "tokenizers/punkt"):
            try:
                nltk.data.find(resource)
                return True
            except LookupError:
                continue
        return False

    if not _has_resource():
        for pkg in ("punkt_tab", "punkt"):
            try:
                nltk.download(pkg, download_dir=local_nltk_dir, quiet=True)
            except Exception:
                pass

    if not _has_resource():
        try:
            nltk.sent_tokenize = _simple_sent_tokenize
            nltk.tokenize.sent_tokenize = _simple_sent_tokenize
        except Exception:
            pass


def _load_unieval_helpers():
    unieval_path = _resolve_unieval_path()
    _ensure_nltk_sentence_tokenizer()

    # UniEval's evaluator imports `metric.*` and `utils` as top-level modules,
    # so UniEval root must be on sys.path.
    if unieval_path not in sys.path:
        sys.path.insert(0, unieval_path)

    utils_py = os.path.join(unieval_path, "utils.py")
    spec = importlib.util.spec_from_file_location("unieval_utils", utils_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load UniEval utils module from {utils_py}")
    unieval_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unieval_utils)

    # Force evaluator's `from utils import ...` to bind to UniEval utils.
    prev_utils = sys.modules.get("utils")
    sys.modules["utils"] = unieval_utils
    try:
        if "metric.evaluator" in sys.modules:
            del sys.modules["metric.evaluator"]
        metric_evaluator = importlib.import_module("metric.evaluator")
        get_evaluator = metric_evaluator.get_evaluator
    finally:
        if prev_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = prev_utils

    return unieval_utils.convert_to_json, get_evaluator


def evaluate_factual_consistency(src_list, output_list):
    convert_to_json, get_evaluator = _load_unieval_helpers()
    src_list, output_list, empty_count = _normalize_unieval_inputs(src_list, output_list)
    task = 'fact'
    data = convert_to_json(output_list=output_list, src_list=src_list)
    evaluator = get_evaluator(task, device='cpu')
    results = evaluator.evaluate(data, print_result=True)

    all_results = [result["consistency"] for result in results]
    mean = sum(all_results) / len(all_results) if all_results else 0.0
    if empty_count:
        print(f"UniEval note: normalized {empty_count} empty predictions before scoring.")
    print("Mean Consistency:", mean)
    print(results)

    log_results = {
        "mean_consistency": mean,
        "results": results
    }

    # save to file
    with open("unieval_scores.json", "w") as f:
        json.dump(log_results, f, indent=4)


def evaluate_factual_consistency_return(src_list, output_list):
    convert_to_json, get_evaluator = _load_unieval_helpers()
    src_list, output_list, empty_count = _normalize_unieval_inputs(src_list, output_list)
    task = 'fact'
    data = convert_to_json(output_list=output_list, src_list=src_list)
    evaluator = get_evaluator(task, device='cpu')
    results = evaluator.evaluate(data, print_result=False)

    all_results = [result["consistency"] for result in results]
    mean = sum(all_results) / len(all_results) if all_results else 0.0
    if empty_count:
        print(f"UniEval note: normalized {empty_count} empty predictions before scoring.", flush=True)
    return {
        "mean_consistency": mean,
        "results": results
    }

def get_src_and_output(file_path, ground_label_key="x_true", prediction_key="x_pred"):
    with open(file_path, "r") as f:
        data = json.load(f)
    src_list = []
    output_list = []
    for item in data:
        candidate = item[ground_label_key]
        src_list.append(candidate)
        reference = item[prediction_key]
        output_list.append(reference)
    
    return src_list, output_list


if __name__ == "__main__":
    src_list, output_list = get_src_and_output("output/p0/inference/test_preds.json", ground_label_key="x_true", prediction_key="x_pred")
    evaluate_factual_consistency(src_list, output_list)
