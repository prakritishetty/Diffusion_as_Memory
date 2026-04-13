import json
from bert_score import score
import logging
import transformers

transformers.logging.set_verbosity_error()


logging.getLogger("bert_score").setLevel(logging.ERROR)

def compute_bert_score(src, output):
    P, R, F1 = score(
        [src],
        [output],
        lang="en",
        verbose=False
    )
    return F1.mean().item()

def is_monotonically_decreasing(lst):
    tolerance = 1
    return all(x >= y - tolerance for x, y in zip(lst, lst[1:]))

def evaluate_bert_score(data):
    """
    For each sample, generate the BERTScore of each degraded text against the original text.
    For a single score for a sample: check if monotonically decreases.
    """
    results = {}
    monotonic_decrease_false_count = 0
    results["monotonic_decrease_fail_count"] = monotonic_decrease_false_count
    results["total_samples"] = len(data)
    for sample in data:
        original_text = sample["original_text"]
        decoded_xt_list = sample["decoded_xt"]
        bert_scores = []
        for t_value, decoded_xt in decoded_xt_list:
            bert_score = compute_bert_score(original_text, decoded_xt)
            bert_scores.append((t_value, bert_score))
        
        # Check if bert_scores are monotonically decreasing with respect to t_value
        bert_scores.sort(key=lambda x: x[0])  # Sort by t_value
        scores_only = [bert_score for _, bert_score in bert_scores]
        if not is_monotonically_decreasing(scores_only):
            print(f"Sample {sample['sample_idx']} does not have monotonically decreasing BERTScore.")
            monotonic_decrease_false_count += 1
        
        print(f"Sample {sample['sample_idx']} BERTScores: {[x[1] for x in bert_scores]}")

        batch_idx = sample["batch_idx"]
        sample_idx = sample["sample_idx"]        
        results[f"{batch_idx}_{sample_idx}"] = {
            "bert_scores": bert_scores,
            "is_monotonic": is_monotonically_decreasing(scores_only)
        }
    
    results["monotonic_decrease_fail_count"] = monotonic_decrease_false_count
    results["total_samples"] = len(data)
    return results


def evaluate_entailement(src_list, output_list):
    pass

def main():
    file_path = "/project/pi_dagarwal_umass_edu/project_3/issinha/output/inference_results_p2_final.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = evaluate_bert_score(data)
    
    results_path = "/project/pi_dagarwal_umass_edu/project_3/issinha/output/evaluation_results_p2_final.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
