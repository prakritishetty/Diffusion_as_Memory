import json
from bert_score import BERTScorer
import logging
import transformers
from transformers import pipeline
from tqdm import tqdm


transformers.logging.set_verbosity_error()

scorer = BERTScorer(
    model_type="roberta-large",
    lang="en",
    rescale_with_baseline=True,
    use_fast_tokenizer=True,
)
logging.getLogger("bert_score").setLevel(logging.ERROR)

print(scorer._tokenizer.__class__.__name__)

nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")

def compute_bert_score(src, output):
    P, R, F1 = scorer.score([src], [output])
    return R.mean().item()

def is_monotonically_decreasing(lst):
    tolerance = 1
    return all(x >= y for x, y in zip(lst, lst[1:]))

def evaluate_bert_score(sample):
    """
    For each sample, generate the BERTScore of each degraded text against the original text.
    For a single score for a sample: check if monotonically decreases.
    """
    # for sample in tqdm(data):
    original_text = sample["original_text"]
    decoded_xt_list = sample["decoded_xt"][:4]
    bert_scores = []
    for t_value, decoded_xt in decoded_xt_list:
        bert_score = compute_bert_score(original_text, decoded_xt)
        bert_scores.append((t_value, bert_score))
    
    # Check if bert_scores are monotonically decreasing with respect to t_value
    bert_scores.sort(key=lambda x: x[0])  # Sort by t_value
    scores_only = [bert_score for _, bert_score in bert_scores]  
    result = {
        "bert_scores": bert_scores,
        "is_monotonic": is_monotonically_decreasing(scores_only)
    }
    return result



def check_entailment(premise, hypothesis):
    result = nli(f"{premise} [SEP] {hypothesis}", truncation=True)
    return result[0]["label"], result[0]["score"]


def evaluate_entailement(sample):
    """
    
    """
    sentences = [x[1] for x in sample["decoded_xt"]][:4]
    metadata = []
    count_detail_loss = 0
    for i in range(len(sentences) - 1):
        s_i, s_next = sentences[i], sentences[i + 1]
        fwd_label, fwd_score = check_entailment(s_i, s_next)
        bwd_label, bwd_score = check_entailment(s_next, s_i)
        
        is_detail_loss = (fwd_label == "entailment" or fwd_label == "neutral") and bwd_label != "entailment"
        # print(fwd_label, bwd_label)
        
        if is_detail_loss:
            count_detail_loss += 1
        
        metadata.append({
            "from": i, "to": i + 1,
            "forward": (fwd_label, round(fwd_score, 3)),
            "backward": (bwd_label, round(bwd_score, 3)),
            "is_detail_loss": is_detail_loss,
            # Asymmetry score: how "one-directional" the entailment is
            "asymmetry": fwd_score - bwd_score if fwd_label == "entailment" else 0.0
        })
    
    total_detail_loss = count_detail_loss / (len(sentences) - 1) if len(sentences) > 1 else 0.0
    return {
        "metadata": metadata,
        "total_detail_loss": round(total_detail_loss, 3)
    }

def calculate(data):
    results = {}
    results["total_monotonic_decrease_score"] = 0
    results["total_entailment_score"] = 0
    
    total_monotonic_decrease_count = 0
    total_entailment_score = 0
    
    for sample in tqdm(data):
        try:
            entailment_res = evaluate_entailement(sample)
            bert_score = evaluate_bert_score(sample)
            batch_idx = sample["batch_idx"]
            sample_idx = sample["sample_idx"]
            results[f"{batch_idx}_{sample_idx}"] = {
                "entailment_evaluation": entailment_res,
                "bert_score_evaluation": bert_score
            }
            
            if bert_score["is_monotonic"] == True:
                total_monotonic_decrease_count += 1
            total_entailment_score += entailment_res["total_detail_loss"]
        except Exception as e:
            print(f"Error processing sample {sample['batch_idx']}_{sample['sample_idx']}: {e}")
        
    print(f"Monotonic Decrease Score: {total_monotonic_decrease_count}")    
    
    results["total_monotonic_decrease_score"] = round(total_monotonic_decrease_count / len(data), 3)
    results["total_entailment_score"] = round(total_entailment_score / len(data), 3)
    return results


def main():
    file_path = "/project/pi_dagarwal_umass_edu/project_3/issinha/output/inference_results_one_step_estimate_joint.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # results = evaluate_bert_score(data)
    results = calculate(data)
    
    results_path = "/project/pi_dagarwal_umass_edu/project_3/issinha/output/evaluation_onestep_joint.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
