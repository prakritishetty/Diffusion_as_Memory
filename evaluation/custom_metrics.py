import json
import os
import re
import sys
import torch
import logging
import importlib.util
from tqdm import tqdm
from bert_score import BERTScorer, score
import transformers
from transformers import pipeline
import nltk

# Suppress warnings
transformers.logging.set_verbosity_error()
logging.getLogger("bert_score").setLevel(logging.ERROR)

class CustomEvaluator:
    def __init__(self, unieval_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.unieval_path = unieval_path or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "UniEval"))
        
        # Initialize BERTScorer with specific device
        self.bert_scorer = BERTScorer(
            model_type="roberta-large",
            lang="en",
            rescale_with_baseline=True,
            use_fast_tokenizer=True,
            device=self.device
        )
        
        # Initialize NLI pipeline
        self.nli = pipeline(
            "text-classification", 
            model="cross-encoder/nli-deberta-v3-base", 
            device=0 if "cuda" in self.device else -1
        )
        
        # Setup UniEval if path exists
        self.has_unieval = os.path.exists(self.unieval_path)
        if self.has_unieval:
            self._setup_unieval()
        else:
            print(f"Warning: UniEval path not found at {self.unieval_path}. Factual consistency scores will be skipped.")

    def _setup_unieval(self):
        if self.unieval_path not in sys.path:
            sys.path.insert(0, self.unieval_path)
        
        nltk.download("punkt_tab", quiet=True)
        
        try:
            from utils import convert_to_json
            from metric.evaluator import get_evaluator
            self.convert_to_json = convert_to_json
            self.get_evaluator = get_evaluator
        except ImportError:
            self.has_unieval = False
            print("Failed to import UniEval components. Check path and dependencies.")

    def compute_bert_score_single(self, src, output):
        # Using the instance's scorer is faster
        P, R, F1 = self.bert_scorer.score([src], [output])
        return R.mean().item()

    def is_monotonically_decreasing(self, lst):
        # Allow small tolerance for noise
        return all(x >= y for x, y in zip(lst, lst[1:]))

    def evaluate_forgetting_trajectory(self, original_text, decoded_xt_list):
        """
        decoded_xt_list: list of (t_value, decoded_text)
        """
        bert_scores = []
        for t_value, decoded_xt in decoded_xt_list:
            score = self.compute_bert_score_single(original_text, decoded_xt)
            bert_scores.append((t_value, score))
        
        # Sort by t_value
        bert_scores.sort(key=lambda x: x[0])
        scores_only = [s for _, s in bert_scores]
        
        return {
            "bert_scores": bert_scores,
            "is_monotonic": self.is_monotonically_decreasing(scores_only)
        }

    def check_entailment(self, premise, hypothesis):
        result = self.nli(f"{premise} [SEP] {hypothesis}", truncation=True)
        return result[0]["label"], result[0]["score"]

    def evaluate_entailment_asymmetry(self, sentences):
        """
        sentences: list of decoded abstractions from specific to generic
        """
        metadata = []
        count_detail_loss = 0
        for i in range(len(sentences) - 1):
            s_i, s_next = sentences[i], sentences[i + 1]
            fwd_label, fwd_score = self.check_entailment(s_i, s_next)
            bwd_label, bwd_score = self.check_entailment(s_next, s_i)
            
            # Detail loss: L(i) entails L(i+1), but L(i+1) does NOT entail L(i)
            is_detail_loss = (fwd_label == "entailment" or fwd_label == "neutral") and bwd_label != "entailment"
            
            if is_detail_loss:
                count_detail_loss += 1
            
            metadata.append({
                "from": i, "to": i + 1,
                "forward": (fwd_label, round(fwd_score, 3)),
                "backward": (bwd_label, round(bwd_score, 3)),
                "is_detail_loss": is_detail_loss,
                "asymmetry": fwd_score - bwd_score if fwd_label == "entailment" else 0.0
            })
        
        total_detail_loss = count_detail_loss / (len(sentences) - 1) if len(sentences) > 1 else 0.0
        return {
            "metadata": metadata,
            "total_detail_loss": round(total_detail_loss, 3)
        }

    def evaluate_factual_consistency(self, src_list, output_list):
        if not self.has_unieval:
            return {"mean_consistency": 0.0, "error": "UniEval not found"}
        
        task = 'fact'
        data = self.convert_to_json(output_list=output_list, src_list=src_list)
        evaluator = self.get_evaluator(task, device=self.device)
        results = evaluator.evaluate(data, print_result=False)
        
        all_results = [result["consistency"] for result in results]
        mean = sum(all_results) / len(all_results) if all_results else 0.0
        return {"mean_consistency": mean, "results": results}

    def evaluate_summarization_intent(self, src_list, output_list):
        """
        Evaluate if the abstraction preserves the intent of the original.
        """
        if not self.has_unieval:
            return {"mean_consistency": 0.0, "error": "UniEval not found"}
            
        task = 'summarization'
        data = self.convert_to_json(output_list=output_list, ref_list=src_list, src_list=src_list)
        evaluator = self.get_evaluator(task, device=self.device)
        results = evaluator.evaluate(data, print_result=False)
        
        all_results = [result["consistency"] for result in results]
        mean = sum(all_results) / len(all_results) if all_results else 0.0
        return {"mean_consistency": mean, "results": results}

def run_full_evaluation(json_results_path, output_metrics_path, device=None):
    with open(json_results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    evaluator = CustomEvaluator(device=device)
    results = {}
    
    total_monotonic_count = 0
    total_entailment_score = 0
    
    for sample in tqdm(data):
        try:
            # Assuming sample structure from implementation plan
            original_text = sample["original_text"]
            decoded_xt = sample["decoded_xt"] # list of [t, text]
            
            entailment_res = evaluator.evaluate_entailment_asymmetry([x[1] for x in decoded_xt])
            forgetting_res = evaluator.evaluate_forgetting_trajectory(original_text, decoded_xt)
            
            sample_id = f"{sample.get('batch_idx', 'b')}_{sample.get('sample_idx', 's')}"
            results[sample_id] = {
                "entailment": entailment_res,
                "forgetting": forgetting_res
            }
            
            if forgetting_res["is_monotonic"]:
                total_monotonic_count += 1
            total_entailment_score += entailment_res["total_detail_loss"]
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            
    summary = {
        "mean_monotonic_score": round(total_monotonic_count / len(data), 3) if data else 0,
        "mean_entailment_score": round(total_entailment_score / len(data), 3) if data else 0,
        "detailed_results": results
    }
    
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    
    print(f"Evaluation complete. Summary: {summary}")
    return summary

if __name__ == "__main__":
    # Test path
    run_full_evaluation("inference_results.json", "metrics_summary.json")
