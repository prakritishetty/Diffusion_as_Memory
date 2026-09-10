import sys
import json

# change this your local path to UniEval.
# where you have cloned the UniEval repository: git clone https://github.com/maszhongming/UniEval.git
UNIEVAL_PATH = "/Users/isheeta.sinha/Documents/uni/cs698ds/UniEval"
sys.path.append(UNIEVAL_PATH)

from utils import convert_to_json
from metric.evaluator import get_evaluator
import nltk

nltk.download("punkt_tab", quiet=True)


def evaluate_task_summarization(src_list, output_list):
    """
    Docstring for evaluate_task_summarization

    In our case:
    src_list: its meant to be source documents, in our case its (x)
    ref_list: its meant to be gold reference summaries, in our case its (x)
    output_list: its meant to be generated summaries, in our case its (v0)

    :param src_list: input memory (x)
    :param output_list: decoded projected latent (v0)
    """
    task = 'summarization'
    data = convert_to_json(output_list=output_list, ref_list=src_list, src_list=src_list)
    evaluator = get_evaluator(task, device='cpu')
    results = evaluator.evaluate(data, print_result=True)

    all_results = [result["consistency"] for result in results]
    mean = sum(all_results) / len(all_results)
    print("Mean Consistency:", mean)
    print(results)

    log_results = {
        "mean_consistency": mean,
        "results": results
    }

    # save to file
    with open("unieval_scores_summarization.json", "w") as f:
        json.dump(log_results, f, indent=4)


def get_src_and_output():
    with open("output/p0/inference/test_preds.json", "r") as f:
        data = json.load(f)
    src_list = []
    output_list = []
    for item in data:
        candidate = item["x_true"]
        src_list.append(candidate)
        reference = item["x_pred"]
        output_list.append(reference)
    
    return src_list, output_list


if __name__ == "__main__":
    src_list, output_list = get_src_and_output()
    evaluate_task_summarization(src_list, output_list)

