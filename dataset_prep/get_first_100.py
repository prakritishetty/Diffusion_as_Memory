import json

with open("train_with_summaries_gemma_better_prompt.json", "r") as f:
    data = json.load(f)  

first_100 = data[:100]
second_100 = data[101:200]

with open("outputs/first_100_train.json", "w") as f:
    json.dump(first_100, f, indent=2)
with open("outputs/first_100_test.json", "w") as f:
    json.dump(second_100, f, indent=2)