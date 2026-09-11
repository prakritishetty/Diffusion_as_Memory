import datasets

dataset = datasets.load_dataset("./msr_data.py", data_dir="./inputs")

# dataset = datasets.load_dataset(
#     "csv",
#     data_files={
#         "train": "./inputs/train.tsv",
#         "validation": "./inputs/valid.tsv",
#         "test": "./inputs/test.tsv",
#     },
#     delimiter="\t"
# )


dataset["train"].to_csv("outputs/train.csv")
dataset["validation"].to_csv("outputs/validation.csv")
dataset["test"].to_csv("outputs/test.csv")
