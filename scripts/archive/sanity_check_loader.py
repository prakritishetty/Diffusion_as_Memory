import os
from src.data.msr_datamodule import MSRGistDataModule

def main():
    # You should export MODEL_DIR before running this (shown in commands below)
    model_dir = os.environ.get("MODEL_DIR", "")
    if not model_dir:
        raise RuntimeError("MODEL_DIR env var not set. Export it to the gemma snapshot dir first.")

    dm = MSRGistDataModule(
        train_path="data/processed/train_part2_clean.json",
        tokenizer_name="bert-base-uncased",
        batch_size=2,
        max_length=128,
        num_workers=0,
        y_key="summary",         # change to "y" if your cleaned file uses y
        include_xt=False,

        make_xplus=True,
        gemma_model_dir=model_dir,
    )

    dm.prepare_data()
    dm.setup()

    # inside main(), after dm.setup()

    dl = dm.train_dataloader()

    num_batches = 0
    num_examples = 0

    for batch_idx, batch in enumerate(dl):
        num_batches += 1
        # batch["x_input_ids"] is [B, L]
        bsz = batch["x_input_ids"].shape[0]
        num_examples += bsz

        # Print a couple batches only (otherwise it will spam)
        if batch_idx < 2:
            print("\nBatch", batch_idx)
            print("x_input_ids:", batch["x_input_ids"].shape)
            print("xplus_input_ids:", batch["xplus_input_ids"].shape)
            print("labels:", batch["labels"].shape)

            for i in range(min(2, bsz)):
                print("\n--- sample", i, "---")
                print("id:", batch["id"][i])
                print("x:", batch["raw_x"][i])
                print("x+:", batch["raw_x_plus"][i])

        # progress print every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"processed {batch_idx+1} batches ({num_examples} examples)")

    print(f"\nEpoch done: {num_batches} batches, {num_examples} examples total")


if __name__ == "__main__":
    main()
