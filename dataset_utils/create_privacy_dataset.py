import os
import json
import torch
import pandas as pd
import wandb
from datasets import load_dataset
from litellm import completion
from tqdm import tqdm

# === CONFIGURATION ===
MODEL = "openai/gpt4o"
API_KEY = "sk-AXYHuFT9GozKcTi1q8m1nw"
API_BASE = "https://thekeymaker.umass.edu/"

NUM_SAMPLES = 500
BATCH_SIZE = 5
WANDB_PROJECT = "text_denoising_diffusion"
WANDB_NAME = "privacy_abstraction_gpt4o"
OUTPUT_FILE = "privasis_gpt4o_abstraction.json"

SYSTEM_PROMPT = """You are an expert in privacy and semantic abstraction. 
Your task is to take a specific sentence (x) and generate N levels of increasing abstraction (xt), where N is determined by the number of distinct PII categories or specific details in the sentence.

Rules for generating xt:
1. Identify all specific details: Names, Locations, Organizations, Dates, IDs, and other identifying information.
2. Create a sequence where each step removes exactly ONE category or group of related details by replacing them with more general terms.
3. The categories should be consistent:
   - Level 1: Usually remove the most specific identifiers (e.g., Names).
   - Level 2: Remove next level (e.g., specific Locations or Dates).
   - ... and so on.
4. The final level in xt should be an extremely generic statement that barely preserves the intent (e.g., 'I went out' or 'Someone did something').
5. N should be at least 5 and can be up to 9 or 10 depending on complexity.

Example:
x: 'John Doe from Google visited Paris on Jan 1st.'
xt: [
  'A person from Google visited Paris on Jan 1st.',
  'A person from a company visited Paris on Jan 1st.',
  'A person visited a city in the winter.',
  'Someone traveled recently.',
  'A person went somewhere.'
]

Return the response strictly in the provided JSON schema."""

JSON_SCHEMA = {
    "name": "abstraction_steps_dataset",
    "schema": {
        "type": "object",
        "properties": {
            "response": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "x": {"type": "string"},
                        "xt": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of N abstraction levels, from slightly general to extremely generic."
                        },
                    },
                    "required": ["id", "x", "xt"],
                }
            } 
        },
        "required": ["response"],
        "additionalProperties": False  
    }
}

def generate_abstractions_batch(batch_data, retries=3):
    input_text = json.dumps(batch_data)
    for attempt in range(retries):
        try:
            response = completion(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Process these samples: {input_text}"}
                ],
                response_format={"type": "json_schema", "json_schema": JSON_SCHEMA, "strict": True},
                api_base=API_BASE,
                api_key=API_KEY,
                temperature=0.2,
                timeout=120, # Increased timeout
            )
            content = response.choices[0].message.content
            return json.loads(content)["response"]
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                import time
                time.sleep(2 ** attempt) # Exponential backoff
            else:
                raise e

def main():
    wandb.init(project=WANDB_PROJECT, name=WANDB_NAME)
    
    print("Loading Privasis dataset...")
    dataset = load_dataset('nvidia/Privasis-Zero', 'corpus', split='train', streaming=True)
    
    all_processed_data = []
    # Try to load existing progress
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            all_processed_data = json.load(f)
        print(f"Resuming from {len(all_processed_data)} existing samples.")
    
    current_batch = []
    processed_count = len(all_processed_data)
    
    print(f"Processing {NUM_SAMPLES} samples in batches of {BATCH_SIZE}...")
    
    for i, example in enumerate(tqdm(dataset, total=NUM_SAMPLES)):
        if i < processed_count:
            continue
        if i >= NUM_SAMPLES:
            break
            
        current_batch.append({
            "id": str(i),
            "x": example['record'].strip()
        })
        
        if len(current_batch) == BATCH_SIZE:
            try:
                processed_batch = generate_abstractions_batch(current_batch)
                all_processed_data.extend(processed_batch)
                
                # Save progress after every batch
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(all_processed_data, f, indent=2)
            except Exception as e:
                print(f"Batch failed after retries, skipping: {e}")
            
            current_batch = []
        
        if current_batch:
            print(f"Processing final batch of {len(current_batch)}...")
            process_and_save_batch(current_batch, all_processed_data, OUTPUT_FILE)

    # Final Log to W&B
    df_rows = []
    for item in all_processed_data:
        row = {"original": item["x"]}
        for idx, level_text in enumerate(item["xt"]):
            row[f"level_{idx+1}"] = level_text
        df_rows.append(row)
    
    df = pd.DataFrame(df_rows)
    table = wandb.Table(dataframe=df)
    wandb.log({"privacy_abstraction_samples": table})
    
    print(f"Done! Total {len(all_processed_data)} samples saved to {OUTPUT_FILE}.")
    wandb.finish()

def process_and_save_batch(batch, data_list, filename):
    try:
        processed_batch = generate_abstractions_batch(batch)
        data_list.extend(processed_batch)
        with open(filename, "w") as f:
            json.dump(data_list, f, indent=2)
    except Exception as e:
        print(f"Batch failed: {e}")

if __name__ == "__main__":
    main()
