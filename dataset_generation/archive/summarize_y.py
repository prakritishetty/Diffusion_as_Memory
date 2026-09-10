import json
import torch
import transformers
from tqdm import tqdm
import sys
import os

class LlamaSummarizer:
    
    def __init__(self, cache_dir="/datasets/ai/gemma/hub"):
        """Initialize model and tokenizer"""
        print("Loading google/gemma-3-12b-it model...", file=sys.stderr)
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-12b-it",
            torch_dtype=torch.bfloat16,
            device_map='cuda',
            cache_dir=cache_dir
        )
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "google/gemma-3-12b-it",
            cache_dir=cache_dir
        )
        
        print("Model loaded successfully!", file=sys.stderr)
    
    def create_messages(self, text):
        """Create messages list with system prompt using chat template"""
        messages = [
            {
                "role": "system",
                "content": """We are trying to model "forgetting" of a sentence (memory item) by forgetting episodic aspects of the memory item (conscious memory) and just retaining the semantics (subconscious memory) of it.

Your job is to create a gist of the given sentence, such that it contains just this information:
1. Semantic memory (key facts and concepts)
2. Schemas (underlying patterns and structures)
3. Affective state (emotional tone and sentiment)
4. Identity (who/what is being discussed)
Make sure that this gist does NOT contain any episodic or declarative memory content
Limit the gist to roughly about half the length of the sentence for conciseness
Do not return any reasoning or extra phrases or filler sentences. Just return a string that represents the gist.

Example:
Text: Except for this small vocal minority, we have just not gotten a lot of groundswell against this from members,  says APA president Philip G. Zimbardo of Stanford University.
Gist: minor opposition against this."""
            },
            {
                "role": "user",
                "content": f"Text: {text}\n\nGist:"
            }
        ]
        return messages
    
    def generate_summary(self, text, max_new_tokens=150):
        """Generate summary using Gemma with chat template and system prompt"""
        messages = self.create_messages(text)
        
        # Apply chat template to format messages properly
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the model's response after the chat template
        if "\nmodel\n" in summary:
            summary = summary.split("\nmodel\n")[-1].strip()
        else:
            summary = summary.split("<end_of_turn>")[-1].strip()
        
        return summary
    
    def process_file(self, input_file, output_file, checkpoint_file=None, start_idx=0):
        """
        Process JSON file with checkpointing support
        
        Args:
            input_file: Path to input JSON
            output_file: Path to output JSON
            checkpoint_file: Path to checkpoint (optional)
            start_idx: Index to resume from
        """
        
        print(f"Loading {input_file}...", file=sys.stderr)
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total records: {len(data)}", file=sys.stderr)
        
        if checkpoint_file and os.path.exists(checkpoint_file):
            print(f"Loading checkpoint from {checkpoint_file}...", file=sys.stderr)
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                data = checkpoint['data']
                start_idx = checkpoint['last_idx']
            print(f"Resuming from index {start_idx}", file=sys.stderr)
        
       
        for idx in tqdm(range(start_idx, len(data)), desc="Summarizing", initial=start_idx):
            record = data[idx]
            
            try:
                summary = self.generate_summary(record['x'])
                record['summary'] = summary
                record['summary_processed'] = True
                
                if checkpoint_file and (idx + 1) % 10 == 0:
                    checkpoint_data = {
                        'data': data,
                        'last_idx': idx + 1,
                        'total': len(data)
                    }
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"\nCheckpoint saved at record {idx + 1}", file=sys.stderr)
            
            except Exception as e:
                print(f"\nError processing record {idx}: {str(e)}", file=sys.stderr)
                record['summary'] = ""
                record['summary_processed'] = False
                continue
        
        print(f"\nSaving to {output_file}...", file=sys.stderr)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        if checkpoint_file and os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("Checkpoint cleaned up", file=sys.stderr)
        
        print(f"Completed! Output saved to {output_file}", file=sys.stderr)
        return data


def main():
    """Main execution with options"""
    input_file = "/project/pi_dagarwal_umass_edu/project_3/bdevarangadi/Data/Processed/train_parsed_v2.0.json"
    output_file = "/project/pi_dagarwal_umass_edu/project_3/bdevarangadi/Data/Processed/train_with_summaries_gemma_better_prompt.json"
    checkpoint_file = "/project/pi_dagarwal_umass_edu/project_3/bdevarangadi/Data/Processed/.summarization_checkpoint_gemma_better_prompt.json"
    
    summarizer = LlamaSummarizer()
    summarizer.process_file(input_file, output_file, checkpoint_file=checkpoint_file)


if __name__ == "__main__":
    main()
