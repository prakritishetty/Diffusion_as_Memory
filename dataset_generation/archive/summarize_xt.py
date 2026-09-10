import json
import torch
import transformers
from tqdm import tqdm
import sys
import os

class XtGenerator:
    
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
                "content": """We are trying to model "forgetting" of a sentence (memory item) by forgetting episodic aspects of the memory item (conscious memory) over time t.

Your job is to create a list of  sentences, such that it contains just this information:
1. Progressive detail attenuation. As t increases, fine-grained specifics like numbers, named entities, dosages, or 
timestamps should fade earlier than the core semantic structure.
2. Monotonic information loss. Once a detail disappears at a higher t, it should not reappear at an even higher t
3. t increases in discrete steps, and with each step, the sentence should become more abstract and less specific, while still retaining the core meaning.
4. t increases as xt list grows longer

Do not return any reasoning or extra phrases or filler sentences. Just return a list that represents the Progressive detail attenuation(xt).

Example-1:
Text(x): I went swimming last Sunday
Progressive detail attenuation(xt): ['I went swimming last weekend','I did sports last weekend',
'I went out weeks ago', 'I did something last weekend', 'I went out']

Example-2:
Text(x): The patient was prescribed 5mg of DrugX daily for hypertension.
Progressive detail attenuation(xt): ['The patient was prescribed a medication daily for hypertension.', 
'The patient was prescribed a medication for hypertension.', 'The patient was prescribed something for hypertension.', 
'The patient was prescribed something.', 'The patient was prescribed something for a condition.'] 
"""
            },
            {
                "role": "user",
                "content": f"Text(x): {text}\n\nProgressive detail attenuation(xt):"
            }
        ]
        return messages
    
    def generate_xt(self, text, max_new_tokens=150):
        """Generate progressive detail attenuation (xt) using Gemma with chat template and system prompt"""
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
        
        xt = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Raw model output: {xt}", file=sys.stderr)
        # Extract only the model's response after the chat template
        if "\nmodel\n" in xt:
            xt = xt.split("\nmodel\n")[-1].strip()
        else:
            xt = xt.split("<end_of_turn>")[-1].strip()
        
        return xt
    
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
        
       
        for idx in tqdm(range(start_idx, len(data)), desc="Generating xt", initial=start_idx):
            record = data[idx]
            
            try:
                xt = self.generate_xt(record['x'])
                record['xtm'] = xt
                record['xtm_processed'] = True
                
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
                record['xtm'] = ""
                record['xtm_processed'] = False
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
    output_file = "/project/pi_dagarwal_umass_edu/project_3/bdevarangadi/Data/Processed/train_with_xt_gemma.json"
    checkpoint_file = "/project/pi_dagarwal_umass_edu/project_3/bdevarangadi/Data/Processed/.xt_checkpoint_gemma.json"
    
    generator = XtGenerator()
    generator.process_file(input_file, output_file, checkpoint_file=checkpoint_file)


if __name__ == "__main__":
    main()
