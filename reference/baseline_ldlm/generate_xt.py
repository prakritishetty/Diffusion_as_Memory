import os, json
from litellm import completion
from system_prompt import SYSTEM_PROMPT as SYSTEM_PROMPT
import itertools

MODEL = "openai/gpt4o"
api_key = "api-key"

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
                            "items": {"type": "string"}
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

def get_user_input(sentence):
    user_input = f"Sentence: {sentence}"
    return user_input


def generate_abstractions(input_json_data) -> dict:
    response = completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_json_data}
        ],
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA, "strict": True},
        api_base="https://thekeymaker.umass.edu/",
        api_key=api_key,
        temperature=0.2,
    )
    content = response.choices[0].message.content
    return json.loads(content)


def get_input_data(input_jsonl_file, start_index, end_index):
    with open(input_jsonl_file, "r") as f:
        sliced_lines = list(itertools.islice(f, start_index, end_index))
    return "".join(sliced_lines)


if __name__ == "__main__":
    input_jsonl_file = "../../input_x.jsonl"
    start = 1140
    batches = 10
    last = 1150
    for i in range(start, last, batches):
        try:
            print(f"Processing lines {i} to {i+batches}...")
            input_data = get_input_data(input_jsonl_file, i, i+batches)
            # print(input_data)
            response = generate_abstractions(input_data)
            with open(f"../../gpt4_final_outputs/output_x_{i}.json", "w") as f:
                json.dump(response, f, indent=2)
            print("_________________________________")
        except Exception as e:
            print(f"Error processing lines {i} to {i+batches}: {e}")
