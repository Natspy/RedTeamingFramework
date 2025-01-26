import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from utils import read_prompts
from transformers.utils.logging import disable_progress_bar
import os


def load_model(hf_token, model_name):
    os.environ["BNB_4BIT_COMPUTE_TYPE"] = "float16"
    login(token=hf_token)
    disable_progress_bar()
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    return tokenizer, model


def generate_model_response(prompt, tokenizer, model, max_new_tokens, temperature, top_k, repetition_penalty):
    """
    Generate responses for loaded prompts
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def preprocess_response(prompt, response):
    """
    Remove the prompt from the beginning of the response if it matches.
    Also, ensure the response does not start with a white space, comma, or period.
    """
    stripped_prompt = prompt.strip()
    stripped_response = response.strip()

    if stripped_response.startswith(stripped_prompt):
        stripped_response = stripped_response[len(stripped_prompt):].lstrip(" ,.")

    return stripped_response


def main():
    parser = argparse.ArgumentParser(description="Collect responses using a HuggingFace model (e.g., Bielik).")
    parser.add_argument("--hf_token", type=str, required=True,
                        help="HuggingFace API token for loading the model.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="The name of the HuggingFace model to load.")
    parser.add_argument("--prompts_file", type=str, required=True,
                        help="Path to the CSV file containing prompts and categories.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the generated responses in JSONL format.")
    parser.add_argument("--prompt_limit", type=int, required=False,
                        help="Number of prompts to process.")
    parser.add_argument("--max_new_tokens", type=int, default=150,
                        help="Maximum number of tokens to generate in the response.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-K sampling value.")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Repetition penalty.")

    args = parser.parse_args()

    # Load the model and tokenizer
    tokenizer, model = load_model(args.hf_token, args.model_name)

    # Read prompts from the CSV file
    prompts_with_categories = read_prompts(args.prompts_file, limit=args.prompt_limit)
    print(f"> Załadowano {len(prompts_with_categories)} promptów z pliku: {args.prompts_file}")

    # Generate responses
    response_data = []
    for item in tqdm(prompts_with_categories, desc="Generating Responses"):
        prompt = item["prompt"]
        categories = item["categories"]
        model_response = generate_model_response(
            prompt,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty
        )

        response_data.append({
            "prompt": prompt,
            "categories": categories,
            "response": model_response
        })

    # Save responses to JSONL
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in response_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    processed_data = []

    with open(args.output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Process each line
    for i, line in enumerate(lines):
        record = json.loads(line.strip())
        prompt = record.get("prompt", "")
        response = record.get("response", "")

        # Apply the preprocessing function
        processed_response = preprocess_response(prompt, response)

        # Update the record with the processed response
        record["response"] = processed_response
        processed_data.append(record)

    # Save the processed data back to a new JSONL file
    with open(args.output_file, "w", encoding="utf-8") as f:
        for record in processed_data:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    print(f"\n> Zapisano odpowiedzi do pliku: {args.output_file}")


if __name__ == "__main__":
    main()
