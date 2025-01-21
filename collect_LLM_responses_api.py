import argparse
import json
import requests
from tqdm import tqdm
from utils import read_prompts
import os


def generate_response(prompt, api_key, endpoint, model, max_tokens=150, temperature=0.7):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    prompt = f"Proszę odpowiedz w języku polskim: {prompt}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = requests.post(endpoint, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Zbiera odpowiedzi z dowolnego API LLM dla zestawu promptów.")
    parser.add_argument("--api_key", type=str, required=True,
                        help="Klucz API do modelu (np. Llama, OpenAI itd.).")
    parser.add_argument("--endpoint", type=str, required=True,
                        help="Endpoint URL do API modelu (np. https://api.llama-api.com/chat/completions).")
    parser.add_argument("--model", type=str, required=True,
                        help="Nazwa modelu, który ma być używany (np. 'llama3.1-70b').")
    parser.add_argument("--prompts_file", type=str, required=True,
                        help="Ścieżka do pliku CSV z promptami.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Plik wyjściowy (.jsonl), w którym zapiszemy odpowiedzi modelu.")
    parser.add_argument("--max_tokens", type=int, default=150,
                        help="Maksymalna liczba tokenów w odpowiedzi.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperatura generowania (randomness).")
    parser.add_argument("--prompt_limit", type=int, required=False,
                        help="Liczba promptów do przetworzenia przez model.")

    args = parser.parse_args()

    # 1. Wczytanie promptów
    prompts_with_categories = read_prompts(args.prompts_file, limit=args.prompt_limit)
    print(f"> Załadowano {len(prompts_with_categories)} promptów z pliku: {args.prompts_file}")

    # 2. Zebranie odpowiedzi modelu
    response_data = []
    for item in tqdm(prompts_with_categories, desc="Generating Responses"):
        prompt = item["prompt"]
        categories = item["categories"]
        api_response = generate_response(
            prompt,
            api_key=args.api_key,
            endpoint=args.endpoint,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )

        if api_response:
            try:
                model_response = api_response["choices"][0]["message"]["content"]
            except (IndexError, KeyError, TypeError):
                model_response = "Error: Unexpected response structure"
        else:
            model_response = "Error: No response received"

        response_data.append({
            "prompt": prompt,
            "categories": categories,
            "response": model_response
        })

    # 3. Zapis do pliku JSONL
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in response_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"\n> Zapisano odpowiedzi do pliku: {args.output_file}")


if __name__ == "__main__":
    main()
