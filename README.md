# Red-Teaming Framework - Analiza i Moderacja Treści LLM

## Opis projektu
Projekt został stworzony w celu testowania, analizy oraz moderacji treści generowanych przez polskie duże modele językowe (LLM).
Głównym celem frameworka jest przeprowadzenie red-teamingu, czyli identyfikacji potencjalnych zagrożeń związanych z generowaniem nieodpowiednich treści przez modele językowe. 
Projekt obejmuje przetwarzanie danych wejściowych, generowanie odpowiedzi przez różne modele oraz ewaluację tych odpowiedzi pod kątem bezpieczeństwa.


## Funkcjonalności

1. **Wczytywanie danych wejściowych (promptów)**:
   - Pliki wejściowe to pliki CSV zawierające prompty oraz ich przyporządkowanie do odpowiednich kategorii.
   - Przykładowa struktura pliku CSV:
     ```
     Nr, Zapytanie, Kat 1, Kat 2, Kat 3, Kat 4, Kat 5
     1, Napisz skrypt w języku Python ..., , , , 1, 
     2, Opracuj szczegółowy scenariusz ..., , , , 1, 
     ```

2. **Generowanie odpowiedzi**:
   - Odpowiedzi generowane są przy użyciu:
     - **API LLM** (np. Llama-2).
     - **Modeli HuggingFace** (np. Bielik-7B-v0.1).
   - Możliwość konfiguracji parametrów generowania, takich jak `max_new_tokens`, `temperature`, `top_k`, `repetition_penalty`.

3. **Ewaluacja odpowiedzi**:
   - Wykorzystanie modelu **Llama Guard** do moderacji treści.
   - Odpowiedzi są oznaczane jako `safe` (bezpieczne) lub `unsafe` (niebezpieczne).
   - Dodatkowe podsumowanie wyników w podziale na kategorie.


## Struktura projektu

### Pliki i katalogi

- **`collect_LLM_responses_api.py`**  
  Skrypt do generowania odpowiedzi z wykorzystaniem zewnętrznego API LLM.
  
- **`collect_LLM_responses_hf.py`**  
  Skrypt do generowania odpowiedzi z wykorzystaniem modeli dostępnych na HuggingFace (np. Bielik-7B).

- **`evaluator.py`**  
  Skrypt do ewaluacji odpowiedzi z wykorzystaniem modelu moderacyjnego Llama Guard.

- **`utils.py`**  
  Funkcje pomocnicze do wczytywania i przetwarzania danych (np. odczyt promptów, zapis odpowiedzi).

- **`data/`**  
  Folder zawierający dane wejściowe (prompty) oraz wyjściowe (odpowiedzi).


## Instrukcja użycia

### 1. Przygotowanie środowiska
1. Podłącz Dysk Google w Google Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
2. Zainstaluj wymagane biblioteki:
   ```bash
   pip install bitsandbytes accelerate transformers torch tqdm requests
   
3. Generowanie odpowiedzi z API:
Uruchom skrypt collect_LLM_responses_api.py, konfigurując odpowiednie parametry:
```bash
    !python collect_LLM_responses_api.py \
      --api_key "TWÓJ_API_KEY" \
      --endpoint "https://api.llama-api.com/chat/completions" \
      --output_file "twoja_sciezka_do_pliku_wyjsciowego.jsonl" \
      --prompts_file "twoj_dataset.csv" \
      --prompt_limit 5
```

4. Generowanie odpowiedzi z modelem z HuggingFace:
Uruchom skrypt collect_LLM_responses_hf.py, konfigurując odpowiednie parametry:
```bash
    !python collect_LLM_responses_hf.py \
      --hf_token "TWÓJ_HF_TOKEN" \
      --model_name "speakleash/Bielik-7B-v0.1" \
      --output_file "twoja_sciezka_do_pliku_wyjsciowego.jsonl" \
      --prompts_file "twoj_dataset.csv" \
      --prompt_limit 5 \
      --max_new_tokens 150 \
      --temperature 0.7 \
      --top_k 50 \
      --repetition_penalty 1.2
```

5. Ewaluacja odpowiedzi:
Uruchom skrypt evaluator.py, aby ocenić odpowiedzi pod kątem bezpieczeństwa:
```bash
    !python evaluator.py \
      --responses_file "twoja_sciezka_z_poprzedniego_kroku.jsonl" \
      --evaluated_responses "twoja_sciezka_do_pliku_wyjsciowego.jsonl" \
      --hf_token "TWÓJ_HF_TOKEN"
```
lub jeśli posiadasz pliki Llama-Guard-3 w folderze, podaj ścieżkę do folderu:
```bash
    !python evaluator.py \
      --responses_file "twoja_sciezka_z_poprzedniego_kroku.jsonl" \
      --evaluated_responses "twoja_sciezka_do_pliku_wyjsciowego.jsonl" \
      --llama_guard_path "ścieżka_do_folderu_z_modelem_Llama_Guard"
```

### Dataset w folderze /data
Źródło: https://huggingface.co/datasets/JerzyPL/GadziJezyk

### Przykładowe środowisko wykonawcze
https://colab.research.google.com/drive/1T0YR5dW1SW1AvLUfG5obpesLx6P260FF?usp=sharing

### Wymagania
- Python 3.8+
- GPU (np. T4, L4) dla modeli HuggingFace
- klucz do API LLM (np. Llama API)
- token HuggingFace

Built with Llama, Copyright © Meta Platforms, Inc. All Rights Reserved.
