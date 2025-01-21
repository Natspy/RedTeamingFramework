import pandas as pd


def read_prompts(prompts_file, limit=None, encoding="utf-8-sig"):
    df = pd.read_csv(prompts_file, sep=",", encoding=encoding)
    df.columns = df.columns.str.strip()  # Usuń białe znaki z nazw kolumn

    prompts = []
    for _, row in df.iterrows():
        prompt = row["Zapytanie"]
        categories = [f"Kat {i}" for i in range(1, 6) if row.get(f"Kat {i}", "").strip() == "1"]
        prompts.append({"prompt": prompt, "categories": categories})

    if limit is not None and limit < len(prompts):
        prompts = prompts[:limit]

    return prompts
