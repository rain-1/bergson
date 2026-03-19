import argparse
from datasets import Dataset, load_from_disk


def prepare_base_model_data(examples, tokenizer):
    prompts = examples["prompt"]
    completions = examples["completion"]

    # We concatenate them exactly how your base model expects to see them
    full_texts = [p + c for p, c in zip(prompts, completions)]

    encodings = tokenizer(full_texts, truncation=True, max_length=2048)
    labels = []

    for i, (p, c) in enumerate(zip(prompts, completions)):
        # Tokenize prompt separately just to find its length
        prompt_len = len(tokenizer(p, add_special_tokens=False)["input_ids"])

        # Everything gets set to -100 except the completion!
        seq_labels = [-100] * prompt_len + encodings["input_ids"][i][prompt_len:]
        labels.append(seq_labels)

    encodings["labels"] = labels
    encodings["length"] = [len(ids) for ids in encodings["input_ids"]]
    # Preserve the original text so bergson show can display it
    encodings["prompt"] = list(prompts)
    encodings["completion"] = list(completions)
    return encodings


def main():
    parser = argparse.ArgumentParser(description="Preprocess a dataset for Bergson fine-tuning")
    parser.add_argument("input", help="Path to input dataset (HuggingFace disk format)")
    parser.add_argument("output", help="Path to save preprocessed dataset")
    parser.add_argument("--model", required=True, help="Model name or path for tokenizer")
    parser.add_argument("--max-length", type=int, default=2048, help="Max token length (default: 2048)")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.input.endswith(".jsonl") or args.input.endswith(".json"):
        dataset = Dataset.from_json(args.input)
    else:
        dataset = load_from_disk(args.input)
    dataset = dataset.map(
        lambda x: prepare_base_model_data(x, tokenizer),
        batched=True,
        desc="Tokenizing",
    )
    dataset.save_to_disk(args.output)
    print(f"Saved preprocessed dataset to {args.output}")


if __name__ == "__main__":
    main()

