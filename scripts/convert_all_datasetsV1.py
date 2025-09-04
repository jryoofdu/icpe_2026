import os
import json
import sys
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path
from huggingface_hub import login
# Hugging Face API login for gated models like LLaMA-2
login(token="")
# Optional: Use Hugging Face datasets if available
try:
    from datasets import load_dataset
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Add TokenSim path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from TokenSim.utils import get_generation_lens

# Dataset name → (local filename, HuggingFace dataset ID)
DATASETS = {
    "longbench": ("longbench_input.json", "THUDM/LongBench-v2"),
  "needle_in_a_haystack": ("haystack.json", None),
   "sharegpt": ("sharegpt.json", "RyokoAI/ShareGPT52K"),
    "bookcorpus": ("bookcorpus.json", "bookcorpusopen/bookcorpusopen"),
   "wikipedia_structured": ("wiki.json", "wikitext/wikitext-103-v1")
   
}

MODELS = {
   "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "internlm2-7b": "internlm/internlm2-chat-7b"
}

# Generation length sampling configuration
GEN_LEN_MEAN = 128
GEN_LEN_RANGE = 64
GEN_DIST = "uniform"

# Prompt string extraction per dataset
def extract_prompt(dataset_name, entry):
    if dataset_name == "longbench":
        context = entry.get("context", "")
        question = entry.get("question", "")
        choices = [
            entry.get("choice_A", ""),
            entry.get("choice_B", ""),
            entry.get("choice_C", ""),
            entry.get("choice_D", "")
        ]
        return context + "\n" + question + "\n" + "\n".join([
            "A. " + choices[0],
            "B. " + choices[1],
            "C. " + choices[2],
            "D. " + choices[3]
        ])
    elif dataset_name == "needle_in_a_haystack":
        return entry.get("context", "") + "\n" + entry.get("question", "")
    elif dataset_name == "sharegpt":
        try:
            convos = entry.get("conversations", [])
            if isinstance(convos, str):
                convos = json.loads(convos)
            if isinstance(convos, list):
                return "\n".join([
                    f"{c.get('role', c.get('from', ''))}: {c.get('content', c.get('value', ''))}"
                    for c in convos if isinstance(c, dict)
                ])
        except Exception:
            return ""
        return entry.get("prompt", "")

    elif dataset_name == "bookcorpus":
        return entry.get("text", "")
    elif dataset_name == "wikipedia_structured":
        return entry.get("text", "")
    elif dataset_name == "opencompass":
        return entry.get("instruction", "") + "\n" + entry.get("input", "")
    else:
        return ""

# Main processing loop
for model_name, tokenizer_id in MODELS.items():
    print(f"\n🚀 Loading tokenizer for {model_name}: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    for dataset_name, (local_filename, hf_id) in DATASETS.items():
        dataset_data = None
        local_path = Path("dataset") / dataset_name / local_filename

        if hf_id and HUGGINGFACE_AVAILABLE:
            try:
                print(f"📡 Trying to load {dataset_name} from Hugging Face: {hf_id}")
                dataset_data = load_dataset(hf_id, split="train", streaming=True)
            except Exception as e:
                print(f"⚠️  Hugging Face load failed for {hf_id}, fallback to local: {e}")

        if dataset_data is None:
            if not local_path.exists():
                print(f"❌ Skipping {dataset_name}: no Hugging Face source or local file {local_path}")
                continue
            print(f"📂 Loading {dataset_name} from local file: {local_path}")
            with open(local_path, "r", encoding="utf-8") as f:
                dataset_data = json.load(f)

        # Tokenize prompts and generate lengths
        prompt_lens = []
        entry_count = 0

        if hasattr(dataset_data, "iter"):
            iterator = dataset_data if isinstance(dataset_data, list) else iter(dataset_data)
        else:
            iterator = iter(dataset_data)

        for raw_entry in tqdm(iterator, desc=f"{dataset_name}-{model_name}"):
            try:
                entry = dict(raw_entry)
                prompt = extract_prompt(dataset_name, entry)
                tokens = tokenizer(prompt, truncation=True)["input_ids"]
                prompt_lens.append(len(tokens))
                entry_count += 1
            except Exception:
                continue

        print(f"✅ Processed {entry_count} prompts.")


        generation_lens = get_generation_lens(
            distribution=GEN_DIST,
            len_mean=GEN_LEN_MEAN,
            len_range=GEN_LEN_RANGE,
            num_prompt=len(prompt_lens)
        )

        output_data = list(zip(prompt_lens, generation_lens))
        output_dir = Path("dataset") / dataset_name / "converted"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{model_name}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"✅ Saved {len(output_data)} pairs to {output_file}")
