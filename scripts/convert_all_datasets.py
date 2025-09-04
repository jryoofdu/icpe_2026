import os
import json
import sys
import requests
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path
import kagglehub
from pathlib import Path
import json
from huggingface_hub import login
import ijson
#from kaggle.api.kaggle_api_extended import KaggleApi
import argparse
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("--max_total_entries", type=int, default=1000000, help="Max total entries across all files")
parser.add_argument("--max_entries_per_file", type=int, default=20000, help="Max entries per JSONL file")
parser.add_argument("--language", type=str, default="en", help="Filter entries by language (e.g., 'en' or 'fr')")
args = parser.parse_args()

 
# Authenticate Hugging Face API
login(token="hf_your_token_here")
try:
    from datasets import load_dataset
    HUGGINGFACE_AVAILABLE = True
except ImportError as ex:
    print(ex)
    HUGGINGFACE_AVAILABLE = False

try:
    import kagglehub
except ImportError:
    kagglehub = None
# Add TokenSim to path for utility
sys.path.append(str(Path(__file__).resolve().parent.parent))
from TokenSim.utils import get_generation_lens

DATASETS = {
    "longbench": ("longbench_input.json", "THUDM/LongBench-v2"),
   # "needle_in_a_haystack": ("haystack.json", None),
  # "sharegpt": ("sharegpt.json", None),
   #"bookcorpus": ("bookcorpus.json", None),  # Use Kaggle local JSONL
  # "wikipedia_structured": ("wiki.json", None),
}

MODELS = {
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "internlm2-7b": "internlm/internlm2-chat-7b"
}

GEN_LEN_MEAN = 128
GEN_LEN_RANGE = 64
GEN_DIST = "uniform"

def extract_prompt(dataset_name, entry):
    if dataset_name == "longbench":
        context = entry.get("context", "")
        question = entry.get("question", "")
        choices = [entry.get(f"choice_{c}", "") for c in ["A", "B", "C", "D"]]
        return context + "\n" + question + "\n" + "\n".join([f"{l}. {c}" for l, c in zip("ABCD", choices)])
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
        # Extract paragraph text from nested sections
        #sections = entry.get("sections", [])
        #return extract_section_text(sections)
    return ""

def fallback_sharegpt_stream():
    base_path = Path("dataset") / "sharegpt" / "raw"
    files = ["sg_90k_part1.json", "sg_90k_part2.json"]
    def generator():
        for file in files:
            file_path = base_path / file
            print(f"📄 Streaming ShareGPT from: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                for obj in ijson.items(f, "item"):
                    if isinstance(obj.get("conversations", None), list):
                        yield obj
    return generator()
def load_bookcorpus_local():
    data_dir = Path("dataset/bookcorpus/raw")
    text_files = list(data_dir.glob("*.txt"))
    
    for txt_file in text_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield {"text": line}
def extract_section_text(sections):
    """Extract all paragraph text from nested section structures."""
    all_text = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        # Use a stack to walk through nested 'has_parts'
        stack = [sec]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                if item.get("type") == "paragraph":
                    val = item.get("value", "").strip()
                    if val:
                        all_text.append(val)
                # Continue traversal if nested
                if "has_parts" in item:
                    stack.extend(item["has_parts"])
    return "\n\n".join(all_text).strip()


# MAIN LOOP
for dataset_name, (local_filename, hf_id) in DATASETS.items():
    local_path = Path("dataset") / dataset_name / local_filename
    dataset_data = None
    print(hf_id)
    print(HUGGINGFACE_AVAILABLE)
    if hf_id and HUGGINGFACE_AVAILABLE:
            try:
                print(f"📡 Trying to load {dataset_name} from Hugging Face: {hf_id}")
                dataset_data = load_dataset(hf_id)["train"]
            except Exception as e:
                print(f"⚠️  Hugging Face load failed for {hf_id}, fallback to local: {e}")

    if dataset_name == "sharegpt":
        print("🔁 Always using fallback loader for ShareGPT...")
        dataset_data = fallback_sharegpt_stream()
    if dataset_name == "bookcorpus":
        print(f"📦 Loading local BookCorpus .txt files...")
        dataset_data = load_bookcorpus_local()
        
    
    elif dataset_name == "wikipedia_structured":
        try:
            print("📡 Checking for Wikipedia Structured dataset via kagglehub...")

            expected_cache_path = (
                Path.home() / ".cache" / "kagglehub" / "datasets" /
                "wikimedia-foundation" / "wikipedia-structured-contents" / "versions" / "1" 
            )

            if expected_cache_path.exists():
                print("✅ Wikipedia Structured already exists locally.")
                dataset_dir = expected_cache_path
            else:
                print("⬇️ Downloading Wikipedia Structured from Kaggle...")
                dataset_dir = Path(kagglehub.dataset_download("wikimedia-foundation/wikipedia-structured-contents"))
                print("✅ Path to dataset files:", dataset_dir)

            jsonl_files = list(dataset_dir.rglob("*.jsonl"))
            if not jsonl_files:
                print(f"❌ No JSONL files found in {dataset_dir}")
                continue

            print(f"✅ Found {len(jsonl_files)} JSONL files. Streaming with:")
            print(f"   🔸 max_total_entries = {args.max_total_entries}")
            print(f"   🔸 max_entries_per_file = {args.max_entries_per_file}")
            print(f"   🔸 language filter = {args.language}")

            def wikipedia_generator(max_total_entries, max_entries_per_file, language_filter):
                total_count = 0
                for file_path in jsonl_files:
                    print(f"📂 Reading file: {file_path.name}")
                    local_count = 0
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f):
                            if total_count >= max_total_entries or local_count >= max_entries_per_file:
                                break
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)

                                # ✅ Language fallback (assume 'en' if missing)
                               # lang = obj.get("language", "en")
                                #if lang != language_filter:
                                    #continue

                                # ✅ Extract paragraph text from sections (no raw "text" field available)
                                sections = obj.get("sections", [])
                                text = extract_section_text(sections)

                                if not text or len(text) <= 0:
                                    #print(f"⚠️ [Line {line_num}] Skipped: no paragraph text or too short ({len(text)} chars)")
                                    continue

                                yield {"text": text}
                                total_count += 1
                                local_count += 1

                                if total_count % 100 == 0:
                                    print(f"✅ Yielded {total_count} valid prompts")

                            except Exception as e:
                                print(f"❌ [Line {line_num}] JSON error: {e}")
                                continue

                    if total_count >= max_total_entries:
                        break


            dataset_data = list(itertools.islice(
                                    wikipedia_generator(
                                        max_total_entries=args.max_total_entries,
                                        max_entries_per_file=args.max_entries_per_file,
                                        language_filter=args.language
                                    ),
            args.max_total_entries  # preload into memory only this many
                ))
            print(f"✅ Loaded {len(dataset_data)} prompt entries into memory.")
           # print("🔍 First few entries from dataset_data:")
            for i, entry in enumerate(dataset_data[:5]):
                print(f"Entry {i + 1}:", entry)


        except Exception as e:
            print(f"❌ Failed to load Wikipedia Structured from kagglehub: {e}")
            continue

        if dataset_data is None:
            if local_path.exists():
                print(f"📂 Loading {dataset_name} from local: {local_path}")
                with open(local_path, "r", encoding="utf-8") as f:
                    dataset_data = json.load(f)
            elif dataset_name == "needle_in_a_haystack":
                print("❌ Skipping needle_in_a_haystack: no data found")
                continue
            else:
                print(f"❌ Skipping {dataset_name}: no data found")
                continue

        print(f"✅ Loaded dataset: {dataset_name}")

    for model_name, tokenizer_id in MODELS.items():
        output_path = Path("dataset") / dataset_name / "converted" / f"{model_name}.json"
        if output_path.exists():
            print(f"⏩ Skipping {dataset_name}-{model_name}, output already exists.")
            continue

        print(f"\n🚀 Tokenizing with {model_name}: {tokenizer_id}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        
        prompt_lens = []
        entry_count = 0

        iterator = dataset_data if isinstance(dataset_data, list) else iter(dataset_data)

        for raw_entry in tqdm(iterator, desc=f"{dataset_name}-{model_name}"):
            try:
                entry = dict(raw_entry)
                prompt = extract_prompt(dataset_name, entry)
                tokens = tokenizer(prompt, truncation=True, max_length=2048)["input_ids"]
                prompt_lens.append(len(tokens))
                entry_count += 1
            except Exception:
                continue

        print(f"✅ {dataset_name}: {entry_count} prompts processed for {model_name}")

        generation_lens = get_generation_lens(
            distribution=GEN_DIST,
            len_mean=GEN_LEN_MEAN,
            len_range=GEN_LEN_RANGE,
            num_prompt=len(prompt_lens)
        )

        output_data = list(zip(prompt_lens, generation_lens))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"💾 Saved {len(output_data)} to {output_path}")
