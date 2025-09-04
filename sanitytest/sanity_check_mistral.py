# sanity_check_mistral.py
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "mistralai/Mistral-7B-v0.3"

    # 1) Load tokenizer
    print(f"Loading tokenizer for {model_name}…")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print("✅ Tokenizer loaded")

    # 2) Load model
    print(f"Loading model {model_name} (this may take a minute)…")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=False  # standard HF code
    )
    print("✅ Model loaded successfully")

    # 3) Do a quick forward pass
    inputs = tok("Hello, world!", return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=5)
    print("✅ Generation successful:", tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
