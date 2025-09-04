# sanity_check.py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B",
    trust_remote_code=True
)
print("loaded Mistral-7B with trust_remote_code=True")
