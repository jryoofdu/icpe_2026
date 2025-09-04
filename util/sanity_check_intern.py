# sanity_check.py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "internlm/internlm2-7b",
    trust_remote_code=True
)
print("loaded InternLM2-7B with trust_remote_code=True")
