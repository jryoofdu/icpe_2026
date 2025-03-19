import argparse

from LLMCompass.software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from LLMCompass.design_space_exploration.dse import (
    template_to_system,
    read_architecture_template,
)
from LLMCompass.software_model.utils import data_type_dict


def get_compass_vars(args: argparse.Namespace):
    if not args.llm_compass:
        return None
    specs = read_architecture_template(args.llm_compass)
    system = template_to_system(specs)
    prefill_model = TransformerBlockInitComputationTP(
        d_model=4096,
        n_heads=32,
        device_count=1,
        data_type=data_type_dict["fp16"],
    )
    decode_model = TransformerBlockAutoRegressionTP(
        d_model=4096,
        n_heads=32,
        device_count=1,
        data_type=data_type_dict["fp16"],
    )
    wrapped_llmcompass_vars = (system, prefill_model, decode_model)
    return wrapped_llmcompass_vars
