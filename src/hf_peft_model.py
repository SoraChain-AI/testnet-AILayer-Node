import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


class CausalLMPEFTModel(torch.nn.Module):
    def __init__(self, model_name_or_path):
        super(CausalLMPEFTModel, self).__init__()
        # PEFT configs
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        full_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
        )
        self.model = get_peft_model(full_model, peft_config)

    def forward(self, input_id):
        output = self.model(input_ids=input_id, return_dict=False)
        return output
