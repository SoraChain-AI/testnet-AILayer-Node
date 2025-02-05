
import torch
import datasets
import argparse
import glob
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM, MistralForCausalLM
from transformers import GPTNeoForCausalLM, GPT2Tokenizer , OPTForCausalLM, OPTConfig, OpenAIGPTTokenizer
from peft import LoraConfig,peft_model, get_peft_model
from safetensors.torch import load_file
from trl import SFTTrainer
from merge import merge_lora_to_base_model


use_flash_attention = True
TF_ENABLE_ONEDNN_OPTS=0

def main():
    print("initiating Inference")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="./workspace/SoraWorkspace/example_project/prod_00/Client2/checkpoint-20",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="./workspace/SoraWorkspace/example_project/prod_00/Client2/peft/global_model.safetensors",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="once upon a time",
    )
    parser.add_argument(
        "--merge_model",
        action="store_true",
    )
    
    
    parser.add_argument("--mode", type=int, default=0)
    args = parser.parse_args()
    
    base_model = args.base_model
    # base_model_weight = load_file(base_model)
    model_path = args.model_path
    # model_files = glob.glob(args.model_path + '*')  # Expand the path using the wildcard

    # if model_files:
    #     print(f"Found model files: {model_files}")
    # else:
    #     print("No files matching the pattern were found.")
    model = AutoModelForCausalLM.from_pretrained(f"{model_path}")

    if args.merge_model:
        is_success = merge_lora_to_base_model(base_model, model_path, f"{model_path}/merge_model")

        if not is_success:
            print("Error in merging Lora to base model")
            return
        model = AutoModelForCausalLM.from_pretrained(f"{model_path}/merge_model")
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = args.prompt

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    input_ids = tokenizer(prompt, padding =True, return_tensors="pt").input_ids

    # inputs = {k:v.to(model.device) for k,v in dict(inputs).items()}
    gen_tokens = model.generate(
        input_ids,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.9,
        top_k=20,
        max_length=300,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print(gen_text)
    # outputs = tokenizer.batch_decode(gen_tokens)
    # for i in outputs:
    #         print(i)


if __name__ == "__main__":
    main()