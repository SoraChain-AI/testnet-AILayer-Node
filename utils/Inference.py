
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

    
    is_success = merge_lora_to_base_model(base_model, model_path, f"{model_path}/model.safetensors")

    if not is_success:
        print("Error in merging Lora to base model")
        return
    
    model = AutoModelForCausalLM.from_pretrained(f"{model_path}/model.safetensors")
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
    # outputs = tokenizer.batch_decode(gen_tokens)
    print(gen_text)
    # for i in outputs:
    #     print(i)


#     # from transformers import T5Tokenizer, T5ForConditionalGeneration

#     # tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
#     # model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")

#     # input_text = "translate English to German: How old are you?"
#     # input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

#     # outputs = model.generate(input_ids)
#     # print(tokenizer.decode(outputs[0]))

# from datasets import load_dataset   

# def main():
#     print("hello")

#     dataset =  load_dataset("VMware/open-instruct", split="train")


if __name__ == "__main__":
    main()
