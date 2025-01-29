# 
# python3  sft_job_FedAPI.py --client_ids client1 client2 
# --model_name_or_path /home/frank/DemoFLV1/Model/MistralNano 
# --data_path /home/frank/DemoFLV1/ProcessedData/DollyNano/training.jsonl 
# --workspace_dir /home/frank/DemoFLV1/workspace/hf_peft_MN 
# --job_dir /home/frank/DemoFLV1/workspace/hf_peft_MN/jobs 
# --train_mode PEFT

import os
import json
import argparse
import yaml
from pathlib import Path
from loguru import logger
import subprocess
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import default_Data_path
from sft_job_FedAPI import get_prod_dir
 

def main():
    args = define_parser()
    modelPath = args.model_name_or_path
    getModel(modelPath) 

    if args.data_path is None:
        args.data_path = f"{default_Data_path}/training.jsonl"

    current_folder = os.path.dirname(os.path.realpath(__file__))

    logger.info(f"Staring configuring profiles for server and Client")
    # subprocess.run([
    #     "python",
    #     "sft_job_FedAPI.py",
    #     "--client_ids",
    #     *args.client_ids,
    #     "--model_name_or_path",
    #     f"{current_folder}/data/Model",
    #     "--data_path",
    #     args.data_path,
    #     "--workspace_dir",
    #     args.workspace_dir,
    #     #currntly by default jobs dir is under workspace
    #     # "--job_dir",
    #     # args.job_dir,
        
    #     if(args.AWS_ACCESS_KEY_ID is not None ):
    #         "--AWS_ACCESS_KEY_ID",
    #         args.AWS_ACCESS_KEY_ID,
    #     if(args.AWS_SECRET_ACCESS_KEY is not None ):
    #         "--AWS_SECRET_ACCESS_KEY",
    #         args.AWS_SECRET_ACCESS_KEY,
    #     if(args.BUCKET_NAME is not None ):
    #         "--AWS_BUCKET_NAME",
    #         args.BUCKET_NAME,
    #     "--train_mode",
    #     args.train_mode,
    # ], check=True)
    
   
    args_list = [
        "python",
        "sft_job_FedAPI.py",
        "--client_ids",
        *args.client_ids,
        "--model_name_or_path",
        f"{current_folder}/data/Model",
        "--data_path",
        args.data_path,
        "--workspace_dir",
        args.workspace_dir,
    ]

    if args.AWS_ACCESS_KEY_ID is not None:
        args_list.extend(["--AWS_ACCESS_KEY_ID", args.AWS_ACCESS_KEY_ID])

    if args.AWS_SECRET_ACCESS_KEY is not None:
        args_list.extend(["--AWS_SECRET_ACCESS_KEY", args.AWS_SECRET_ACCESS_KEY])

    if args.BUCKET_NAME is not None:
        args_list.extend(["--AWS_BUCKET_NAME", args.BUCKET_NAME])

    subprocess.run(args_list, check=True)

    logger.info(f"finished conofiguring profiles for server and Client")
    logger.info(f"Starting Aggregator Node on the server")

    current_folder = os.path.dirname(os.path.realpath(__file__))

    config_folder_path = get_prod_dir(args.workspace_dir)
    project_file = os.path.join(args.workspace_dir, "project.yml")

    # Open the YAML file
    with open(project_file, 'r') as file:
        # Load the YAML file
        data = yaml.safe_load(file)

    # Extract the value of the 'name' field
    server_name = data['participants'][0]['name']    
    server_startup_file = f"{config_folder_path}/{server_name}/startup/start.sh"    
    args = 'localhost'
    logger.info(f"Starting Aggregator Node on the server at {server_startup_file} with args {args}")
    subprocess.run([server_startup_file, args])

def getProjectFile():
    pass

def getModel(path ):
    #download model
    current_folder = os.path.dirname(os.path.realpath(__file__))
    logger.info("Downloading Model from repo")
    model = AutoModelForCausalLM.from_pretrained(path)    
    
    logger.info("Downloading Tokensizer from repo")
    tokenizer = AutoTokenizer.from_pretrained(path)   
    
    #savinf model and config
    model.save_pretrained(f"{current_folder}/data/Model")
    tokenizer.save_pretrained(f"{current_folder}/data/Model")



def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client_ids",
        nargs="+",
        type=str,
        default="client1 client2",
        help="Clinet IDs, used to get the data path for each client",
    )
    parser.add_argument(
        "--FLType",
        type=str,
        default="server",
        help="run the script for server or client",
    )
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="./workspace/SoraWorkspace",
        help="work directory, default to './workspace/SoraWorkspace'",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="crumb/nano-mistral",
        #"meta-llama/llama-3.2-1b",
        help="model name or path",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="root directory for training and validation data",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="PEFT",
        help="training mode, SFT or PEFT, default to PEFT",
    )
    parser.add_argument(
        "--quantize_mode",
        type=str,
        default=None,
        help="quantization mode, float16 or blockwise8, default to None (no quantization)",
    )
    parser.add_argument(
        "--AWS_ACCESS_KEY_ID",
        type=str,
        default=None,
        help="AWS_ACCESS_KEY_ID",
    )
    parser.add_argument(
        "--AWS_SECRET_ACCESS_KEY",
        type=str,
        default=None,
        help="secret key aws",
    )
    parser.add_argument(
        "--BUCKET_NAME",
        type=str,
        default=None,
        help="BucketName",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()