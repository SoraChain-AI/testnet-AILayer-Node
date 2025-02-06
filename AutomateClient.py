import os
import json
import argparse
import subprocess
from pathlib import Path
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import default_Data_path , default_training_server
from utils.S3Uploader import S3Uploader
from src.PreprocessProject import get_sp_end_point

def main():
    args = define_parser()
    modelPath = args.model_name_or_path
    getModel(modelPath) 

    if args.data_path is None:
        args.data_path = f"{default_Data_path}/training.jsonl"

    current_folder = os.path.dirname(os.path.realpath(__file__))

    workspace = args.workspace_dir
    if args.SORA_ACCESS_KEY_ID is not None or args.SORA_SECRET_ACCESS_KEY is not None or args.SORA_BUCKET_NAME is not None:        
        getConfig(workspace,args.client_id, args.SORA_ACCESS_KEY_ID, args.SORA_SECRET_ACCESS_KEY, args.SORA_BUCKET_NAME)

        #start Client
    if args.training_server is not None:
        startClient(args.client_id, workspace, args.training_server)
    else:
        startClient(args.client_id, workspace,default_training_server)


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client_id",
        type=str,
        default="client1",
        required=True,
        help="Clinet ID, used to get the data path for each client",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=":8002:8003",
        help="Clinet ID, used to get the data path for each client",
    )
    
    parser.add_argument(
        "--FLType",
        type=str,
        default="client",
        help="run the script for server or client",
    )
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="./workspace/SoraWorkspace",
        required=True,
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
        "--training_server",
        type=str,
        default="localhost",
        help="address of the training server, default to 'localhost'",
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
        "--SORA_ACCESS_KEY_ID",
        type=str,
        default=None,
        help="AWS_ACCESS_KEY_ID",
    )
    parser.add_argument(
        "--SORA_SECRET_ACCESS_KEY",
        type=str,
        default=None,
        help="secret key aws",
    )
    parser.add_argument(
        "--SORA_BUCKET_NAME",
        type=str,
        default=None,
        help="BucketName, directory of the config files",
    )
    return parser.parse_args()

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

def getConfig(workspace,client_id,aws_access_key_id, aws_secret_access_key, bucket_name):

    # Create an instance of the S3Uploader class
    uploader = S3Uploader(aws_access_key_id, aws_secret_access_key, bucket_name)

    # Call the upload_config_folder method
    current_folder = os.path.dirname(os.path.realpath(__file__))

    logger.info("Starting download configs from cloud bucket")
    uploader.fetch_config_folder(bucket_name, client_id, f"{workspace}")
    logger.info(f"Download configs at {workspace}  ")

def startClient(client_id,workspace, training_server ):
    client_name = client_id

    client_startup_file = f"{workspace}/startup/start.sh"    
    # client_startup_file = f"{workspace}/{client_name}/startup/start.sh"    
    # args = [f'{training_server}{port}' , client_name]
    endpoint = get_sp_end_point(f"{workspace}/startup/fed_client.json")
    args = [endpoint,client_name]
    logger.info(f"Starting Trainer Node on the Client using {client_startup_file} with args {args}")
    subprocess.run(['bash', client_startup_file] + args)
    

if __name__ == "__main__":
    main()