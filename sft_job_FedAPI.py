
import argparse
import os
from pathlib import Path
from nvflare import FedJob, FilterType
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.pt.quantization.dequantizor import ModelDequantizor
from nvflare.app_opt.pt.quantization.quantizor import ModelQuantizor
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.apis.workspace import Workspace
# from nvflare.tool.poc import create_workspace
from nvflare.tool.poc.poc_commands import start_poc,stop_poc,_prepare_poc,_prepare_jobs_dir,get_poc_workspace,get_examples_dir
from nvflare.tool.poc.poc_commands import old_start_poc , DEFAULT_WORKSPACE, DEFAULT_PROJECT_NAME,get_prod_dir
from nvflare.tool.job.job_cli import internal_submit_job
from utils.constants import default_Data_path, default_project_title
from loguru import logger
from utils.S3Uploader import S3Uploader
# from blockConnector  import BlockConnector 
# from blockConnector import config


def main():
    args = define_parser()

    #Authenticate User on CHain
    # AuthnticateNode(args.address)
    

    train_script = "src/hf_sft_peft_fl.py"
    client_ids = args.client_ids
    num_clients = len(client_ids)

    if args.threads:
        num_threads = args.threads
    else:
        num_threads = num_clients

    if num_threads < num_clients:
        print("The number of threads smaller than the number of clients, runner clean-up will be performed.")
        clean_up = 1
    else:
        clean_up = 0

    num_rounds = args.num_rounds
    workspace_dir = args.workspace_dir
    #createing job folder and job config
    job_path = Path(workspace_dir)/"jobs"
    job_path.mkdir(parents=True, exist_ok=True)
    job_dir = job_path
    # job_dir = args.job_dir
    model_name_or_path = args.model_name_or_path
    train_mode = args.train_mode
    message_mode = args.message_mode

    # Create the FedJob
    if train_mode.lower() == "sft":
        job = FedJob(name="llm_hf_sft", min_clients=num_clients)
        output_path = "sft"
    elif train_mode.lower() == "peft":
        job = FedJob(name="llm_hf_peft", min_clients=num_clients)
        output_path = "peft"
    else:
        raise ValueError(f"Invalid train_mode: {train_mode}, only SFT and PEFT are supported.")

    # Define the FedAvg controller workflow and send to server
    controller = FedAvg(
        num_clients=num_clients,
        num_rounds=num_rounds,
    )
    job.to(controller, "server")

    if args.quantize_mode:
        # If using quantization, add quantize filters.
        quantizor = ModelQuantizor(quantization_type=args.quantize_mode)
        dequantizor = ModelDequantizor()
        job.to(quantizor, "server", tasks=["train"], filter_type=FilterType.TASK_DATA)
        job.to(dequantizor, "server", tasks=["train"], filter_type=FilterType.TASK_RESULT)

    # Define the model persistor and send to server
    # First send the model to the server
    job.to("src/hf_sft_model.py", "server")
    # Then send the model persistor to the server
    model_args = {"path": "src.hf_sft_model.CausalLMModel", "args": {"model_name_or_path": model_name_or_path}}
    job.to(PTFileModelPersistor(model=model_args), "server", id="persistor")

    # Add model selection widget and send to server
    job.to(IntimeModelSelector(key_metric="eval_loss", negate_key_metric=True), "server", id="model_selector")

    # Send ScriptRunner to all clients
    for i in range(num_clients):
        client_id = client_ids[i]
        site_name = f"{client_id}"
        data_path_train = os.path.join(args.data_path, "training.jsonl")
        data_path_valid = os.path.join(args.data_path,"validation.jsonl")

        script_args = f"--model_name_or_path {model_name_or_path} --data_path_train {data_path_train} --data_path_valid {data_path_valid} --output_path {output_path} --train_mode {train_mode} --message_mode {message_mode} --clean_up {clean_up}"
        if message_mode == "tensor":
            params_exchange_format = "pytorch"
        elif message_mode == "numpy":
            params_exchange_format = "numpy"
        else:
            raise ValueError(f"Invalid message_mode: {message_mode}, only numpy and tensor are supported.")

        runner = ScriptRunner(
            script=train_script,
            script_args=script_args,
            params_exchange_format=params_exchange_format,
            launch_external_process=False,
        )
        job.to(runner, site_name, tasks=["train"])

        if args.quantize_mode:
            job.to(quantizor, site_name, tasks=["train"], filter_type=FilterType.TASK_RESULT)
            job.to(dequantizor, site_name, tasks=["train"], filter_type=FilterType.TASK_DATA)

    # Export the job

    # Run the job
    print("workspace_dir=", workspace_dir)
    print("num_threads=", num_threads)
    # job.simulator_run(workspace_dir, threads=num_threads, gpu=args.gpu)
    PreparePOC(args.workspace_dir, args.client_ids)
    
    print("job_dir=", job_dir)
    job.export_job(job_dir)

    #Uploading configuration to the backend
    if(args.AWS_ACCESS_KEY_ID is None or args.AWS_SECRET_ACCESS_KEY is None or args.BUCKET_NAME is None):
        print("Please provide AWS access ,else data will not be uploaded")
    else:
        UploadServerConfiguration(workspace_dir, args.AWS_ACCESS_KEY_ID, args.AWS_SECRET_ACCESS_KEY, args.BUCKET_NAME)
    # _prepare_jobs_dir(job_dir, args.workspace_dir)
    # start_poc(workspace_dir, num_threads, args.gpu, job_dir)

def PreparePOC( workspacePath , client_ids ):

    list_of_clients = []
    for client in client_ids:
        list_of_clients.append(client)
        logger.info(f"client approved for training: {client}")

    # list_of_clients = ["client1", "client2"]

    num_clients = len(list_of_clients)
    workspace = workspacePath;

    #updating folder structure
    
    DEFAULT_WORKSPACE = workspace
    DEFAULT_PROJECT_NAME = default_project_title        #loads default project title from config file,edit file to make changes
    _prepare_poc(list_of_clients, num_clients, workspace)

    logger.debug(f"prod dir: {get_production_dir(workspace)}")    
    
    

# def AuthnticateNode(userAddress):
#     # check blockchain connection
#     ##
#     try:
#         connector = BlockConnector(config.RPC_URL, config.CONTRACT_ADDRESS, config.ABI_FILE)
#         user_address = userAddress
#         print("connecting to address,", user_address )
#         is_trainer = connector.is_user_trainer(user_address)
#         if is_trainer:
#             print(f"User {user_address} has staked tokens and is a trainer. Proceed with AI training.")
#         else:
#             print(f"User {user_address} has not staked tokens.")
#             raise Exception ("User is not a trainer")

#     except Exception as e:
#         print(f"Error checking blockchain connection: {e}")
#     ##

def UploadServerConfiguration(workspace , AWS_KEY_ID, AWS_SECRET_KEY, BUCKET):


    #provide it with command line argument
    AWS_ACCESS_KEY_ID = AWS_KEY_ID.
    AWS_SECRET_ACCESS_KEY = AWS_SECRET_KEY
    BUCKET_NAME = BUCKET

    # Create an instance of the S3Uploader class
    uploader = S3Uploader(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME)

    # Call the upload_config_folder method
    current_folder = os.path.dirname(os.path.realpath(__file__))

    config_folder_path = get_production_dir(workspace)
    logger.info(f"full path to Production config folder : {config_folder_path}")
    logger.info("Starting Upload configs to cloud bucket")
    uploader.upload_config_folder(f"{config_folder_path}")

    logger.info(f"Config folder uploaded to S3 bucket: {BUCKET_NAME}")
    logger.info(f"Config folder path on S3: https://{BUCKET_NAME}.s3.amazonaws.com/")
    
def get_production_dir(workspace : str):
    current_folder = os.path.dirname(os.path.realpath(__file__))

    config_folder_path = get_prod_dir(workspace)
    return config_folder_path
    # return os.path.join(current_folder, config_folder_path)

 

def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client_ids",
        nargs="+",
        type=str,
        default="",
        help="Clint IDs, used to get the data path for each client",
    )
    parser.add_argument(
        "--FLType",
        type=str,
        default="client",
        help="run the script for server or client",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="poc",
        help="run the job in simulator or poc mode",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=3,
        help="Number of rounds, default to 5",
    )
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="./workspace/SoraWorkspace",
        help="work directory, default to 'workspace'",
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        default="./workspace/SChainPEFT/jobs",
        help="directory for job export, default to './workspace/SChainPEFT'",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/llama-3.2-1b",
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
        default="SFT",
        help="training mode, SFT or PEFT, default to SFT",
    )
    parser.add_argument(
        "--quantize_mode",
        type=str,
        default=None,
        help="quantization mode, float16 or blockwise8, default to None (no quantization)",
    )
    parser.add_argument(
        "--message_mode",
        type=str,
        default="numpy",
        help="message mode, numpy or tensor, default to numpy",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="number of threads to use for FL simulation, default to the number of clients",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="gpu assignments for simulating clients, comma separated, default to single gpu",
    )
    parser.add_argument(
        "--address",
        type=str,
        help="Define address of the connecting node",
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
    # UploadServerConfiguration("./workspace/SoraWorkspace")
