# testnet-AILayer-Node
The repo contains a demo script for you to finetune a open Source Model with SoraEngine

## Configure Environment

## Preprocess Data
 python ./src/preprocess_nanoArticles.py --output_dir ./data/Output

## Start Automation
### project configuration , submitting Config files and starting server

python AutoMateServer.py --client_ids Client1 Client2 --model_name_or_path crumb/nano-mistral --data_path ${PWD}/data/Output --workspace_dir ${PWD}/workspace/SoraWorkspace --train_mode PEFT
