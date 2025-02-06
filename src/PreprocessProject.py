import os
import yaml
from typing import OrderedDict, Dict,List
from nvflare.lighter.provision import gen_default_project_config, prepare_project
from nvflare.lighter.utils import update_project_server_name_config, update_server_default_host, load_yaml
import re
import json

def createProjectFile(workspace):
    os.makedirs(workspace, exist_ok=True)
    project_file = os.path.join(workspace, "project.yml")
    
    with open(project_file, 'w') as f:
    # write to the file        
        gen_default_project_config("dummy_project.yml", project_file)
        
    # Replace dummy_project.yml with project_SoraChain while building the project
    return project_file

def update_server_host(src_project_file, hostname):
    """
    Updates the server host in a project configuration file.

    Args:
        src_project_file (str): The path to the source project configuration file.
        hostname (str): The new hostname for the server.

    Returns:
        None
    """
    dst_project_file = src_project_file
    project_config: OrderedDict = load_yaml(src_project_file)
    project_config = update_server_default_host(project_config, hostname)
    project_config= update_server_name(project_config,hostname)
    print("sp_end_point for ", src_project_file )
    project_config =sp_end_point(project_config, hostname)
    save_project_config(project_config, dst_project_file)

def update_server_name(project_config,server_name):
    old_server_name = get_fl_server_name(project_config)    
    print("old_server_name",old_server_name)
    if old_server_name != server_name:
        return update_project_server_name_config(project_config, old_server_name, server_name)
    return project_config

def get_fl_server_name(project_config: OrderedDict) -> str:
    participants: List[dict] = project_config["participants"]
    servers = [p["name"] for p in participants if p["type"] == "server"]
    if len(servers) == 1:
        return servers[0]
    else:
        raise Exception(f"project should only have one server, but {len(servers)} are provided: {servers}")
    
def sp_end_point(project_config, hostname):

    participants: List[dict] = project_config["participants"]
    serverFLPort = [p["fed_learn_port"] for p in participants if p["type"] == "server"]
    admin_port = [p["admin_port"] for p in participants if p["type"] == "server"]

    for b in project_config.get("builders"):
        path = b.get("path")
        args = b.get("args")
        if b.get("path") == "nvflare.lighter.impl.static_file.StaticFileBuilder":
            path ="nvflare.ha.dummy_overseer_agent.DummyOverseerAgent"
            sp_end_point = args["overseer_agent"]["args"]["sp_end_point"]
            args["overseer_agent"]["args"]["sp_end_point"] = f"{hostname}:{serverFLPort[0]}:{admin_port[0]}"

    return project_config

def get_sp_end_point(src_project_file) ->str:
    try:
        print("src_project_file",src_project_file)
        with open(src_project_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

            # Navigate to the required key
            sp_end_point = data.get("overseer_agent", {}).get("args", {}).get("sp_end_point")

            if sp_end_point:
                print("SP End Point:", sp_end_point)
                return sp_end_point
            else:
                print("sp_end_point not found in the JSON file.")
                return None

    except Exception as e:
        print("Error reading JSON file:", e)
    return None



# Example usage
# modify_secure_train("substart.sh")
def modify_secure_train(file_path):
    """Modify the secure_train value from true to false in a shell script."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Replace secure_train=true with secure_train=false
        modified_content = re.sub(r'(--set secure_train)=true', r'\1=false', content)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(modified_content)
        
        print("Updated secure_train to false in", file_path)
    except Exception as e:
        print("Error modifying file:", e)


def save_project_config(project_config, project_file):
    with open(project_file, "w") as file:
        yaml.dump(project_config, file)