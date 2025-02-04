import os
import yaml
from typing import OrderedDict, Dict,List
from nvflare.lighter.provision import gen_default_project_config, prepare_project
from nvflare.lighter.utils import update_project_server_name_config, update_server_default_host, load_yaml

def createProjectFile(workspace):
    os.makedirs(workspace, exist_ok=True)
    project_file = os.path.join(workspace, "project.yml")
    
    with open(project_file, 'w') as f:
    # write to the file        
        gen_default_project_config("dummy_project.yml", project_file)
        
    # Replace dummy_project.yml with project_SoraChain while building the project
    return project_file

def update_server_host(src_project_file, hostname):
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

    for b in project_config.get("builders"):
        path = b.get("path")
        args = b.get("args")
        print (path , args)
        if b.get("path") == "nvflare.lighter.impl.static_file.StaticFileBuilder":
            path ="nvflare.ha.dummy_overseer_agent.DummyOverseerAgent"
            sp_end_point = args["overseer_agent"]["args"]["sp_end_point"]
            args["overseer_agent"]["args"]["sp_end_point"] = f"{hostname}:8002:8003"
            print(sp_end_point)

    return project_config



def save_project_config(project_config, project_file):
    with open(project_file, "w") as file:
        yaml.dump(project_config, file)