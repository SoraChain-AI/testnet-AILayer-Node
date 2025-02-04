import os
import yaml



def loadTemplate(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def updateTemplateParam(path, argument_name,new_value):
    with open(path, "r") as f:
        data =yaml.safe_load(f)
        data.update({argument_name:new_value})
