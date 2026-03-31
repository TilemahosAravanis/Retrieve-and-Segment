from omegaconf import OmegaConf

import importlib

def import_model_class(model_type):
    """
    Import class from path models/<model_type>/<model_type>.py
    and class name CamelCase(<model_type>)
    """
    
    # model_type in lowercase
    model = model_type.lower()

    module_path = f"models.{model}.{model}"
    
    module = importlib.import_module(module_path)
    return getattr(module, model_type)

def build_model(cfg, class_names, dataset_key=None):
    """
    Build model from config
    """
    
    model_cfg = OmegaConf.to_container(cfg, resolve=True)
    model_type = model_cfg['type']
    
    # pop the type key
    model_cfg.pop('type')
    
    ModelClass = import_model_class(model_type)

    return ModelClass(**model_cfg, class_names=class_names, dataset_key=dataset_key)