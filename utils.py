import importlib.util
import os
from types import ModuleType
from typing import Callable, Tuple

from kohgpjax.kohmodel import KOHModel
from kohgpjax.parameters import ModelParameterPriorDict


def load_config_from_model_dir(model_dir) -> ModuleType:
    """Dynamically loads a config module from a model directory."""
    config_path = os.path.join(model_dir, "config.py")
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def load_model_from_model_dir(
    model_dir,
) -> Tuple[KOHModel, Callable[[ModuleType, dict], ModelParameterPriorDict]]:
    """Dynamically loads a model and its parameter prior dictionary function from a model directory."""
    model_path = os.path.join(model_dir, "model.py")
    spec = importlib.util.spec_from_file_location("model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model, module.get_ModelParameterPriorDict
