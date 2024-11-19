from dataclasses import dataclass
from typing import Optional, Any, Union
from gops.env.env_gen_ocp.env_model.pyth_base_model import RobotModel, EnvModel
from gops.env.env_gen_ocp.env_model.pyth_idsim_model import idSimEnvModel

from gops.env.env_gen_ocp.pyth_idsim import get_idsimcontext
from gops.env.env_gen_ocp.pyth_base import State
from idsim_model.params import ModelConfig
from idsim.config import Config
import numpy as np
import torch

from idsim_model.model_context import State as ModelState
from idsim_model.model import IdSimModel



class idSimEnvPlanningModel(idSimEnvModel):
    pass

def env_model_creator(**kwargs):
    """
    make env model `pyth_idsim_model`
    """
    return idSimEnvPlanningModel(**kwargs)
