from model import AEProb, AEDist
from lightning import LightningModule
import torch

class GenericLightning(LightningModule):
    def __init__(self):
        super().__init__()

def get_model_by_name(model_type):
    match model_type:
        case "AEProb":
            return AEProb()
        case "Lightning":
            return GenericLightning()
        case "AEDist":
            return AEDist()
        case _:
            raise NotImplementedError()


def load_model_from_checkpoint_file(filepath, model_name = None):
    """
    Given a checkpoint file, loads the correspond pytorch model.
    If it's a lightning .ckpt file, that's all we need; these files contain the architecture info.
    If it's a pytorch .pt file, we also need the corresponding model class.
    """
    # If its a lightning checkpoint, we don't need the model name
    if filepath[-5:] == '.ckpt': # it's a lightning module
        model = get_model_by_name("lightning")
        model = model.load_from_checkpoint(filepath)
    elif filepath[-3:] == '.pt':
        # must discover model name
        if model_name is not None:
            pass
        elif "probae" in filepath.lower() or "aeprob" in filepath.lower():
            model_name = "AEProb"
        else:
            raise NotImplementedError("Unknown model_name")
        # load model 
        model = get_model_by_name(model_name)
        state_dict = torch.load(filepath)
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError("Checkpoint filename should have .pt or .ckpt format.")
    return model
