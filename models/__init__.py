import importlib
from models.vanilla_gan import Vanilla_GAN

def get_model(model_name: str, fabric, args: dict):
    model_filename = "models." + model_name

    if model_name == "vanilla_gan":
        model = Vanilla_GAN(args, fabric)
    else:
        model = None

    # Check if the model was found
    if model is None:
        raise NotImplementedError(f"{model_name} not found in {model_filename}")

    return model
