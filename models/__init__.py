import importlib

def get_model(model_name: str, model_type: str, args: dict):
    model_filename = "models." + model_name

    # Import model
    try: 
        modellib = importlib.import_module(model_filename)
    except ImportError:
        raise NotImplementedError(f"Model {model_name} not implemented")

    # Get the model
    model = None
    for name, cls in modellib.__dict__.items(): 
        if name.lower() == model_type.lower(): 
            model = cls(args)
            break

    # Check if the model was found
    if model is None:
        raise NotImplementedError(f"{model_type} not found in {model_filename}")

    return model
