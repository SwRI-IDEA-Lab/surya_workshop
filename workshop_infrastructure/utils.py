import torch
from peft import LoraConfig, get_peft_model

def apply_peft_lora(
    model: torch.nn.Module,
    config: dict,
) -> torch.nn.Module:
    """
    Applies PEFT LoRA to the HelioSpectformer1D model

    Args:
        model: The HelioSpectformer1D model to apply LoRA to.
        config: Configuration object containing LoRA settings.
        logger: Standard python logging.Logger object.

    Returns:
        Model with PEFT LoRA adapters applied.
    """
    if "lora_config" not in config["model"].keys():
        print("No LoRA configuration found. Using default LoRA settings.")
        lora_config = {
            "r": 8,  # LoRA rank
            "lora_alpha": 8,  # LoRA alpha parameter
            "target_modules": [
                "q_proj",
                "v_proj",
                "k_proj",
                "out_proj",
                "fc1",
                "fc2",
            ],  # Target modules for LoRA
            "lora_dropout": 0.1,
            "bias": "none",
        }
    else:
        lora_config = config["model"]["lora_config"]

    print(f"Applying PEFT LoRA with configuration: {lora_config}")

    # Create LoRA configuration
    peft_config = LoraConfig(
        r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("lora_alpha", 8),
        target_modules=lora_config.get(
            "target_modules", ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        ),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        bias=lora_config.get("bias", "none"),
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)

    # Log the number of trainable parameters
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )

    return model