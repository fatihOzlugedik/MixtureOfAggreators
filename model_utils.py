import os 

import torch
import torch.nn as nn

from torchvision import transforms
from transformers import BeitFeatureExtractor, Data2VecVisionModel, ViTModel, AutoImageProcessor


def get_models(modelname, image_size, saved_model_path=None):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # --- histology-pretrained models
    if modelname.lower() == "ctranspath":
        model = get_ctranspath(saved_model_path)
    elif modelname.lower() == "phikon":
        model = Phikon()
    elif modelname.lower() == "uni":
        model = get_uni(saved_model_path)
    elif modelname.lower() == "conch":
        model= get_conch(saved_model_path)
    elif modelname.lower() == "conchv1.5":
        from dinov2.eval.patch_level.models.conch_v1_5 import build_conch, CONCHConfig
        model, _ = build_conch(CONCHConfig())
    elif modelname.lower() == "h-optimus-0":
        import timm
        model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)
        

    elif modelname.lower() in ["dinobloom_s","dinobloom_b","dinobloom_l","dinobloom_g"]:
        modelname_dict= {"dinobloom_s":"dinov2_vits14", "dinobloom_b":"dinov2_vitb14", "dinobloom_l":"dinov2_vitl14", "dinobloom_g":"dinov2_vitg14"}
        modelname = modelname_dict[modelname]
        model = get_dino_finetuned_downloaded(saved_model_path,modelname,image_size)

    elif modelname.lower() in ["superbloom"]:
        modelname_dict= {"superbloom":"dinov2_vitl14"}
        modelname = modelname_dict[modelname]
        model = get_dino_finetuned_downloaded(saved_model_path,modelname,image_size)
    else: 
        raise ValueError(f"Model {modelname} not found")

    model = model.to(device)
    model.eval()

    return model


def get_uni(saved_model_path):
    import timm
    model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
    model.load_state_dict(torch.load(os.path.join(saved_model_path), map_location="cpu"), strict=True)
    # model.load_state_dict(torch.load(os.path.join(saved_model_path, "pytorch_model.bin"), map_location="cpu"), strict=True)
    return model

def get_conch(model_path):
    from conch.open_clip_custom import create_model_from_pretrained 
    model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=str(model_path))
    return model



# for 224
def get_dino_finetuned_downloaded(model_path, modelname,image_size):
    model = torch.hub.load("facebookresearch/dinov2", modelname)
    # load finetuned weights

    # pos_embed has wrong shape
    if model_path is not None:
        pretrained = torch.load(model_path, map_location=torch.device("cpu"))
        # make correct state dict for loading
        new_state_dict = {}
        for key, value in pretrained["teacher"].items():
            if "dino_head" in key or "ibot_head" in key:
                pass
            else:
                new_key = key.replace("backbone.", "")
                new_state_dict[new_key] = value
        # change shape of pos_embed
        input_dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }
        num_tokens=int(1+(image_size/14)**2)
        pos_embed = nn.Parameter(torch.zeros(1, num_tokens, input_dims[modelname.replace("_reg","")]))
        model.pos_embed = pos_embed
        # load state dict
        model.load_state_dict(new_state_dict, strict=True)
        
    return model


def get_ctranspath(model_path):
    from dinov2.eval.patch_level.models.ctran import ConvStem
    from dinov2.eval.patch_level.models.swin_transformer import swin_tiny_patch4_window7_224
    model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
    model.head = nn.Identity()
    pretrained = torch.load(model_path)
    model.load_state_dict(pretrained["model"], strict=True)
    return model


def get_transforms(model_name,image_size=224, saved_model_path=None):
    # from imagenet, leave as is
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if model_name.lower() in ["ctranspath", "uni", "conch", "conchv1.5"]:
        size = 224
    elif model_name.lower() == "phikon":
        image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        mean, std = image_processor.image_mean, image_processor.image_std
        size = image_processor.size['height']
 
    # change later to correct value
    elif model_name.lower().replace("_reg","") in [
        "dinov2_vits14",
        "dinov2_vits14_classifier",
        "dinov2_vitb14",
        "dinov2_vitl14",
        "dinov2_vitg14",
        "dinov2_finetuned",
        "dinov2_vits14_interpolated",
        "dinov2_finetuned_downloaded",
        "remedis",
        "vim_finetuned",
        "dinobloom_s",
        "dinobloom_b",
        "dinobloom_l",
        "dinobloom_g",
        "superbloom",
        "dinov2_vitl16",
    ]:
        size = image_size
    elif model_name.lower() == "h-optimus-0":
        size = 224
        mean=(0.707223, 0.578729, 0.703617)
        std=(0.211883, 0.230117, 0.177517)
    else:
        raise ValueError("Model name not found")
    
    size=(size,size)
    
    transforms_list = [transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

    preprocess_transforms = transforms.Compose(transforms_list)

    if model_name.lower() == "conch":
        from conch.open_clip_custom import create_model_from_pretrained 
        _ , preprocess_transforms = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=str(saved_model_path))
    elif model_name.lower() == "conchv1.5":
        from dinov2.eval.patch_level.models.conch_v1_5 import build_conch, CONCHConfig
        _, preprocess_transforms = build_conch(CONCHConfig())

    return preprocess_transforms


class Phikon(nn.Module):
    def __init__(self):
        super(Phikon, self).__init__()
        self.model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

    def forward(self, x):
        outputs = self.model(x)
        features = outputs.last_hidden_state[:, 0, :]
        return features