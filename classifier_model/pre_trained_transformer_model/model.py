from torchvision.models.video import swin3d_t, Swin3D_T_Weights
import torch.nn as nn

def create_swin3d_t_model_training():
    model = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
    orig_head_input = model.head.in_features
    model.head = nn.Linear(orig_head_input, 3)
    return model

def create_swin3d_t_model_inference():
    model = swin3d_t()
    orig_head_input = model.head.in_features
    model.head = nn.Linear(orig_head_input, 3)
    return model
