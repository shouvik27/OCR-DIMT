"""Projection module for LayoutLMv3 + T5 model."""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from torch.nn import LayerNorm, GELU

def build_projection(t5_model: T5ForConditionalGeneration) -> nn.Sequential:
    """Build projection layer from LayoutLMv3 to T5 embedding space.
    
    Args:
        t5_model: T5 model instance
        
    Returns:
        Sequential projection module
    """
    return nn.Sequential(
        nn.Linear(768, t5_model.config.d_model),
        LayerNorm(t5_model.config.d_model),
        GELU()
    )