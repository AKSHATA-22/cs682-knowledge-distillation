import math
import torch
import torch.nn as nn

def quantize(x, bit_width):
    sigma = x.std()  # Compute standard deviation of input
    # print(2**bit_width)
    q_levels = torch.linspace(0, 3, 2**bit_width) 
    # print(q_levels) # Quantization levels (simplified)
    # print(sigma)
    # print(sigma*q_levels)
        # Apply quantization: map each value to the closest quantization level
    quantized_x = torch.zeros_like(x)
    for i, q in enumerate(q_levels):
        mask = (x >= (sigma * q_levels[i - 1] if i > 0 else 0)) & (x < sigma * q)
        quantized_x[mask] = sigma * q

    return quantized_x

def feature_map(quant_activations_importances):
    return sum((imp * quant_activation) for imp, quant_activation in quant_activations_importances)

def fitnet_loss(F_T, F_S):
    convLayer = nn.Conv2d(1, 1, (1, 1))