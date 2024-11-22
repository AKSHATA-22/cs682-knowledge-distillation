import torch
import torch.nn as nn

class QuantizationLayer(nn.Module):
    def __init__(self, bit_width):
        super(QuantizationLayer, self).__init__()
        self.bit_width = bit_width

    def forward(self, x):
        # return 2*(torch.floor(((2**self.bit_width-1)/2)*(x+1))/(2**self.bit_width-1) - 0.5)
        # sigma = x.std()  # Standard deviation for scaling
        # return (sigma * torch.bucketize(x / sigma, torch.linspace(0, 3, 2**self.bit_width, device=x.device) , right=True) / (2**self.bit_width - 1)) * sigma
        return x.std() * (x.clamp(-1, 1).mul(2**self.bit_width).round() / (2**self.bit_width))
    
    # def forward_new(self, x):
    #     sigma = x.std()  
    #     q_levels = torch.linspace(0, 3, 2**self.bit_width) 
    #     quantized_x = torch.zeros_like(x)
    #     for i, q in enumerate(q_levels):
    #         mask = (x >= (sigma * q_levels[i - 1] if i > 0 else 0)) & (x < sigma * q)
    #         quantized_x[mask] = sigma * q

    #     return quantized_x
   
