import torch
import torch.nn as nn

class QuantizeConv2D(nn.Conv2d):
    def __init__(self, *args, bit_width, **kwargs):
        """
        A Conv2d layer that dynamically quantizes weights during the forward pass.

        Args:
            quantization_fn (callable): Custom function to quantize weights.
                                        If None, no quantization is applied.
        """
        super(QuantizeConv2D, self).__init__(*args, **kwargs)
        self.bit_width = bit_width  # Store custom quantization function

    def forward(self, input):
        quantized_weights = self.weight.std() * (self.weight.clamp(-1, 1).mul(2**self.bit_width).round() / (2**self.bit_width))

        # Perform convolution with quantized weights
        return nn.functional.conv2d(
            input, quantized_weights, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )
   
