import torch
import torch.nn as nn
import pywt
import numpy as np
import math
from model.conv_bn_relu import ConvBNRelu
from pytorch_wavelets import DWTForward, DWTInverse
from options import HiDDenConfiguration
import torch.fft
# try:
#     # PyTorch 1.7.0 and newer versions
#     import torch.fft

#     def dct1_rfft_impl(x):
#         return torch.view_as_real(torch.fft.rfft(x, dim=1))
    
#     def dct_fft_impl(v):
#         return torch.view_as_real(torch.fft.fft(v, dim=1))

#     def idct_irfft_impl(V):
#         return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
# except ImportError:
#     # PyTorch 1.6.0 and older versions
#     def dct1_rfft_impl(x):
#         return torch.rfft(x, 1)
    
#     def dct_fft_impl(v):
#         return torch.rfft(v, 1, onesided=False)

#     def idct_irfft_impl(V):
#         return torch.irfft(V, 1, onesided=False)

class Decoder(nn.Module):
    """
    Decoder module to extract the embedded message from the HH DCT coefficients of each channel.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Decoder, self).__init__()
        self.channels = config.decoder_channels
        self.message_length = config.message_length
        self.watermark_radius = config.watermark_radius  # Set watermark radius from config
        self.strength = 50
        #self.threshold = nn.Parameter(torch.tensor(0.5))

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(config.decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))
        
        # layers.append(ConvBNRelu(self.channels, self.message_length))
        # layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        #self.linear = nn.Linear(config.message_length, config.message_length)
    class WaveletTransforms:
        def __init__(self, wave='db1', device='cuda' if torch.cuda.is_available() else 'cpu'):
            # Set device (default to CUDA if available)
            self.device = device
            
            # Initialize DWT and IDWT with the specified wavelet type
            self.dwt = DWTForward(J=1, wave=wave, mode='zero').to(self.device)
            self.idwt = DWTInverse(wave=wave, mode='zero').to(self.device)

        def dwt2(self, x):
            """Apply 2D Discrete Wavelet Transform using pytorch_wavelets on the specified device."""
            # Ensure input is on the correct device
            x = x.to(self.device)
            
            # Apply DWTForward (DWT) on the input
            LL, HH = self.dwt(x)
            
            # Return the coefficients
            return LL, HH

    def dct(self, x, norm=None):
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = torch.fft.fft(v, dim=1)

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc.real * W_r - Vc.imag * W_i

        if norm == 'ortho':
            V[:, 0] /= math.sqrt(N) * 2
            V[:, 1:] /= math.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)

        return V


    def dct_2d(self, x):
        """Apply 2D DCT by applying 1D DCT twice (along height and width)."""
        # Apply DCT along the height (last dimension)
        x = self.dct(x)
        # Apply DCT along the width (second-last dimension)
        x = self.dct(x.transpose(-1, -2)).transpose(-1, -2)
        return x



    
    def extract_dct_watermark(self, image_dct, expected_length):
        device = image_dct.device
        batch_size, channels, height, width = image_dct.shape
        center = (height // 2, width // 2)

        # Calculate distance and mask for high-frequency region
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device) - center[0],
            torch.arange(width, device=device) - center[1]
        )
        distance_matrix = torch.sqrt(y_coords**2 + x_coords**2)
        mask = (distance_matrix <= self.watermark_radius)
        masked_indices = torch.nonzero(mask, as_tuple=True)
        selected_indices = (masked_indices[0][:expected_length],
                            masked_indices[1][:expected_length])

        # Extract watermark data from specified positions
        extracted_values = image_dct[:, 0, selected_indices[0], selected_indices[1]]

        #decoded_message = (extracted_values > self.threshold).float()
        # decoded_message = (extracted_values > self.strength / 2).float()
        # Use a dynamic threshold for decoding
        threshold = extracted_values.mean(dim=1, keepdim=True)
        decoded_message = (extracted_values > threshold).float()
        return decoded_message

    def forward(self, image_with_wm):
        wavelet_transforms = self.WaveletTransforms(wave='db1')
        image = image_with_wm.to(wavelet_transforms.device)

        LL, HH = wavelet_transforms.dwt2(image)
        for i, element in enumerate(HH):
           # print(f"Element {i}: {type(element)}, Shape: {getattr(element, 'shape', 'N/A')}")
            HH_high = element[:, :, 2, :, :]
        HH_dct = self.dct_2d(HH_high)
        x = self.layers(HH_dct)
        extracted_message = self.extract_dct_watermark(x, self.message_length)

        # # Perform DWT on the output to retrieve the HH band
        # _, (_, _, HH) = self.dwt2(image_with_wm)
        # # Apply DCT on HH band
        # HH_dct = self.dct_2d(HH)
        #  # Ensure the shape is (batch_size, channels, height, width)
        # if HH_dct.shape[0] != image_with_wm.shape[0]:
        #     HH_dct = HH_dct.permute(1, 0, 2, 3)  # Swap batch and channel dimensions if necessary

        # # Apply convolution layers to preprocess
        # x = self.layers(HH_dct)
        # #Extract message from DCT coefficients in HH band
        # extracted_message = self.extract_dct_watermark(x, self.message_length)
        return extracted_message
        # x.squeeze_(3).squeeze_(2)
        # x = self.linear(x)
        # return x
