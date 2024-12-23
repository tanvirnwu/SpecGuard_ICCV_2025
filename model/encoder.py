import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import math
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse
from model.conv_bn_relu import ConvBNRelu
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

class Encoder(nn.Module):
    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks
        self.watermark_radius = config.watermark_radius
        self.strength = 50  # Embedding strength for robustness

        # Encoder layers
        layers = [ConvBNRelu(3, self.conv_channels)]
        for _ in range(config.encoder_blocks - 1):
            layers.append(ConvBNRelu(self.conv_channels, self.conv_channels))

        self.conv_layers = nn.Sequential(*layers)
        # self.after_watermark_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
        #                                      self.conv_channels)
        self.after_watermark_layer = ConvBNRelu(self.conv_channels,
                                             self.conv_channels)
        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

    def apply_dct_watermark(self, dct_coeffs, message):
        """Embed the watermark message into the high-frequency DCT coefficients."""
        
        # Ensure tensors are on the same device
        device = dct_coeffs.device
        message = message.to(device)

        batch_size, channels, height, width = dct_coeffs.shape
        center = (height // 2, width // 2)

        # Create a grid of distances from the center
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device) - center[0],
            torch.arange(width, device=device) - center[1]
        )
        distance_matrix = torch.sqrt(y_coords**2 + x_coords**2)

        # Create a mask for the high-frequency components within the watermark_radius
        mask = (distance_matrix <= self.watermark_radius)

        # Get the indices where the mask is True (i.e., within the radius)
        masked_indices = torch.nonzero(mask, as_tuple=True)

        # Adjust message to match batch_size and channels for embedding
        # Now `message` has shape [batch_size, message_length], and we repeat across channels
        message_length = message.shape[1]
        repeated_message = message[:, None, :].expand(batch_size, channels, message_length)

        # Determine the number of elements in the message and adjust the mask to fit
        num_message_elements = repeated_message.shape[-1]
        selected_indices = (masked_indices[0][:num_message_elements],
                            masked_indices[1][:num_message_elements])

        # Apply the watermark to the DCT coefficients at selected indices
        dct_coeffs[:, 0, selected_indices[0], selected_indices[1]] += (
            repeated_message[:, 0, :num_message_elements] * self.strength
        )
        return dct_coeffs



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

        def idwt2(self, coeffs):
            """Apply 2D Inverse Discrete Wavelet Transform using pytorch_wavelets on the specified device."""
            # Unpack the coefficients
            LL, HH = coeffs
            
            # Ensure coefficients are on the correct device
            #LL, HH = LL.to(self.device),  HH.to(self.device)
            
            # Apply DWTInverse (IDWT) to reconstruct the image
            reconstructed_image = self.idwt((LL, HH))
            
            return reconstructed_image



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

    def idct(self, X, norm=None):
        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= math.sqrt(N) * 2
            X_v[:, 1:] *= math.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * math.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.complex(V_r, V_i)

        v = torch.fft.irfft(V, n=N, dim=1)
        # Reshape back to the original shape and combine parts
        x = torch.zeros_like(X_v)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.view(*x_shape)

    def dct_2d(self, x):
        """Apply 2D DCT by applying 1D DCT twice (along height and width)."""
        # Apply DCT along the height (last dimension)
        x = self.dct(x)
        # Apply DCT along the width (second-last dimension)
        x = self.dct(x.transpose(-1, -2)).transpose(-1, -2)
        return x

    def idct_2d(self, x):
        """Apply 2D IDCT by applying 1D IDCT twice (along height and width)."""
        # Apply IDCT along the height (last dimension)
        x = self.idct(x)
        # Apply IDCT along the width (second-last dimension)
        x = self.idct(x.transpose(-1, -2)).transpose(-1, -2)
        return x


    def forward(self, image, message):
        wavelet_transforms = self.WaveletTransforms(wave='db1')
        image = image.to(wavelet_transforms.device)
        #print (image.shape)
        # Convert image to DWT and DCT space
        LL, HH = wavelet_transforms.dwt2(image)
        for i, element in enumerate(HH):
           # print(f"Element {i}: {type(element)}, Shape: {getattr(element, 'shape', 'N/A')}")
            HH_high = element[:, :, 2, :, :]
        #print (HH.shape)    
        #print (HH_high.shape)
        HH_dct = self.dct_2d(HH_high)
         # Ensure the shape is (batch_size, channels, height, width)
        # if HH_dct.shape[0] != image.shape[0]:
        #     HH_dct = HH_dct.permute(1, 0, 2, 3)  # Swap batch and channel dimensions if necessary
        #     LH = LH.permute(1, 0, 2, 3)
        #     HL = HL.permute(1, 0, 2, 3)
        #     LL = LL.permute(1, 0, 2, 3)
        # Pass HH_dct through convolutional layers
        HH_dct_1 = self.conv_layers(HH_dct)
        # expanded_message = message.unsqueeze(-1)
        # expanded_message.unsqueeze_(-1)

        # expanded_message = expanded_message.expand(-1,-1, self.H//2, self.W//2)
        # concat = torch.cat([expanded_message, HH_dct_1, HH_dct], dim=1)

        # Apply DCT-based watermark embedding
        HH_dct_watermarked = self.apply_dct_watermark(HH_dct_1, message)
        im_w = self.after_watermark_layer(HH_dct_watermarked)
        im_w = self.final_layer(im_w)

        # Transform back to spatial domain
        im_w = self.idct_2d(im_w)
        # im_w = self.after_watermark_layer(HH_with_watermark)
        # im_w = self.final_layer(im_w)
        # print (im_w.shape)
        # print (LL.shape,HL.shape,LH.shape)
        for i, element in enumerate(HH):
            element[:, :, 2, :, :] = im_w
        
        watermarked_image = wavelet_transforms.idwt2((LL,HH))

        # # Continue with neural network layers
        # expanded_message = message.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.H, self.W)
        # concat = torch.cat([expanded_message, watermarked_image, image], dim=1)
        
        # im_w = self.final_layer(im_w)
        return watermarked_image


