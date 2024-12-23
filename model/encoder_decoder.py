import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from noise_layers.noiser import Noiser
from options import HiDDenConfiguration

class EncoderDecoder(nn.Module):
    """
    Combines Encoder -> Noiser -> Decoder into a single pipeline.
    The input is the cover image and the message to be hidden (watermark).
    The module inserts the message as a watermark in Fourier space, applies noise layers,
    and then tries to recover the message from the noised image.
    """
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(config)
        self.noiser = noiser
        self.decoder = Decoder(config)

    def forward(self, image, message):
        # Encode image with the message embedded in Fourier space
        encoded_image = self.encoder(image, message)
     
        # Apply noise layers to simulate distortions
        noised_and_cover = self.noiser([encoded_image, image])
        noised_image = noised_and_cover[0]
        
        # Decode to retrieve the message from the noised image
        decoded_message = self.decoder(noised_image)
        
        return encoded_image, noised_image, decoded_message
