import torch
import torch.nn as nn
import numpy as np
from options import HiDDenConfiguration
from model.encoder_decoder import EncoderDecoder
from model.discriminator import Discriminator
from noise_layers.noiser import Noiser

class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, noiser: Noiser, tb_logger):
        self.encoder_decoder = EncoderDecoder(configuration, noiser).to(device)
        self.discriminator = Discriminator(configuration).to(device)

        # Separate parameters for BER-specific gradients (typically, decoder parameters)
        ber_params = list(self.encoder_decoder.decoder.parameters())
        other_params = list(self.encoder_decoder.encoder.parameters())
        
        # #Freeze encoder parameters
        # for param in self.encoder_decoder.decoder.parameters():
        #     param.requires_grad = False
        
        # Optimizer for only the decoder parameters
        # Use two optimizers with different learning rates
        self.optimizer_ber = torch.optim.Adam(ber_params, lr=0.001)
        # Initialize a learning rate scheduler for the BER optimizer
        self.scheduler_ber = torch.optim.lr_scheduler.StepLR(self.optimizer_ber, step_size=10, gamma=0.5)

        self.optimizer_other = torch.optim.Adam(other_params, lr=0.01)
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())
        
        self.config = configuration
        self.device = device

        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        self.cover_label = 1
        self.encoded_label = 0

    def calculate_ber(self, decoded_message, original_message):
        # decoded_message and original_message should be binary (0 and 1)
        bit_errors = (decoded_message.round() != original_message).float()
        ber = bit_errors.mean()
        return ber

    def train_on_batch(self, batch: list):
        images, messages = batch
        batch_size = images.shape[0]

        images = images.to(self.device)
        messages = messages.to(self.device)

        self.encoder_decoder.train()
        self.discriminator.train()
        with torch.enable_grad():
            # ---------------- Train the discriminator -----------------------------
            self.optimizer_discrim.zero_grad()
            # train on cover
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device, dtype=torch.float)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device, dtype=torch.float)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device, dtype=torch.float)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            d_loss_on_cover.backward()

            # train on fake
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
            # ber = (decoded_messages != messages).float().mean() * 100
            # print(f"Bit Error Rate (BER): {ber.item():.2f}%")
            decoded_messages = decoded_messages.to(self.device)
            d_on_encoded = self.discriminator(encoded_images.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)

            d_loss_on_encoded.backward()
            self.optimizer_discrim.step()

            # --------------Train the generator (encoder-decoder) ---------------------
            # Zero gradients for both optimizers
            self.optimizer_ber.zero_grad()
            self.optimizer_other.zero_grad()
            # target label for encoded images should be 'cover', because we want to fool the discriminator
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)

            g_loss_enc = self.mse_loss(encoded_images, images)

            # print(f"Device of images: {images.device}")
            # print(f"Device of messages: {messages.device}")
            # print(f"Device of encoded_images: {encoded_images.device}")
            # print(f"Device of decoded_messages: {decoded_messages.device}")
            g_loss_dec = self.mse_loss(decoded_messages, messages)
            ber_loss = self.calculate_ber(decoded_messages, messages)  # BER-based loss
            g_loss = (
            self.config.encoder_loss * g_loss_enc +
            self.config.decoder_loss * g_loss_dec +
            self.config.ber_loss_weight * ber_loss  # Weight for BER loss
                )

            g_loss.backward()
            # Update parameters
            self.scheduler_ber.step()      # Update BER-specific parameters
            self.optimizer_other.step()    # Update other parameters

        # Log the BER and other loss components for monitoring
        losses = {
            'total_loss      ': g_loss.item(),
            'encoder_mse     ': g_loss_enc.item(),
            'decoder_mse     ': g_loss_dec.item(),
            'ber_loss        ': ber_loss.item(),  # Log the BER
        }
        
        return losses, (encoded_images, noised_images, decoded_messages)

    def validate_on_batch(self, batch: list):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param batch: batch of validation data, in form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        # # if TensorboardX logging is enabled, save some of the tensors.
        # if self.tb_logger is not None:
        #     encoder_final = self.encoder_decoder.encoder._modules['final_layer']
        #     self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)
        #     decoder_final = self.encoder_decoder.decoder._modules['linear']
        #     self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)
        #     discrim_final = self.discriminator._modules['linear']
        #     self.tb_logger.add_tensor('weights/discrim_out', discrim_final.weight)

        images, messages = batch

        batch_size = images.shape[0]

        images = images.to(self.device)
        messages = messages.to(self.device)

        self.encoder_decoder.eval()
        self.discriminator.eval()
        with torch.no_grad():
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device, dtype=torch.float)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device, dtype=torch.float)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device, dtype=torch.float)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)

            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            d_on_encoded = self.discriminator(encoded_images)
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)

            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)

            
            g_loss_enc = self.mse_loss(encoded_images, images)

            # ber = (decoded_messages != messages).float().mean() * 100
            # print(f"Bit Error Rate (BER): {ber.item():.2f}%")
            decoded_messages = decoded_messages.to(self.device)
            g_loss_dec = self.mse_loss(decoded_messages, messages)
            ber_loss = self.calculate_ber(decoded_messages, messages)  # BER-based loss
            g_loss = (
            self.config.encoder_loss * g_loss_enc +
            self.config.decoder_loss * g_loss_dec +
            self.config.ber_loss_weight * ber_loss  # Weight for BER loss
                )

        # Log the BER and other loss components for monitoring
        losses = {
            'total_loss      ': g_loss.item(),
            'encoder_mse     ': g_loss_enc.item(),
            'decoder_mse     ': g_loss_dec.item(),
            'ber_loss        ': ber_loss.item(),  # Log the BER
        }
        
        return losses, (encoded_images, noised_images, decoded_messages)

    def to_stirng(self):
        return '{}\n{}'.format(str(self.encoder_decoder), str(self.discriminator))