import torch
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from options import HiDDenConfiguration
from model.hidden import Hidden
from noise_layers.noiser import Noiser
import utils
from PIL import Image
import torchvision.transforms.functional as TF

def load_image(image_path, height, width, device):
    """Loads an image, resizes it, and prepares it for the model."""
    image_pil = Image.open(image_path).convert("RGB")
    image = np.array(image_pil)
    image = TF.to_tensor(image)
    image = TF.resize(image, [height, width])
    image = image * 2 - 1  # Scale from [0, 1] to [-1, 1]
    return image.unsqueeze(0).to(device)

def decode_image(model, image_tensor):
    """Passes an image through the decoder to extract the embedded message."""
    with torch.no_grad():
        _, _, decoded_message = model.encoder_decoder(image_tensor, torch.zeros(1, model.config.message_length).to(image_tensor.device))
    return decoded_message.cpu().numpy().round()

def main():
    parser = argparse.ArgumentParser(description="Evaluate face-swapped images for watermark integrity.")
    parser.add_argument("--options-file", required=True, type=str, help="Path to the options file.")
    parser.add_argument("--checkpoint-file", required=True, type=str, help="Path to the model checkpoint file.")
    parser.add_argument("--watermarked-image", required=True, type=str, help="Path to the watermarked input image.")
    parser.add_argument("--faceswapped-image", required=True, type=str, help="Path to the face-swapped image.")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model configuration, noiser, and checkpoint
    train_options, hidden_config, noise_config = utils.load_options(args.options_file)
    noiser = Noiser(noise_config, device)
    model = Hidden(hidden_config, device, noiser, None)
    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    utils.model_from_checkpoint(model, checkpoint)

    # Load images
    watermarked_image_tensor = load_image(args.watermarked_image, hidden_config.H, hidden_config.W, device)
    faceswapped_image_tensor = load_image(args.faceswapped_image, hidden_config.H, hidden_config.W, device)

    # Decode messages
    decoded_bits_watermarked = decode_image(model, watermarked_image_tensor)
    decoded_bits_faceswapped = decode_image(model, faceswapped_image_tensor)

    # Calculate bit accuracy
    bit_accuracy = accuracy_score(decoded_bits_watermarked.flatten(), decoded_bits_faceswapped.flatten())

    # Print results
    print("Extracted Bits (Watermarked Image):", decoded_bits_watermarked)
    print("Extracted Bits (Face-Swapped Image):", decoded_bits_faceswapped)
    print(f"Bit Accuracy: {bit_accuracy:.3f}")

if __name__ == "__main__":
    main()


# python faceswap_attack.py --options-file "D:\Research\CVPR'24\Watermark\runs\celebhq-subset-no-noise\options-and-config.pickle" --checkpoint-file "D:\Research\CVPR'24\Watermark\runs\celebhq-subset-no-noise\checkpoints\celebhq-subset--epoch-96.pyt" --watermarked-image "D:\Research\CVPR'24\data\celeba_hq\watermarked\5035.jpg" --faceswapped-image "D:\Research\CVPR'24\data\celeba_hq\faceswaped\5035_res.png"
