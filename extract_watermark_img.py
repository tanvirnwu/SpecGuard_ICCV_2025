import torch
import os
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
import argparse
import cv2
from options import HiDDenConfiguration
from model.hidden import Hidden
from noise_layers.noiser import Noiser
import utils


def load_image(image_path, height, width, device):
    """Load and preprocess the input image."""
    image_pil = Image.open(image_path).convert("RGB")
    image = cv2.resize(np.array(image_pil), (width, height))
    image_tensor = TF.to_tensor(image).to(device)
    image_tensor = image_tensor * 2 - 1  # Scale from [0, 1] to [-1, 1]
    return image_tensor.unsqueeze(0)


def watermark_image(image_tensor, model, message):
    """Encode the message into the image using only the encoder."""
    with torch.no_grad():
        encoded_image = model.encoder_decoder.encoder(image_tensor, message)
    return encoded_image

# def watermark_image(image_tensor, model, message):
#     """Encode the message into the image using the encoder."""
#     with torch.no_grad():
#         encoded_image, _, _ = model.encoder_decoder.forward(image_tensor, message)
#     return encoded_image

def save_image(tensor, save_path, target_size=(512, 512)):
    """Save a tensor as an image with the specified resolution."""
    image_np = ((tensor.squeeze().cpu().numpy() + 1) / 2 * 255).astype(np.uint8).transpose(1, 2, 0)
    image_pil = Image.fromarray(image_np)
    image_pil = image_pil.resize(target_size, Image.LANCZOS)  # Resize to target size (512x512)
    image_pil.save(save_path)


def process_images(input_path, output_folder, model, hidden_config, device, excel_path):
    """Process a single image or all images in a directory and save embedding info to Excel."""
    os.makedirs(output_folder, exist_ok=True)
    records = []  # List to store image file names and message bits for Excel

    if os.path.isfile(input_path):
        # Process single image
        image_tensor = load_image(input_path, hidden_config.H, hidden_config.W, device)
        message = torch.Tensor(np.random.choice([0, 1], (1, hidden_config.message_length))).to(device)
        watermarked_image = watermark_image(image_tensor, model, message)
        output_path = os.path.join(output_folder, os.path.basename(input_path))
        save_image(watermarked_image, output_path)
        print(f"Watermarked image saved to: {output_path}")

        # Store record for Excel
        records.append({"File Name": os.path.basename(output_path), "Embedded Message": message.cpu().numpy().astype(int).tolist()})

    elif os.path.isdir(input_path):
        # Process directory of images
        for image_file in os.listdir(input_path):
            if image_file.endswith(('.jpg', '.png')):
                image_path = os.path.join(input_path, image_file)
                image_tensor = load_image(image_path, hidden_config.H, hidden_config.W, device)
                message = torch.Tensor(np.random.choice([0, 1], (1, hidden_config.message_length))).to(device)
                watermarked_image = watermark_image(image_tensor, model, message)
                output_path = os.path.join(output_folder, image_file)
                save_image(watermarked_image, output_path)
                print(f"Watermarked image saved to: {output_path}")

                # Store record for Excel
                records.append(
                    {"File Name": image_file, "Embedded Message": message.cpu().numpy().astype(int).tolist()})

    # Save records to Excel
    df = pd.DataFrame(records)
    df.to_excel(excel_path, index=False)
    print(f"Embedded messages saved to Excel at: {excel_path}")


def main():
    parser = argparse.ArgumentParser(description="Watermark an image or a directory of images and save the result.")
    parser.add_argument("--options-file", required=True, type=str, help="Path to options file")
    parser.add_argument("--checkpoint-file", required=True, type=str, help="Path to model checkpoint file")
    parser.add_argument("--input-path", required=True, type=str, help="Path to the input image or directory")
    parser.add_argument("--output-folder", required=True, type=str, help="Folder to save the watermarked images")
    parser.add_argument("--excel-path", required=True, type=str,
                        help="Path to save the Excel file with embedded messages")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model configuration and checkpoint
    train_options, hidden_config, noise_config = utils.load_options(args.options_file)
    noiser = Noiser(noise_config, device)
    checkpoint = torch.load(args.checkpoint_file)
    model = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(model, checkpoint)

    # Process single image or directory of images
    process_images(args.input_path, args.output_folder, model, hidden_config, device, args.excel_path)


if __name__ == "__main__":
    main()



"""
python extract_watermark_img.py --options-file "D:\Research\CVPR'24\Watermark\runs\celebhq-key-128\options-and-config.pickle" --checkpoint-file "D:\Research\CVPR'24\Watermark\runs\celebhq-key-128\checkpoints\celebhq--epoch-49.pyt" --input-path "D:\Research\CVPR'24\data\celeba_hq\mixed" --output-folder "D:\Research\CVPR'24\data\celeba_hq\wm_128" --excel-path "D:\Research\CVPR'24\data\celeba_hq/wm_128.xlsx"

"""



# import torch
# import os
# import numpy as np
# from PIL import Image
# import torchvision.transforms.functional as TF
# import argparse
# import cv2
# from options import HiDDenConfiguration
# from model.hidden import Hidden
# from noise_layers.noiser import Noiser
# import utils
#
# def load_image(image_path, height, width, device):
#     """Load and preprocess the input image."""
#     image_pil = Image.open(image_path).convert("RGB")
#     image = cv2.resize(np.array(image_pil), (width, height))
#     image_tensor = TF.to_tensor(image).to(device)
#     image_tensor = image_tensor * 2 - 1  # Scale from [0, 1] to [-1, 1]
#     return image_tensor.unsqueeze(0)
#
# def watermark_image(image_tensor, model, message):
#     """Encode the message into the image using the encoder."""
#     with torch.no_grad():
#         encoded_image, _, _ = model.encoder_decoder.forward(image_tensor, message)
#     return encoded_image
#
# def save_image(tensor, save_path, target_size=(512, 512)):
#     """Save a tensor as an image with the specified resolution."""
#     image_np = ((tensor.squeeze().cpu().numpy() + 1) / 2 * 255).astype(np.uint8).transpose(1, 2, 0)
#     image_pil = Image.fromarray(image_np)
#     image_pil = image_pil.resize(target_size, Image.LANCZOS)  # Resize to target size (512x512)
#     image_pil.save(save_path)
#
# def process_images(input_path, output_folder, model, hidden_config, device):
#     """Process a single image or all images in a directory."""
#     os.makedirs(output_folder, exist_ok=True)
#
#     if os.path.isfile(input_path):
#         # Single image
#         image_tensor = load_image(input_path, hidden_config.H, hidden_config.W, device)
#         message = torch.Tensor(np.random.choice([0, 1], (1, hidden_config.message_length))).to(device)
#         watermarked_image = watermark_image(image_tensor, model, message)
#         output_path = os.path.join(output_folder, os.path.basename(input_path))
#         save_image(watermarked_image, output_path)
#         print(f"Watermarked image saved to: {output_path}")
#
#     elif os.path.isdir(input_path):
#         # Directory of images
#         for image_file in os.listdir(input_path):
#             if image_file.endswith(('.jpg', '.png')):
#                 image_path = os.path.join(input_path, image_file)
#                 image_tensor = load_image(image_path, hidden_config.H, hidden_config.W, device)
#                 message = torch.Tensor(np.random.choice([0, 1], (1, hidden_config.message_length))).to(device)
#                 watermarked_image = watermark_image(image_tensor, model, message)
#                 output_path = os.path.join(output_folder, image_file)
#                 save_image(watermarked_image, output_path)
#                 print(f"Watermarked image saved to: {output_path}")
#
# def main():
#     parser = argparse.ArgumentParser(description="Watermark an image or a directory of images and save the result.")
#     parser.add_argument("--options-file", required=True, type=str, help="Path to options file")
#     parser.add_argument("--checkpoint-file", required=True, type=str, help="Path to model checkpoint file")
#     parser.add_argument("--input-path", required=True, type=str, help="Path to the input image or directory")
#     parser.add_argument("--output-folder", required=True, type=str, help="Folder to save the watermarked images")
#     args = parser.parse_args()
#
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#
#     # Load model configuration and checkpoint
#     train_options, hidden_config, noise_config = utils.load_options(args.options_file)
#     noiser = Noiser(noise_config, device)
#     checkpoint = torch.load(args.checkpoint_file)
#     model = Hidden(hidden_config, device, noiser, None)
#     utils.model_from_checkpoint(model, checkpoint)
#
#     # Process single image or directory of images
#     process_images(args.input_path, args.output_folder, model, hidden_config, device)
#
# if __name__ == "__main__":
#     main()
#
# # #python extract_watermark_img.py --options-file "D:\Research\CVPR'24\Watermark\runs\celebhq-key-128\options-and-config.pickle" --checkpoint-file "D:\Research\CVPR'24\Watermark\runs\celebhq-key-128\checkpoints\celebhq--epoch-49.pyt" --input-path "D:\Research\CVPR'24\data\celeba_hq\mixed" --output-folder "D:\Research\CVPR'24\data\celeba_hq\wm_128"
# # #python extract_watermark_img.py --options-file "D:\Research\CVPR'24\Watermark\runs\celebhq-subset-no-noise\options-and-config.pickle" --checkpoint-file "D:\Research\CVPR'24\Watermark\runs\celebhq-subset-no-noise\checkpoints\celebhq-subset--epoch-96.pyt" --input-path "D:\Research\CVPR'24\data\CelebA-HQ-img\CelebA-HQ-img\target\5035.jpg" --output-folder "D:\Research\CVPR'24\data\celeba_hq\watermarked"
