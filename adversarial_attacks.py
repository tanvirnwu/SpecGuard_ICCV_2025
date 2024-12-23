import torch
import numpy as np
import argparse
import cv2
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from options import HiDDenConfiguration
from model.hidden import Hidden
from noise_layers.noiser import Noiser
import utils
from PIL import Image
import torchvision.transforms.functional as TF


def load_image(path, height, width, device):
    image_pil = Image.open(path)
    image = cv2.resize(np.array(image_pil), (width, height))
    image_tensor = TF.to_tensor(image).to(device)
    image_tensor = image_tensor * 2 - 1  # Scale from [0, 1] to [-1, 1]
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor, np.array(image_pil)


def apply_attack(watermarked_image, attack_type):
    attacked_image = watermarked_image.clone().detach().cpu().numpy().squeeze().transpose(1, 2, 0) * 255
    attacked_image = attacked_image.astype(np.uint8)

    if attack_type == "Blur":
        attacked_image = cv2.GaussianBlur(attacked_image, (5, 5), 0)
    elif attack_type == "Bright":
        attacked_image = cv2.convertScaleAbs(attacked_image, alpha=1.2, beta=30)
    elif attack_type == "Contrast":
        attacked_image = cv2.convertScaleAbs(attacked_image, alpha=1.5, beta=0)
    elif attack_type == "Crop":
        h, w, _ = attacked_image.shape
        attacked_image = attacked_image[h // 4:h * 3 // 4, w // 4:w * 3 // 4]
        attacked_image = cv2.resize(attacked_image, (w, h))
    elif attack_type == "Noise":
        noise = np.random.normal(0, 0, attacked_image.shape).astype(np.uint8)
        attacked_image = cv2.add(attacked_image, noise)
    elif attack_type == "JPEG":
        _, encoded_image = cv2.imencode('.jpg', attacked_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        attacked_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    elif attack_type == "Rotation":
        h, w, _ = attacked_image.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1)
        attacked_image = cv2.warpAffine(attacked_image, matrix, (w, h))

    attacked_image_tensor = TF.to_tensor(attacked_image).unsqueeze(0) * 2 - 1
    return attacked_image_tensor.to(watermarked_image.device), attacked_image


def evaluate_message(embedded_message, decoded_message):
    bit_accuracy = accuracy_score(embedded_message, decoded_message)
    return bit_accuracy


def main():
    parser = argparse.ArgumentParser(description="Apply adversarial attacks and evaluate on multiple images")
    parser.add_argument("--options-file", required=True, type=str, help="Options file path")
    parser.add_argument("--checkpoint-file", required=True, type=str, help="Model checkpoint file")
    parser.add_argument("--folder-path", required=True, type=str, help="Folder path containing input images")
    parser.add_argument("--attack-type", required=True, type=str, help="Type of attack")
    parser.add_argument("--excel-file", required=True, type=str, help="Path to the Excel file with embedded messages")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load options and model
    train_options, hidden_config, noise_config = utils.load_options(args.options_file)
    noiser = Noiser(noise_config, device)
    checkpoint = torch.load(args.checkpoint_file)
    model = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(model, checkpoint)

    # Load Excel file with embedded messages
    excel_data = pd.read_excel(args.excel_file)
    excel_data['Embedded Message'] = excel_data['Embedded Message'].apply(
        lambda x: np.array(eval(x)).flatten()
    )

    # Initialize variables for accumulating results
    total_bit_accuracy = 0
    num_images = 0

    # Process each image in the folder
    for _, row in excel_data.iterrows():
        image_file = row['File Name']
        embedded_message = row['Embedded Message']
        image_path = os.path.join(args.folder_path, image_file)

        if not os.path.exists(image_path):
            print(f"Image {image_file} not found in the folder.")
            continue

        # Load image
        image_tensor, original_np = load_image(image_path, hidden_config.H, hidden_config.W, device)

        # Encode (skip actual encoding as per requirement)
        encoded_image = image_tensor

        # Apply the selected adversarial attack
        attacked_image_tensor, _ = apply_attack(encoded_image, args.attack_type)

        # # Decode the message from the attacked image
        # _, _, decoded_message_attacked = model.encoder_decoder.forward(attacked_image_tensor, None)
        #
        # # Flatten and round decoded message for comparison
        # decoded_message_attacked = decoded_message_attacked.cpu().numpy().round().flatten()
        # Decode the message from the attacked image
        decoded_message_attacked = model.encoder_decoder.decoder(attacked_image_tensor)

        # Flatten and round decoded message for comparison
        decoded_message_attacked = decoded_message_attacked.cpu().numpy().round().flatten()

        # Evaluate message extraction and accumulate bit accuracy
        bit_accuracy = evaluate_message(embedded_message, decoded_message_attacked)
        total_bit_accuracy += bit_accuracy
        num_images += 1

    # Calculate and print the average bit accuracy
    avg_bit_accuracy = total_bit_accuracy / num_images if num_images > 0 else 0
    print(f"Attack Type: {args.attack_type}")
    print(f"Processed {num_images} images")
    print(f"Average Bit Accuracy: {avg_bit_accuracy:.3f}")


if __name__ == "__main__":
    main()

"""
python adversarial_attacks.py --options-file "D:\Research\CVPR'24\Watermark\runs\celebhq-key-128\options-and-config.pickle" --checkpoint-file "D:\Research\CVPR'24\Watermark\runs\celebhq-key-128\checkpoints\celebhq--epoch-49.pyt" --folder-path "D:\Research\CVPR'24\data\celeba_hq\wm_128" --attack-type "Noise" --excel-file "D:\Research\CVPR'24\data\celeba_hq\wm_128.xlsx"

"""

# import torch
# import numpy as np
# import argparse
# import cv2
# import os
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, roc_auc_score
# from options import HiDDenConfiguration
# from model.hidden import Hidden
# from noise_layers.noiser import Noiser
# import utils
# from PIL import Image
# import torchvision.transforms.functional as TF
# from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
#
#
# def load_image(path, height, width, device):
#     image_pil = Image.open(path)
#     image = cv2.resize(np.array(image_pil), (width, height))
#     image_tensor = TF.to_tensor(image).to(device)
#     image_tensor = image_tensor * 2 - 1  # Scale from [0, 1] to [-1, 1]
#     image_tensor = image_tensor.unsqueeze(0)
#     return image_tensor, np.array(image_pil)
#
#
# def apply_attack(watermarked_image, attack_type):
#     attacked_image = watermarked_image.clone().detach().cpu().numpy().squeeze().transpose(1, 2, 0) * 255
#     attacked_image = attacked_image.astype(np.uint8)
#
#     if attack_type == "Blur":
#         attacked_image = cv2.GaussianBlur(attacked_image, (5, 5), 0)
#     elif attack_type == "Bright":
#         attacked_image = cv2.convertScaleAbs(attacked_image, alpha=1.2, beta=30)
#     elif attack_type == "Contrast":
#         attacked_image = cv2.convertScaleAbs(attacked_image, alpha=1.5, beta=0)
#     elif attack_type == "Crop":
#         h, w, _ = attacked_image.shape
#         attacked_image = attacked_image[h // 4:h * 3 // 4, w // 4:w * 3 // 4]
#         attacked_image = cv2.resize(attacked_image, (w, h))
#     elif attack_type == "Noise":
#         noise = np.random.normal(0, 0, attacked_image.shape).astype(np.uint8)
#         attacked_image = cv2.add(attacked_image, noise)
#     elif attack_type == "JPEG":
#         _, encoded_image = cv2.imencode('.jpg', attacked_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
#         attacked_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
#     elif attack_type == "Rotation":
#         h, w, _ = attacked_image.shape
#         matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1)
#         attacked_image = cv2.warpAffine(attacked_image, matrix, (w, h))
#
#     attacked_image_tensor = TF.to_tensor(attacked_image).unsqueeze(0) * 2 - 1
#     return attacked_image_tensor.to(watermarked_image.device), attacked_image
#
#
# def calculate_quality(original_np, attacked_np):
#     if attacked_np.shape[0] == 3:
#         attacked_np = attacked_np.transpose(1, 2, 0)
#     attacked_resized = cv2.resize(attacked_np, (original_np.shape[1], original_np.shape[0]))
#     psnr_value = psnr(original_np, attacked_resized, data_range=255)
#     ssim_value = ssim(original_np, attacked_resized, data_range=255, channel_axis=-1)
#     degradation = (1 - ssim_value) * 100  # Percentage of quality degradation
#     return psnr_value, ssim_value, degradation
#
#
# def evaluate_message(original_message, decoded_message):
#     original_message = original_message.cpu().numpy().round()
#     decoded_message = decoded_message.cpu().numpy().round()
#     bit_accuracy = accuracy_score(original_message, decoded_message)
#     return bit_accuracy
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Apply adversarial attacks and evaluate on multiple images")
#     parser.add_argument("--options-file", required=True, type=str, help="Options file path")
#     parser.add_argument("--checkpoint-file", required=True, type=str, help="Model checkpoint file")
#     parser.add_argument("--folder-path", required=True, type=str, help="Folder path containing input images")
#     parser.add_argument("--attack-type", required=True, type=str, help="Type of attack")
#     args = parser.parse_args()
#
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#
#     # Load options and model
#     train_options, hidden_config, noise_config = utils.load_options(args.options_file)
#     noiser = Noiser(noise_config, device)
#     checkpoint = torch.load(args.checkpoint_file)
#     model = Hidden(hidden_config, device, noiser, None)
#     utils.model_from_checkpoint(model, checkpoint)
#
#     # Initialize variables for accumulating results
#     total_bit_accuracy = 0
#     num_images = 0
#
#     # Process each image in the folder
#     for image_file in os.listdir(args.folder_path):
#         image_path = os.path.join(args.folder_path, image_file)
#         if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#             continue
#
#         # Load image and create a random message
#         image_tensor, original_np = load_image(image_path, hidden_config.H, hidden_config.W, device)
#         message = torch.Tensor(np.random.choice([0, 1], (1, hidden_config.message_length))).to(device)
#
#         # Encode the message into the image
#         encoded_image, _, decoded_message = model.encoder_decoder.forward(image_tensor, message)
#
#         # Apply the selected adversarial attack
#         attacked_image_tensor, attacked_np = apply_attack(encoded_image, args.attack_type)
#
#         # Decode the message from the attacked image
#         _, _, decoded_message_attacked = model.encoder_decoder.forward(attacked_image_tensor, message)
#
#         # Evaluate message extraction and accumulate bit accuracy
#         bit_accuracy = evaluate_message(message, decoded_message_attacked)
#         total_bit_accuracy += bit_accuracy
#         num_images += 1
#
#     # Calculate and print the average bit accuracy
#     avg_bit_accuracy = total_bit_accuracy / num_images if num_images > 0 else 0
#     print(f"Attack Type: {args.attack_type}")
#     print(f"Processed {num_images} images")
#     print(f"Average Bit Accuracy: {avg_bit_accuracy:.3f}")
#
#
# if __name__ == "__main__":
#     main()


"""
python adversarial_attacks.py --options-file "D:\Research\CVPR'24\Watermark\runs\celebhq-key-128\options-and-config.pickle" --checkpoint-file "D:\Research\CVPR'24\Watermark\runs\celebhq-key-128\checkpoints\celebhq--epoch-49.pyt" --folder-path "D:\Research\CVPR'24\data\celeba_hq\wm_128" --attack-type "Noise"

"""

# import torch
# import numpy as np
# import argparse
# import cv2
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, roc_auc_score
# from options import HiDDenConfiguration
# from model.hidden import Hidden
# from noise_layers.noiser import Noiser
# import utils
# from PIL import Image
# import torchvision.transforms.functional as TF
# from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
#
#
# def load_image(path, height, width, device):
#     image_pil = Image.open(path)
#     image = cv2.resize(np.array(image_pil), (width, height))
#     image_tensor = TF.to_tensor(image).to(device)
#     image_tensor = image_tensor * 2 - 1  # Scale from [0, 1] to [-1, 1]
#     image_tensor = image_tensor.unsqueeze(0)
#     return image_tensor, np.array(image_pil)
#
#
# def apply_attack(watermarked_image, attack_type):
#     attacked_image = watermarked_image.clone().detach().cpu().numpy().squeeze().transpose(1, 2, 0) * 255
#     attacked_image = attacked_image.astype(np.uint8)
#
#     if attack_type == "Blur":
#         attacked_image = cv2.GaussianBlur(attacked_image, (5, 5), 0)
#     elif attack_type == "Bright":
#         attacked_image = cv2.convertScaleAbs(attacked_image, alpha=1.2, beta=30)
#     elif attack_type == "Contrast":
#         attacked_image = cv2.convertScaleAbs(attacked_image, alpha=1.5, beta=0)
#     elif attack_type == "Crop":
#         h, w, _ = attacked_image.shape
#         attacked_image = attacked_image[h // 4:h * 3 // 4, w // 4:w * 3 // 4]
#         attacked_image = cv2.resize(attacked_image, (w, h))
#     elif attack_type == "Noise":
#         noise = np.random.normal(0, 0, attacked_image.shape).astype(np.uint8)
#         attacked_image = cv2.add(attacked_image, noise)
#     elif attack_type == "JPEG":
#         _, encoded_image = cv2.imencode('.jpg', attacked_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
#         attacked_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
#     elif attack_type == "Rotation":
#         h, w, _ = attacked_image.shape
#         matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1)
#         attacked_image = cv2.warpAffine(attacked_image, matrix, (w, h))
#
#     attacked_image_tensor = TF.to_tensor(attacked_image).unsqueeze(0) * 2 - 1
#     return attacked_image_tensor.to(watermarked_image.device), attacked_image
#
#
# def calculate_quality(original_np, attacked_np):
#     if attacked_np.shape[0] == 3:
#         attacked_np = attacked_np.transpose(1, 2, 0)
#     attacked_resized = cv2.resize(attacked_np, (original_np.shape[1], original_np.shape[0]))
#     psnr_value = psnr(original_np, attacked_resized, data_range=255)
#     ssim_value = ssim(original_np, attacked_resized, data_range=255, channel_axis=-1)
#     degradation = (1 - ssim_value) * 100  # Percentage of quality degradation
#     return psnr_value, ssim_value, degradation
#
#
# def evaluate_message(original_message, decoded_message):
#     original_message = original_message.cpu().numpy().round()
#     decoded_message = decoded_message.cpu().numpy().round()
#     bit_accuracy = accuracy_score(original_message, decoded_message)
#     true_positives = np.sum((original_message == 1) & (decoded_message == 1))
#     false_positives = np.sum((original_message == 0) & (decoded_message == 1))
#     true_positive_rate = true_positives / np.sum(original_message == 1)
#     false_positive_rate = false_positives / np.sum(original_message == 0)
#     auc = roc_auc_score(original_message.flatten(), decoded_message.flatten())
#     return bit_accuracy, true_positive_rate, false_positive_rate, auc, decoded_message
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Apply adversarial attacks and evaluate")
#     parser.add_argument("--options-file", required=True, type=str, help="Options file path")
#     parser.add_argument("--checkpoint-file", required=True, type=str, help="Model checkpoint file")
#     parser.add_argument("--image-path", required=True, type=str, help="Path to the input image")
#     parser.add_argument("--attack-type", required=True, type=str, help="Type of attack")
#     args = parser.parse_args()
#
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#
#     # Load options and model
#     train_options, hidden_config, noise_config = utils.load_options(args.options_file)
#     noiser = Noiser(noise_config, device)
#     checkpoint = torch.load(args.checkpoint_file)
#     model = Hidden(hidden_config, device, noiser, None)
#     utils.model_from_checkpoint(model, checkpoint)
#
#     # Load image and create a random message
#     image_tensor, original_np = load_image(args.image_path, hidden_config.H, hidden_config.W, device)
#     message = torch.Tensor(np.random.choice([0, 1], (1, hidden_config.message_length))).to(device)
#
#     # Encode the message into the image
#     encoded_image, _, decoded_message = model.encoder_decoder.forward(image_tensor, message)
#     encoded_np = ((encoded_image.squeeze().detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8).transpose(1, 2, 0)
#
#     # Extract and display the decoded message for the watermarked image
#     print("Decoded message from watermarked image:", decoded_message.cpu().numpy().round())
#
#     # Apply the selected adversarial attack
#     attacked_image_tensor, attacked_np = apply_attack(encoded_image, args.attack_type)
#     attacked_np = ((attacked_image_tensor.squeeze().cpu().numpy() + 1) / 2 * 255).astype(np.uint8).transpose(1, 2, 0)
#
#     # Decode the message from the attacked image
#     _, _, decoded_message_attacked = model.encoder_decoder.forward(attacked_image_tensor, message)
#
#     # Extract and display the decoded message for the attacked image
#     print("Decoded message from attacked image:", decoded_message_attacked.cpu().numpy().round())
#
#     # Calculate image quality degradation
#     psnr_value, ssim_value, degradation = calculate_quality(original_np, attacked_np)
#
#     # Evaluate message extraction
#     bit_accuracy, true_positive_rate, false_positive_rate, auc, _ = evaluate_message(message, decoded_message_attacked)
#
#     # Display the images
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     axes[0].imshow(original_np)
#     axes[0].set_title("Original Image")
#     axes[0].axis("off")
#     axes[1].imshow(encoded_np)
#     axes[1].set_title("Watermarked Image")
#     axes[1].axis("off")
#     axes[2].imshow(attacked_np)
#     axes[2].set_title(f"Attacked Image ({args.attack_type})")
#     axes[2].axis("off")
#     plt.show()
#
#     # Print results
#     print(f"Attack Type: {args.attack_type}")
#     print(f"PSNR: {psnr_value:.3f}, SSIM: {ssim_value:.3f}, Degradation: {degradation:.2f}%")
#     print(
#         f"Bit Accuracy: {bit_accuracy:.3f}, TPR: {true_positive_rate:.3f}, FPR: {false_positive_rate:.3f}, AUC: {auc:.3f}")
#
#
# if __name__ == "__main__":
#     main()
#
#
# python adversarial_attacks.py --options-file "D:\Research\CVPR'24\Watermark\runs\celebhq-subset-no-noise\options-and-config.pickle" --checkpoint-file "D:\Research\CVPR'24\Watermark\runs\celebhq-subset-no-noise\checkpoints\celebhq-subset--epoch-96.pyt" --image-path "D:\Research\CVPR'24\data\celeba_hq\mixed\000004.jpg" --attack-type "Bright"
