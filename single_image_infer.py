import torch
import torch.nn
import argparse
import os
import cv2
import numpy as np
from options import HiDDenConfiguration
import utils
from model.hidden import *
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision.transforms.functional as TF
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from pytorch_msssim import ms_ssim
import lpips
from torch_fidelity import calculate_metrics


def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


def load_and_preprocess_image(path, height, width, device):
    image_pil = Image.open(path)
    image = randomCrop(np.array(image_pil), height, width)
    image_tensor = TF.to_tensor(image).to(device)
    image_tensor = image_tensor * 2 - 1
    image_tensor.unsqueeze_(0)
    return image_tensor, np.array(image_pil)


def calculate_metrics(original_np, encoded_np, lpips_fn, original_tensor, encoded_tensor):
    psnr_value = psnr(original_np, encoded_np, data_range=255)

    # Set win_size explicitly for SSIM to avoid the issue
    ssim_value = ssim(original_np, encoded_np, data_range=255, channel_axis=-1, win_size=7)  # Adjust win_size as needed
    mse_value = np.mean((original_np - encoded_np) ** 2)
    lpips_value = lpips_fn(original_tensor, encoded_tensor).item()
    ms_ssim_value = ms_ssim(original_tensor, encoded_tensor, data_range=1.0).item()

    return psnr_value, ssim_value, mse_value, lpips_value, ms_ssim_value


# def calculate_metrics(original_np, encoded_np, lpips_fn, original_tensor, encoded_tensor):
#     psnr_value = psnr(original_np, encoded_np, data_range=255)
#     ssim_value = ssim(original_np, encoded_np, data_range=255, multichannel=True)
#     mse_value = np.mean((original_np - encoded_np) ** 2)
#     lpips_value = lpips_fn(original_tensor, encoded_tensor).item()
#     ms_ssim_value = ms_ssim(original_tensor, encoded_tensor, data_range=1.0).item()
#     return psnr_value, ssim_value, mse_value, lpips_value, ms_ssim_value


def process_image(hidden_net, image_tensor, message, lpips_fn):
    losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch([image_tensor, message])
    original_np = ((image_tensor.squeeze().cpu().numpy() + 1) / 2 * 255).transpose(1, 2, 0).astype(np.uint8)
    encoded_np = ((encoded_images.squeeze().cpu().numpy() + 1) / 2 * 255).transpose(1, 2, 0).astype(np.uint8)
    psnr_value, ssim_value, mse_value, lpips_value, ms_ssim_value = calculate_metrics(
        original_np, encoded_np, lpips_fn, image_tensor, encoded_images
    )
    return psnr_value, ssim_value, mse_value, lpips_value, ms_ssim_value, encoded_images, decoded_messages, message


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str, help='Options file path')
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str, help='Model checkpoint file')
    parser.add_argument('--source-image', '-s', required=True, type=str, help='Single image or directory of images')
    args = parser.parse_args()

    train_options, hidden_config, noise_config = utils.load_options(args.options_file)
    noiser = Noiser(noise_config, device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    checkpoint = torch.load(args.checkpoint_file)
    hidden_net = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)

    if os.path.isdir(args.source_image):
        psnr_values, ssim_values, mse_values, lpips_values, ms_ssim_values = [], [], [], [], []
        image_files = [os.path.join(args.source_image, f) for f in os.listdir(args.source_image) if f.endswith(('.jpg', '.png'))]
        for img_file in image_files:
            image_tensor, original_np = load_and_preprocess_image(img_file, hidden_config.H, hidden_config.W, device)
            message = torch.Tensor(np.random.choice([0, 1], (image_tensor.shape[0], 64))).to(device)
            psnr_value, ssim_value, mse_value, lpips_value, ms_ssim_value, encoded_images, decoded_messages, message = process_image(
                hidden_net, image_tensor, message, lpips_fn)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            mse_values.append(mse_value)
            lpips_values.append(lpips_value)
            ms_ssim_values.append(ms_ssim_value)
        print(f'Average PSNR: {np.mean(psnr_values):.3f}')
        print(f'Average SSIM: {np.mean(ssim_values):.3f}')
        print(f'Average MSE: {np.mean(mse_values):.3f}')
        print(f'Average LPIPS: {np.mean(lpips_values):.3f}')
        print(f'Average MS-SSIM: {np.mean(ms_ssim_values):.3f}')

    else:
        image_tensor, original_np = load_and_preprocess_image(args.source_image, hidden_config.H, hidden_config.W, device)
        message = torch.Tensor(np.random.choice([0, 1], (image_tensor.shape[0], 64))).to(device)
        psnr_value, ssim_value, mse_value, lpips_value, ms_ssim_value, encoded_images, decoded_messages, message = process_image(
            hidden_net, image_tensor, message, lpips_fn)
        print(f'PSNR: {psnr_value:.3f}')
        print(f'SSIM: {ssim_value:.3f}')
        # print(f'MSE: {mse_value:.3f}')
        # print(f'LPIPS: {lpips_value:.3f}')
        print(f'MS-SSIM: {ms_ssim_value:.3f}')
        print(f'LPIPS: {lpips_value:.3f}')
        print(f'MSE: {mse_value:.3f}')
        utils.save_images(image_tensor.cpu(), encoded_images.cpu(), 'test', '.', resize_to=(512, 512))


if __name__ == '__main__':
    main()


#Single Image
# python single_image_infer.py --options-file "D:\Research\CVPR'24\Watermark\runs\celebhq-subset-no-noise\options-and-config.pickle" --checkpoint-file "D:\Research\CVPR'24\Watermark\runs\celebhq-subset-no-noise\checkpoints\celebhq-subset--epoch-96.pyt" --source-image "D:\Research\CVPR'24\data\coco2017\mixed\000000000001.jpg"
