import os
import argparse
import subprocess
import numpy as np

def run_single_inference(single_infer_script, options_file, checkpoint_file, image_path):
    """Runs single_image_infer.py for one image and retrieves the metrics as output."""
    command = [
        "python", single_infer_script,
        "--options-file", options_file,
        "--checkpoint-file", checkpoint_file,
        "--source-image", image_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout
    print(output)  # Print output for debugging purposes

    # Initialize default values for metrics
    metrics = {'PSNR': 0, 'SSIM': 0, 'MSE': 0, 'LPIPS': 0, 'MS-SSIM': 0}

    for line in output.splitlines():
        if "PSNR:" in line:
            metrics['PSNR'] = float(line.split(":")[1].strip())
        elif "SSIM:" in line:
            metrics['SSIM'] = float(line.split(":")[1].strip())
        elif "MSE:" in line:
            metrics['MSE'] = float(line.split(":")[1].strip())
        elif "LPIPS:" in line:
            metrics['LPIPS'] = float(line.split(":")[1].strip())
        elif "MS-SSIM:" in line:
            metrics['MS-SSIM'] = float(line.split(":")[1].strip())
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Perform multi-image inference using single_image_infer.py')
    parser.add_argument('--single-infer-script', type=str, required=True, help='Path to single_image_infer.py')
    parser.add_argument('--options-file', type=str, required=True, help='Path to options file')
    parser.add_argument('--checkpoint-file', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory containing images for inference')
    args = parser.parse_args()

    psnr_values, ssim_values, mse_values, lpips_values, ms_ssim_values = [], [], [], [], []
    image_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.jpg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(args.image_dir, image_file)
        metrics = run_single_inference(args.single_infer_script, args.options_file, args.checkpoint_file, image_path)
        psnr_values.append(metrics['PSNR'])
        ssim_values.append(metrics['SSIM'])
        mse_values.append(metrics['MSE'])
        lpips_values.append(metrics['LPIPS'])
        ms_ssim_values.append(metrics['MS-SSIM'])

    # Calculate and print the averages for all metrics
    print(f'Average PSNR: {np.mean(psnr_values):.3f}')
    print(f'Average SSIM: {np.mean(ssim_values):.3f}')
    print(f'Average MSE: {np.mean(mse_values):.3f}')
    print(f'Average LPIPS: {np.mean(lpips_values):.3f}')
    print(f'Average MS-SSIM: {np.mean(ms_ssim_values):.3f}')


if __name__ == '__main__':
    main()
