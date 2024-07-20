import os
import requests
# Define the checkpoints directory
CheckpointsDir = "./musetalk/models"

# Function to download and verify files
def download_file(url, output):
    response = requests.get(url)
    with open(output, 'wb') as f:
        f.write(response.content)

    # Verify file size is greater than 0
    if os.path.getsize(output) == 0:
        raise Exception(f"Error: File {output} is empty or download failed.")

# Create the models directory if it does not exist
if os.path.isdir(CheckpointsDir):
    print("start downloading...")

    # Download MuseTalk weights
    musetalk_dir = os.path.join(CheckpointsDir, "musetalk")
    os.makedirs(musetalk_dir, exist_ok=True) 
    download_file("https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin", os.path.join(musetalk_dir, "pytorch_model.bin"))
    download_file("https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json", os.path.join(musetalk_dir, "musetalk.json"))

    # Download SD VAE weights
    sd_vae_dir = os.path.join(CheckpointsDir, "sd-vae-ft-mse")
    os.makedirs(sd_vae_dir, exist_ok=True) 
    download_file("https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json",os.path.join(sd_vae_dir,"config.json"))
    download_file("https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin",os.path.join(sd_vae_dir,"diffusion_pytorch_model.bin"))

    # Download DWPose weights
    dwpose_dir = os.path.join(CheckpointsDir, "dwpose")
    os.makedirs(dwpose_dir, exist_ok=True)
    download_file("https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth",os.path.join(dwpose_dir,"dw-ll_ucoco_384.pth"))
    
    # Download Whisper weights
    whisper_dir = os.path.join(CheckpointsDir, "whisper")
    os.makedirs(whisper_dir, exist_ok=True)
    download_file("https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt", os.path.join(whisper_dir, "tiny.pt"))

    # Download Face Parse Bisent weights
    face_parse_dir = os.path.join(CheckpointsDir, "face-parse-bisent")
    os.makedirs(face_parse_dir, exist_ok=True)
    download_file("https://drive.usercontent.google.com/u/0/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812&export=download",os.path.join(face_parse_dir, "79999_iter.pth"))

    # Download ResNet weights
    download_file("https://download.pytorch.org/models/resnet18-5c106cde.pth", os.path.join(face_parse_dir, "resnet18-5c106cde.pth"))
    
    print("All models downloaded")
else:
    print("Model already downloaded.")
