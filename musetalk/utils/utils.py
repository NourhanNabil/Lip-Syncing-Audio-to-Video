import os
import numpy as np
import torch

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print("please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"
    
from musetalk.whisper.audio2feature import Audio2Feature
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet,PositionalEncoding

def load_all_model():
    audio_processor = Audio2Feature(model_path="./musetalk/models/whisper/tiny.pt")
    vae = VAE(model_path = "./musetalk/models/sd-vae-ft-mse/")
    unet = UNet(unet_config="./musetalk/models/musetalk/musetalk.json",
                model_path ="./musetalk/models/musetalk/pytorch_model.bin")
    pe = PositionalEncoding(d_model=384)
    return audio_processor,vae,unet,pe



def datagen(whisper_chunks,
            vae_encode_latents,
            batch_size=8,
            delay_frame=0):
    whisper_batch, latent_batch = [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i+delay_frame)%len(vae_encode_latents)
        latent = vae_encode_latents[idx]
        whisper_batch.append(w)
        latent_batch.append(latent)

        if len(latent_batch) >= batch_size:
            whisper_batch = np.stack(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)
            yield whisper_batch, latent_batch
            whisper_batch, latent_batch = [], []

    # the last batch may smaller than batch size
    if len(latent_batch) > 0:
        whisper_batch = np.stack(whisper_batch)
        latent_batch = torch.cat(latent_batch, dim=0)

        yield whisper_batch, latent_batch
