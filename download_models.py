import os
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict
import aiohttp
import asyncio
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
RETRY_COUNT = 5
RETRY_DELAY = 2  # seconds
CHUNK_SIZE = 1024 * 1024  # 1MB chunks
GITHUB_MIRROR = "https://mirror.ghproxy.com/"  # GitHub mirror for faster downloads

MODEL_URLS = {
    # BS-Roformer models
    "model_bs_roformer_ep_368_sdr_12.9628": {
        "ckpt": GITHUB_MIRROR + "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        "yaml": "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml"
    },
    "model_bs_roformer_ep_317_sdr_12.9755": {
        "ckpt": GITHUB_MIRROR + "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "yaml": "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_317_sdr_12.9755.yaml"
    },
    
    # Kim MelBand Roformer (keep HuggingFace URL direct)
    "Kim_MelRoformer": {
        "ckpt": "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt",
        "yaml": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml"
    },
    
    # InstVoc model
    "MDX23C-8KFFT-InstVoc_HQ": {
        "ckpt": GITHUB_MIRROR + "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/MDX23C-8KFFT-InstVoc_HQ.ckpt",
        "yaml": "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_2_stem_full_band_8k.yaml"
    },
    
    # VitLarge model
    "model_vocals_segm_models_sdr_9.77": {
        "ckpt": GITHUB_MIRROR + "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt",
        "yaml": GITHUB_MIRROR + "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_segm_models.yaml"
    },
    
    # ONNX models
    "UVR-MDX-NET-Voc_FT": {
        "onnx": GITHUB_MIRROR + "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx"
    },
    "UVR-MDX-NET-Inst_HQ_4": {
        "onnx": GITHUB_MIRROR + "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_4.onnx"
    }
}

async def download_file(session: aiohttp.ClientSession, url: str, dest_path: Path) -> None:
    """Asynchronous file download with progress tracking and retry mechanism"""
    for attempt in range(RETRY_COUNT):
        try:
            async with session.get(url, allow_redirects=True) as response:
                if response.status == 404:
                    logger.error(f"File not found: {url}")
                    raise Exception(f"404 Not Found: {url}")
                    
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                # Create parent directories if they don't exist
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download with progress bar
                with open(dest_path, 'wb') as f, tqdm(
                    desc=dest_path.name,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                    leave=False
                ) as progress:
                    async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                        f.write(chunk)
                        progress.update(len(chunk))
                
                # Remove file size check to accept any file size once downloaded.
                return
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Attempt {attempt+1}/{RETRY_COUNT} failed for {url}: {str(e)}")
            if dest_path.exists():
                dest_path.unlink()
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
    
    raise Exception(f"Failed to download {url} after {RETRY_COUNT} attempts")

async def download_all_models(model_dir: str = "models"):
    """Download all models to the specified directory"""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as session:
        tasks = []
        for model_name, urls in MODEL_URLS.items():
            for file_type, url in urls.items():
                # Get original filename from URL
                filename = url.split('/')[-1].split('?')[0]
                dest_path = model_dir / filename
                
                if not dest_path.exists():
                    logger.info(f"Downloading {model_name} {file_type}...")
                    task = download_file(session, url, dest_path)
                    tasks.append(task)
                else:
                    logger.info(f"Skipping {model_name} {file_type} (already exists)")
        
        if tasks:
            await asyncio.gather(*tasks)
            logger.info("All downloads completed successfully!")
        else:
            logger.info("All models already downloaded!")

def install_dependencies():
    """Install required packages automatically"""
    required = {'numpy', 'aiohttp', 'tqdm'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed

    if missing:
        logger.info(f"Installing missing dependencies: {', '.join(missing)}")
        os.system(f"pip install {' '.join(missing)}")

if __name__ == "__main__":
    # Install dependencies first
    import pkg_resources
    install_dependencies()
    
    # Run main download process
    asyncio.run(download_all_models()) 