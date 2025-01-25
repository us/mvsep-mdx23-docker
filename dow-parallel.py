import os
import torch
import logging
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict
import aiohttp
import asyncio
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure retry policy
RETRY_COUNT = 5
RETRY_DELAY = 2  # seconds
CHUNK_SIZE = 1024 * 1024  # 1MB chunks

# Add GitHub mirror to bypass rate limiting
GITHUB_MIRROR = "https://mirror.ghproxy.com/"

MODEL_URLS = {
    # BS-Roformer models
    "model_bs_roformer_ep_368_sdr_12.9628": {
        "ckpt": GITHUB_MIRROR + "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        "yaml": GITHUB_MIRROR + "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml"
    },
    # ... (keep other URLs same structure with GITHUB_MIRROR prefix)
}

async def download_file(session: aiohttp.ClientSession, url: str, dest_path: Path) -> None:
    """Asynchronous file download with progress tracking and retries"""
    for attempt in range(RETRY_COUNT):
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(dest_path, 'wb') as f, tqdm(
                    desc=dest_path.name,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress:
                    async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                        f.write(chunk)
                        progress.update(len(chunk))
                return
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Attempt {attempt + 1}/{RETRY_COUNT} failed: {str(e)}")
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            
    raise Exception(f"Failed to download {url} after {RETRY_COUNT} attempts")

async def download_model_files(model_dir: Path, urls: Dict[str, str]) -> None:
    """Download all files for a single model"""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as session:
        tasks = []
        for file_type, url in urls.items():
            # Get filename from URL
            filename = url.split('/')[-1].split('?')[0]
            dest_path = model_dir / filename
            
            if not dest_path.exists() or dest_path.stat().st_size == 0:
                tasks.append(download_file(session, url, dest_path))
        
        await asyncio.gather(*tasks)

def download_models(model_dir: str = "models") -> None:
    """Main download coordinator"""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create event loop for async downloads
    loop = asyncio.get_event_loop()
    
    # Create tasks for all models
    tasks = []
    for model_name, urls in MODEL_URLS.items():
        tasks.append(download_model_files(model_dir, urls))
    
    # Run all downloads in parallel
    loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()

if __name__ == "__main__":
    # Install required dependencies if missing
    try:
        import numpy
    except ImportError:
        logger.info("Installing numpy...")
        os.system("pip install numpy")
    
    try:
        import aiohttp
    except ImportError:
        logger.info("Installing aiohttp...")
        os.system("pip install aiohttp")
    
    download_models()
