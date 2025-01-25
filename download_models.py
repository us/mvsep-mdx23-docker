import os
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_URLS = {
    # BS-Roformer models
    "model_bs_roformer_ep_368_sdr_12.9628": {
        "ckpt": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        "yaml": "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml"
    },
    "model_bs_roformer_ep_317_sdr_12.9755": {
        "ckpt": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "yaml": "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_317_sdr_12.9755.yaml"
    },
    
    # Kim MelBand Roformer
    "Kim_MelRoformer": {
        "ckpt": "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt",
        "yaml": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml"
    },
    
    # InstVoc model
    "MDX23C-8KFFT-InstVoc_HQ": {
        "ckpt": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/MDX23C-8KFFT-InstVoc_HQ.ckpt",
        "yaml": "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_2_stem_full_band_8k.yaml"
    },
    
    # VitLarge model
    "model_vocals_segm_models_sdr_9.77": {
        "ckpt": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt",
        "yaml": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_segm_models.yaml"
    },
    
    # ONNX models
    "UVR-MDX-NET-Voc_FT": {
        "onnx": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx"
    },
    "UVR-MDX-NET-Inst_HQ_4": {
        "onnx": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_4.onnx"
    }
}

def download_file(url: str, dest_path: Path) -> None:
    """Download a file from URL to destination path"""
    try:
        logger.info(f"Downloading {url} to {dest_path}")
        torch.hub.download_url_to_file(url, str(dest_path))
        logger.info(f"Successfully downloaded {dest_path.name}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        raise

def download_models(model_dir: str = "models") -> None:
    """Download all model files to the specified directory"""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, urls in MODEL_URLS.items():
        # Download checkpoint files
        if "ckpt" in urls:
            ckpt_path = model_dir / f"{model_name}.ckpt"
            if not ckpt_path.exists():
                download_file(urls["ckpt"], ckpt_path)
                
        # Download YAML config files
        if "yaml" in urls:
            yaml_path = model_dir / f"{model_name}.yaml"
            if not yaml_path.exists():
                download_file(urls["yaml"], yaml_path)
                
        # Download ONNX models
        if "onnx" in urls:
            onnx_path = model_dir / f"{model_name}.onnx"
            if not onnx_path.exists():
                download_file(urls["onnx"], onnx_path)

if __name__ == "__main__":
    download_models() 