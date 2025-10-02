# MVSEP-MDX23-Docker

A Docker-based implementation of the MVSEP-MDX23 music source separation model. This project containerizes the powerful MVSEP-MDX23 model that can separate vocals and instruments from mixed audio tracks, making it easy to deploy and run in any environment.

## Features

- **Docker Integration**:
  - Containerized environment for easy deployment
  - GPU support through NVIDIA Container Toolkit
  - Consistent runtime environment across platforms

- **Multiple Separation Modes**:
  - Vocals/Instrumental separation
  - 4-STEMS separation (vocals, drums, bass, other)

- **Advanced Model Ensemble**:
  - BSRoformer
  - Kim MelRoformer
  - InstVoc
  - VitLarge (optional)
  - InstHQ4 (optional)
  - VOCFT (optional)
  - Demucs models (for 4-STEMS mode)

- **Customizable Settings**:
  - Adjustable model weights for ensemble
  - Input gain control
  - Output format selection (PCM_16, FLOAT, FLAC)
  - Overlap control for different models
  - BigShifts feature for improved separation
  - Optional vocal filtering below 50Hz
  - Gain restoration option

  **Lyrics Transcription (NEW)**
  - High-accuracy Whisper large-v3 model
  - Automatic language detection and optional alignment
  - Save transcription results directly to S3
  - We currently support English, Spanish and French (en,es,fr)


## Requirements

- Docker
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit (for GPU support)
- input as audio with sample rate of 44100 and wav format



## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mvsep-mdx23-docker.git
cd mvsep-mdx23-docker
```

2. Build the Docker image:
```bash
python utils/download_models.py
docker build -t mvsep-mdx23 .
```

## Usage

### Docker Run

Basic usage:
```bash
docker run --gpus all -v /path/to/input:/input -v /path/to/output:/output mvsep-mdx23 \
    --input_audio "audio.wav" \
    --output_folder "/output"
```

Advanced usage with custom parameters:
```bash
docker run --gpus all -v /path/to/input:/input -v /path/to/output:/output mvsep-mdx23 \
    --input_audio "audio.wav" \
    --output_folder "/output" \
    --BSRoformer_model "ep_368_1296" \
    --weight_BSRoformer 9.18 \
    --weight_Kim_MelRoformer 10 \
    --weight_InstVoc 3.39 \
    --output_format "FLAC" \
    --BigShifts 3 \
    --input_gain 0
```
windows cmd:
docker run --gpus device=0 ^
  -e AWS_ACCESS_KEY_ID=ACCESS_KEY ^
  -e AWS_SECRET_ACCESS_KEY=ECRET_ACCESS_KEY ^
  -e AWS_DEFAULT_REGION=REGION ^
  -e WHISPER_MODEL=large-v3 ^
  -e WHISPER_MODEL_CACHE=./models ^
  -v /path/to/test_input.json:/test_input.json ^
  mvsep-mdx23-test

OR

docker run --gpus all ^
  -e AWS_ACCESS_KEY_ID=ACCESS_KEY ^
  -e AWS_SECRET_ACCESS_KEY=ECRET_ACCESS_KEY ^
  -e AWS_DEFAULT_REGION=REGION ^
  -e WHISPER_MODEL=large-v3 ^
  -e WHISPER_MODEL_CACHE=./models ^
  -v /path/to/test_input.json:/test_input.json ^
  mvsep-mdx23-test


example json is in
  input/test_input.json


### Parameters

- `--input_audio`: name of audio file (wav with sample rate of 44100)
- `--output_folder`: Path to output directory (inside container)
- `--output_format`: Output format (PCM_16, FLOAT, or FLAC)
- `--vocals_only`: Only separate vocals/instrumental (omit for 4-STEMS mode)
- `--input_gain`: Input volume gain (dB)
- `--restore_gain`: Restore original gain after separation
- `--filter_vocals`: Remove audio below 50Hz in vocals stem
- `--BigShifts`: Number of shifts for improved separation (1-41)

Model-specific parameters:
- `--BSRoformer_model`: Model version (ep_317_1297 or ep_368_1296)
- `--weight_BSRoformer`: Weight for BSRoformer model (0-10)
- `--weight_Kim_MelRoformer`: Weight for Kim MelRoformer model (0-10)
- `--weight_InstVoc`: Weight for InstVoc model (0-10)
- `--use_VitLarge`: Enable VitLarge model
- `--weight_VitLarge`: Weight for VitLarge model (0-10)
- `--use_InstHQ4`: Enable InstHQ4 model
- `--weight_InstHQ4`: Weight for InstHQ4 model (0-10)
- `--use_VOCFT`: Enable VOCFT model
- `--weight_VOCFT`: Weight for VOCFT model (0-10)

### Environment Variables

The following environment variables can be set when running the container:
- `AWS_ACCESS_KEY_ID`: Access key ID for AWS
- `AWS_SECRET_ACCESS_KEY`: Secret access key for AWS
- `AWS_REGION`: AWS region
