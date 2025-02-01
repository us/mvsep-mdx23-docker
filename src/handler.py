import runpod
import os
import json
import traceback
import torch
import logging
import soundfile as sf
import numpy as np
import tempfile
import boto3
from pathlib import Path
from moviepy import VideoFileClip
import static_ffmpeg
from pydub import AudioSegment
import librosa
import shutil

from inference import EnsembleDemucsMDXMusicSeparationModel, predict_with_model

# Initialize ffmpeg paths early
static_ffmpeg.add_paths()

# Initialize logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance
MODEL = None

def init_model(options):
    """Initialize the model with given options"""
    global MODEL
    if MODEL is None:
        logger.info("Initializing model...")
        MODEL = EnsembleDemucsMDXMusicSeparationModel(options)
    return MODEL

def create_workspace():
    """Create a single workspace directory for all temporary files"""
    return tempfile.mkdtemp(prefix="audio_processing_")

def cleanup_workspace(workspace_dir):
    """Clean up the workspace directory and all its contents"""
    if os.path.exists(workspace_dir):
        try:
            shutil.rmtree(workspace_dir)
            logger.info(f"Cleaned up workspace: {workspace_dir}")
        except Exception as e:
            logger.error(f"Error cleaning workspace {workspace_dir}: {str(e)}")

def process_audio_file(input_path, workspace_dir):
    """Process audio file and return path to processed WAV file"""
    output_path = os.path.join(workspace_dir, "processed_audio.wav")
    logger.info(f"Processing audio file: {input_path}")
    
    try:
        if input_path.lower().endswith('.m4a'):
            logger.info('Converting M4A to WAV')
            audio = AudioSegment.from_file(input_path, format="m4a")
            audio.export(output_path, format="wav", parameters=["-ac", "2", "-ar", "44100"])
            
        elif input_path.lower().endswith('.mp4'):
            logger.info('Extracting audio from MP4')
            video = VideoFileClip(input_path, fps_source='tbr')
            video.audio.write_audiofile(output_path, fps=44100)
            video.close()
            
        else:
            logger.info('Loading and converting audio file')
            audio, sr = librosa.load(input_path, sr=44100, mono=False)
            if len(audio.shape) == 1:
                audio = np.stack([audio, audio])
            sf.write(output_path, audio.T, sr)

        logger.info(f"Audio processing complete: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise

def handler(event):
    """RunPod handler function for audio separation"""
    workspace_dir = create_workspace()
    logger.info(f"Created workspace directory: {workspace_dir}")
    
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION")
        )

        # Extract input parameters
        input_data = event["input"]
        audio_url = input_data["audio_url"]
        job_id = event["id"]
        output_bucket = input_data.get("output_bucket")
        output_prefix = input_data.get("output_prefix", "")
        output_format = input_data.get("output_format", "FLOAT")
        options = input_data.get("options", {})

        # Download input file
        bucket_name, key = audio_url.replace("s3://", "").split("/", 1)
        input_path = os.path.join(workspace_dir, "input" + os.path.splitext(key)[1])
        logger.info(f"Downloading from S3: {audio_url}")
        s3_client.download_file(bucket_name, key, input_path)

        # Process audio file
        processed_path = process_audio_file(input_path, workspace_dir)
        
        # Get sample rate for output
        audio_info = sf.info(processed_path)
        sample_rate = audio_info.samplerate

        # Set up model options
        options.update({
            "input_audio": [processed_path],
            "output_folder": workspace_dir,
            "output_format": output_format
        })

        # Run model inference
        logger.info("Initializing model and running inference")
        model = init_model(options)
        result = predict_with_model(options)

        # Upload results
        output_urls = {"s3": {}}
        output_files = [f for f in os.listdir(workspace_dir)
                       if os.path.isfile(os.path.join(workspace_dir, f))
                       and f != os.path.basename(processed_path)
                       and f != os.path.basename(input_path)]

        for output_file in output_files:
            local_path = os.path.join(workspace_dir, output_file)
            s3_key = f"{output_prefix}/{job_id}/{output_file}" if output_prefix else f"{job_id}/{output_file}"
            
            if output_bucket:
                logger.info(f"Uploading result: {output_file}")
                s3_client.upload_file(local_path, output_bucket, s3_key)
                output_urls["s3"][output_file] = f"s3://{output_bucket}/{s3_key}"

        return {
            "output": output_urls,
            "sample_rate": sample_rate,
        }

    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        exec_error = traceback.format_exc()
        print(exec_error)
        return {"error": str(e)}

    finally:
        # Clean up workspace in all cases
        cleanup_workspace(workspace_dir)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler}) 
    