import runpod
import os
import json
import torch
import logging
import soundfile as sf
import numpy as np
import tempfile
import boto3
from pathlib import Path
from moviepy.editor import VideoFileClip
import static_ffmpeg
from pydub import AudioSegment
import librosa

from inference import EnsembleDemucsMDXMusicSeparationModel, predict_with_model

# Initialize ffmpeg paths early
static_ffmpeg.add_paths()

# Initialize logging
logging.basicConfig(level=logging.INFO)
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

def download_file_from_s3(client, s3_path):
    """Download a file from S3 to a temporary location"""
    bucket_name, key = s3_path.replace("s3://", "").split("/", 1)
    temp_dir = tempfile.mkdtemp()
    local_filename = os.path.join(temp_dir, os.path.basename(key))
    logger.info(f"Downloading file from S3: {s3_path} to {local_filename}")
    client.download_file(bucket_name, key, local_filename)
    return local_filename

def upload_file_to_s3(client, local_path, bucket_name, s3_path):
    """Upload a file to S3"""
    client.upload_file(local_path, bucket_name, s3_path)
    logger.info(f"Uploaded file to S3: {s3_path}")
    return f"s3://{bucket_name}/{s3_path}"

def process_audio_file(file_path):
    """Process different audio file formats and return numpy array"""
    try:
        if file_path.lower().endswith('.m4a'):
            logger.info('Processing M4A file using pydub.')
            output_path = os.path.join(tempfile.mkdtemp(), "temp_audio.wav")
            audio_segment = AudioSegment.from_file(file_path, format="m4a")
            audio_segment.export(output_path, format="wav", parameters=["-ac", "2", "-ar", "44100"])
            audio, sample_rate = librosa.load(output_path, sr=44100, mono=False)
            os.remove(output_path)
        elif file_path.lower().endswith('.mp4'):
            logger.info('Processing MP4 file using MoviePy.')
            output_path = os.path.join(tempfile.mkdtemp(), "temp_audio.wav")
            video = VideoFileClip(file_path, fps_source='tbr')
            video.audio.write_audiofile(output_path, fps=44100)
            video.close()
            audio, sample_rate = librosa.load(output_path, sr=44100, mono=False)
            os.remove(output_path)
        else:
            logger.info('Loading audio file directly with librosa.')
            audio, sample_rate = librosa.load(file_path, sr=44100, mono=False)

        # Ensure stereo format
        if len(audio.shape) == 1:
            logger.info('Converting mono to stereo.')
            audio = np.stack([audio, audio], axis=0)
        elif len(audio.shape) == 2 and audio.shape[0] > 2:
            logger.info('Transposing multi-channel audio.')
            audio = audio.T

        return audio, sample_rate

    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        raise

def handler(event):
    """
    RunPod handler function for audio separation.
    Expects input in the format:
    {
        "input": {
            "audio_url": "s3://bucket-name/path/to/input.mp3",
            "output_bucket": "output-bucket-name",
            "output_prefix": "path/to/output/folder",
            "output_format": "FLOAT",  # or "FLAC" for FLAC output
            "options": {
                "overlap_demucs": 0.1,
                "overlap_VOCFT": 0.1,
                "overlap_VitLarge": 1,
                "overlap_InstVoc": 1,
                "weight_InstVoc": 8,
                "weight_VOCFT": 1,
                "weight_VitLarge": 5,
                "large_gpu": True,
                "BigShifts": 7,
                "vocals_only": False,
                "use_VOCFT": False,
                "input_gain": 0,
                "restore_gain": False
            }
        }
    }
    """
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3',
                               aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                               aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                               region_name=os.getenv("AWS_REGION"))

        # Get input data
        input_data = event["input"]
        audio_url = input_data["audio_url"]
        job_id = event["id"]
        
        # Get S3 output settings
        output_bucket = input_data.get("output_bucket")
        output_prefix = input_data.get("output_prefix", "")
        output_format = input_data.get("output_format", "FLOAT")
        
        # Get processing options
        options = input_data.get("options", {})
        
        # Create temporary output directory
        temp_output_dir = tempfile.mkdtemp()
        
        # Set required options for the model
        options.update({
            "input_audio": [audio_url],  # Will be replaced with local path
            "output_folder": temp_output_dir,
            "output_format": output_format
        })
        
        # Download and process audio file
        local_audio_path = download_file_from_s3(s3_client, audio_url)
        try:
            # Process audio file
            audio, sample_rate = process_audio_file(local_audio_path)
            
            # Update options with local path
            options["input_audio"] = [local_audio_path]
            
            # Initialize model and run prediction
            model = init_model(options)
            result = predict_with_model(options)
            
            # Upload results
            output_urls = {"s3": {}}
            
            # Get list of output files
            output_files = [f for f in os.listdir(temp_output_dir) 
                          if os.path.isfile(os.path.join(temp_output_dir, f))]
            
            # Upload each output file
            for output_file in output_files:
                local_path = os.path.join(temp_output_dir, output_file)
                s3_key = f"{output_prefix}/{job_id}/{output_file}" if output_prefix else f"{job_id}/{output_file}"
                
                if output_bucket:
                    output_urls["s3"][output_file] = upload_file_to_s3(
                        s3_client, local_path, output_bucket, s3_key
                    )
                
                # Clean up local file
                os.unlink(local_path)
            
            # Clean up temporary directories
            os.rmdir(temp_output_dir)
            if os.path.exists(local_audio_path):
                os.unlink(local_audio_path)
                os.rmdir(os.path.dirname(local_audio_path))
            
            return {
                "output": output_urls,
                "sample_rate": sample_rate,
            }
            
        except Exception as e:
            # Clean up in case of error
            if os.path.exists(local_audio_path):
                os.unlink(local_audio_path)
                os.rmdir(os.path.dirname(local_audio_path))
            raise
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler}) 