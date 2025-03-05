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
import firebase_admin
from firebase_admin import credentials, messaging

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
# Firebase app instance
FIREBASE_APP = None

def download_from_s3(s3_client, s3_url, local_path):
    """Download a file from S3 to a local path"""
    logger.info(f"Downloading from S3: {s3_url} to {local_path}")
    try:
        bucket_name, key = s3_url.replace("s3://", "").split("/", 1)
        s3_client.download_file(bucket_name, key, local_path)
        return True
    except Exception as e:
        logger.error(f"Error downloading from S3: {str(e)}")
        return False

def init_firebase(s3_client=None, credentials_s3_url=None, workspace_dir=None):
    """Initialize Firebase app with service account credentials from S3 or local path"""
    global FIREBASE_APP
    
    if FIREBASE_APP is not None:
        # Already initialized
        return True
        
    try:
        # Determine credentials path
        cred_path = None
        
        # If S3 URL is provided, download the credentials
        if credentials_s3_url and s3_client and workspace_dir:
            local_cred_path = os.path.join(workspace_dir, "firebase_credentials.json")
            if download_from_s3(s3_client, credentials_s3_url, local_cred_path):
                cred_path = local_cred_path
                logger.info(f"Downloaded Firebase credentials from {credentials_s3_url}")
        

        # Initialize Firebase with credentials
        cred = credentials.Certificate(cred_path)
        FIREBASE_APP = firebase_admin.initialize_app(cred)
        logger.info(f"Firebase initialized successfully with credentials from {cred_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {str(e)}")
        return False

def send_notification(token, title, body, data=None):
    """Send a Firebase Cloud Messaging notification"""
    if not FIREBASE_APP:
        logger.warning("Firebase not initialized, skipping notification")
        return False
    
    try:
        if not token:
            logger.warning("No FCM token provided, skipping notification")
            return False
            
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body
            ),
            data=data or {},
            token=token
        )
        
        response = messaging.send(message)
        logger.info(f"Successfully sent notification: {response}")
        return True
    except Exception as e:
        logger.error(f"Error sending notification: {str(e)}")
        return False

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
    
    # Extract notification settings from input
    input_data = event.get("input", {})
    notification_config = input_data.get("notification", {})
    fcm_token = notification_config.get("fcm_token")
    enable_notifications = bool(fcm_token) and notification_config.get("enabled", False)
    firebase_creds_url = notification_config.get("credentials_url")
    
    # Initialize S3 client (moved up to use for Firebase credentials download)
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )
    
    # Initialize Firebase if notifications are enabled
    if enable_notifications:
        init_firebase(s3_client, firebase_creds_url, workspace_dir)
    
    # Send notification that processing has started
    if enable_notifications:
        job_id = event.get("id", "unknown")
        send_notification(
            fcm_token,
            "Processing Started",
            f"Audio separation job {job_id} has started processing",
            {"job_id": job_id, "status": "started"}
        )
    
    try:
        # Extract input parameters
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
        
        # Send notification that file download completed
        if enable_notifications:
            send_notification(
                fcm_token,
                "File Downloaded",
                f"Audio file for job {job_id} has been downloaded and is being prepared",
                {"job_id": job_id, "status": "downloaded"}
            )

        # Process audio file
        processed_path = process_audio_file(input_path, workspace_dir)
        
        # Get sample rate for output
        audio_info = sf.info(processed_path)
        sample_rate = audio_info.samplerate
        
        # Send notification that processing is starting
        if enable_notifications:
            send_notification(
                fcm_token,
                "Model Processing",
                f"Audio separation for job {job_id} is now being processed by the model",
                {"job_id": job_id, "status": "processing"}
            )

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
        
        # Send completion notification with output details
        if enable_notifications:
            file_count = len(output_files)
            send_notification(
                fcm_token,
                "Processing Complete",
                f"Audio separation job {job_id} completed successfully with {file_count} output files",
                {
                    "job_id": job_id, 
                    "status": "completed",
                    "file_count": str(file_count),
                    "sample_rate": str(sample_rate)
                }
            )

        return {
            "output": output_urls,
            "sample_rate": sample_rate,
        }

    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        exec_error = traceback.format_exc()
        print(exec_error)
        
        # Send error notification
        if enable_notifications:
            send_notification(
                fcm_token,
                "Processing Error",
                f"Error processing job {event.get('id', 'unknown')}: {str(e)}",
                {"job_id": event.get("id", "unknown"), "status": "error", "error": str(e)}
            )
            
        return {"error": str(e)}

    finally:
        # Clean up workspace in all cases
        cleanup_workspace(workspace_dir)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler}) 
    