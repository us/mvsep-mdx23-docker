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

from typing import Optional
import subprocess
import whisperx
from chords_tempo import analyze



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

# whisperx Configuration
MODEL_CACHE_DIR = os.getenv("WHISPER_MODEL_CACHE", "./models")
COMPUTE_TYPE = "float16"  # Changed to float16 for better cuda compatibility
BATCH_SIZE = 16
# Model cache for performance optimization
model_cache = {}


# ------------------- UTILITIES ------------------- #
def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "cached": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
            "max_cached": torch.cuda.max_memory_reserved() / 1024**3
        }
    return {}

def list_files_with_size(directory: str):
    """List files in a directory with size in MB"""
    files_info = []
    try:
        for root, _, files in os.walk(directory):
            for f in files:
                fpath = os.path.join(root, f)
                try:
                    size_mb = os.path.getsize(fpath) / (1024 * 1024)
                    files_info.append({
                        "path": fpath,
                        "size_mb": round(size_mb, 2)
                    })
                except Exception:
                    pass
    except Exception as e:
        files_info.append({"error": str(e)})
    return files_info



def get_system_usage():
    """Return disk, memory, and file listings"""
    usage = {}
    try:
        # Disk usage
        disk = subprocess.check_output(["df", "-h", "/"]).decode("utf-8").split("\n")[1].split()
        usage["disk_total"] = disk[1]
        usage["disk_used"] = disk[2]
        usage["disk_available"] = disk[3]
        usage["disk_percent"] = disk[4]

        # Memory usage
        mem_output = subprocess.check_output(["free", "-h"]).decode("utf-8").split("\n")
        if len(mem_output) > 1:
            mem_parts = mem_output[1].split()
            usage["mem_total"] = mem_parts[1]
            usage["mem_used"] = mem_parts[2]
            usage["mem_free"] = mem_parts[3]
            usage["mem_shared"] = mem_parts[4]
            usage["mem_cache"] = mem_parts[5]
            usage["mem_available"] = mem_parts[6]

    except Exception as e:
        usage["error"] = str(e)

    # Add GPU memory info
    usage["gpu_memory"] = get_gpu_memory_usage()
    return usage


# ------------------- MAIN HANDLER ------------------- #


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



# ### NEW: Boost vocal volume and ensure 16 kHz mono again
def ensure_model_cache_dir():
    """Ensure model cache directory exists and is accessible"""
    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        # Test if directory is writable
        test_file = os.path.join(MODEL_CACHE_DIR, "test.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception as e:
        logger.error(f"Model cache directory error: {str(e)}")
        return False

def boost_and_resample(input_path: str, gain_db: float,workspace_dir:str) -> str:
    """
    Boosts the audio (dB) and ensures 16k mono PCM16.
    """

    boosted_path= os.path.join(workspace_dir, "temp_vocals_boosted.wav")

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", input_path,
            "-vn",
            "-filter:a", f"volume={gain_db}dB",
            "-ac", "1",
            "-ar", "16000",
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            boosted_path
        ], check=True)
        return boosted_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg boost failed: {str(e)}")
        raise RuntimeError(f"FFmpeg boost failed: {str(e)}")

def load_model(model_size: str, language: Optional[str]):
    """Load Whisper model with GPU optimization"""
    cache_key = f"{model_size}_{language if language else 'no_lang'}"

    if cache_key in model_cache:
        logger.info(f"Using cached model: {cache_key}")
        return model_cache[cache_key]

    try:
        if not ensure_model_cache_dir():
            logger.error(f"Model cache directory is not accessible")
            raise RuntimeError("Model cache directory is not accessible")


        model = whisperx.load_model(
            model_size,
            device="cuda",
            compute_type=COMPUTE_TYPE,
            download_root=MODEL_CACHE_DIR,
            language=language if language and language != "-" else None
        )

        model_cache[cache_key] = model
        return model

    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

def load_alignment_model(language_code: str):
    """Load alignment model with fallback options"""
    try:
        # Try to load the default model first
        return whisperx.load_align_model(language_code=language_code, device="cuda")
    except Exception as e:
        logger.warning(f"Failed to load default alignment model for {language_code}, trying fallback: {str(e)}")

        # Define fallback models for specific languages
        fallback_models = {
            "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",  # Hindi
            "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese", # Portuguese
            "he": "imvladikon/wav2vec2-xls-r-300m-hebrew", # Hebrew
        }

        if language_code in fallback_models:
            try:
                # Try to load the fallback model
                return whisperx.load_align_model(
                    model_name=fallback_models[language_code],
                    device="cuda"
                )
            except Exception as fallback_e:
                logger.error(f"Failed to load fallback alignment model for {language_code}: {str(fallback_e)}")
                raise RuntimeError(f"Alignment model loading failed for {language_code}")
        else:
            logger.error(f"No alignment model available for language: {language_code}")
            raise RuntimeError(f"No alignment model available for language: {language_code}")

def transcribe_audio(audio_path: str, model_size: str, language: Optional[str], align: bool):
    """Core transcription logic with optional translation"""
    try:
        model = load_model(model_size, language)
        result = model.transcribe(audio_path, batch_size=BATCH_SIZE)
        detected_language = result.get("language", language if language else "en")
        # assuming detected_language is a language code like "en", "es", or "fr"
        allowed_langs = {"en", "es", "fr"}   # English, Spanish, French


        if detected_language in allowed_langs:
            # ✅ language is supported – run transcription
            if align and detected_language != "unknown":
                try:
                    align_model, metadata = load_alignment_model(detected_language)
                    result = whisperx.align(
                        result["segments"],
                        align_model,
                        metadata,
                        audio_path,
                        device="cuda",
                        return_char_alignments=False
                    )
                except Exception as e:
                    logger.error(f"Alignment skipped: {str(e)}")
                    # Continue without alignment if it fails
                    return {
                        "error": f"alignment failed: {str(e)}",
                        "status": False
                    }

            if not result.get("segments") or len(result["segments"]) == 0:
                return {
                    "error": "segments not found",
                    "status": False
                }

            # NEW: Fix missing word timestamps after alignment
            # result = fix_missing_word_timestamps(result)

            return {
                "text": " ".join(seg["text"] for seg in result["segments"]),
                "segments": result["segments"],
                "language": detected_language,
                "model": model_size,
                "status": True,
                "alignment_success": "alignment_error" not in result
            }

        else:
            # ❌ not supported – return or raise an error
            return {
                "error": f"Unsupported language: {detected_language}",
                "status": False
            }

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
       # raise RuntimeError(f"Transcription failed: {str(e)}")
        return {
            "error": f"Transcription failed: {str(e)}",
            "status": False
        }

def save_response_to_s3(s3_client,output_dir,output_bucket,job_id, response_data, status="success"):
    """
    Save response to S3 bucket in the appropriate directory structure

    Args:
        job_id: The ID of the job
        response_data: The response data to save
        status: Status of the job (success, error, failed)
    """
    if not output_bucket:
        logger.warning("S3_BUCKET not configured, skipping response save")
        return False

    try:
        # Create the directory path
        directory_path = f"{output_dir}/{job_id}/lyrics/"
        file_key = f"{directory_path}response.json"

        # Convert response to JSON string
        response_json = json.dumps(response_data, indent=2, ensure_ascii=False)

        # Upload to S3
        s3_client.put_object(
            Bucket=output_bucket,
            Key=file_key,
            Body=response_json,
            ContentType='application/json'
        )

        logger.info(f"Response saved to S3: s3://{output_bucket}/{file_key}")
        return True

    except Exception as e:
        logger.error(f"Failed to save response to S3: {str(e)}")
        return False

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
    vocal_gain_db = input_data.get("vocal_gain_db")

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

    try:
        # Extract input parameters
        audio_url = input_data["audio_url"]
        job_id = event["id"]
        output_bucket = input_data.get("output_bucket")
        output_prefix = input_data.get("output_prefix", "")
        output_format = input_data.get("output_format", "FLOAT")
        options = input_data.get("options", {})

        # Extract song name from the audio URL
        song_name = os.path.basename(audio_url).split('.')[0]

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



        # generate lyrics with whisperx

        # Find matching files
        matches = [
            f for f in os.listdir(workspace_dir)
            if 'processed_audio_vocals' in f.lower()
            and f.lower().endswith(('.wav', '.flac'))
        ]

        if not matches:
            raise ValueError("Vocals file not found in workspace directory")

        vocals_path = os.path.join(workspace_dir, matches[0])
        logger.info(f"Found vocals file: {vocals_path}")


        if not os.path.exists(vocals_path):
            raise FileNotFoundError(f"Expected file does not exist: {vocals_path}")

        logger.info(f"Using vocals file for transcription: {vocals_path}")

        isVocalBoost = False
        try:
            processed_audio_path = boost_and_resample(vocals_path, vocal_gain_db, workspace_dir)
            isVocalBoost= True
        except Exception as e:
            # prepare_audio already logs and falls back; this is just extra safety
            logger.warning(f"prepare_audio_for_transcription error: {str(e)}")
            isVocalBoost = False
            processed_audio_path = vocals_path

        transcribeResult = transcribe_audio(
            processed_audio_path,
            input_data.get("model_size", "large-v3"),
            input_data.get("language", None),
            input_data.get("align", False)
        )

        # --- delete the boosted temp file after use ---
        try:
            if os.path.exists(processed_audio_path):
                if isVocalBoost:
                    os.remove(processed_audio_path)
                    logger.info(f"Deleted temporary processed file: {processed_audio_path}")
        except Exception as cleanup_err:
            logger.warning(f"Could not delete {processed_audio_path}: {cleanup_err}")

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


        # save result  in s3
        transcribeResult["system_usage"] = get_system_usage()
        transcrib_upload = save_response_to_s3(s3_client,output_prefix,output_bucket,job_id, transcribeResult, "success")
        logger.info(f"transcrib file upload processed done: {transcrib_upload}")

        # Chords, tempo, and key analysis
        analysis_result = None
        if input_data.get("enable_analysis", False):
            try:
                logger.info("Running chord/key/tempo analysis")

                if os.path.exists(vocals_path):
                    analysis_result = analyze(vocals_path, rounding=2)
                    logger.info(f"Analysis complete: Key={analysis_result['key']}, Tempo={analysis_result['tempo']}, Chords={len(analysis_result['chords'])}")

                    analysis_filename = os.path.splitext(os.path.basename(processed_path))[0] + '_analysis.json'
                    analysis_path = os.path.join(workspace_dir, analysis_filename)
                    with open(analysis_path, 'w') as f:
                        json.dump(analysis_result, f, indent=2)
                    logger.info(f"Analysis saved to: {analysis_filename}")

                    # Upload analysis to S3
                    if output_bucket:
                        s3_key = f"{output_prefix}/{job_id}/{analysis_filename}" if output_prefix else f"{job_id}/{analysis_filename}"
                        logger.info(f"Uploading analysis: {analysis_filename}")
                        s3_client.upload_file(analysis_path, output_bucket, s3_key)
                        output_urls["s3"][analysis_filename] = f"s3://{output_bucket}/{s3_key}"
                        logger.info(f"Analysis uploaded to S3: {s3_key}")
                else:
                    logger.warning(f"Vocals file not found for analysis: {vocals_path}")

            except Exception as e:
                logger.error(f"Error during chord/key/tempo analysis: {str(e)}")
                logger.error(traceback.format_exc())

        # Send completion notification with new message format
        if enable_notifications:
            send_notification(
                fcm_token,
                f"'{song_name}' is ready!",
                f"Vocals and instruments have been successfully separated ✅",
                {
                    "job_id": job_id,
                    "status": "completed",
                    "file_count": str(len(output_files)),
                    "sample_rate": str(sample_rate)
                }
            )

        response = {
            "output": output_urls,
            "sample_rate": sample_rate,
        }

        if analysis_result:
            response["analysis"] = {
                "key": analysis_result.get("key"),
                "tempo": analysis_result.get("tempo"),
                "chords": analysis_result.get("chords", [])
            }

        return response

    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        exec_error = traceback.format_exc()
        print(exec_error)

        # Extract song name for error message
        try:
            audio_url = input_data.get("audio_url", "")
            song_name = os.path.basename(audio_url).split('.')[0]
        except:
            song_name = "Unknown"


        transcribeResult = {
                "error": f"Separation failed for '{song_name}'",
                "status": False
            }

        # save result  in s3
        transcribeResult["system_usage"] = get_system_usage()
        transcrib_upload = save_response_to_s3(s3_client,output_prefix,output_bucket,job_id, transcribeResult, "success")
        logger.info(f"transcrib file upload processed done: {transcrib_upload}")

        # Send error notification with new message format
        if enable_notifications:
            send_notification(
                fcm_token,
                f"Error: Separation failed",
                f"Separation failed for '{song_name}'. Please try again later ❌",
                {"job_id": event.get("id", "unknown"), "status": "error", "error": str(e)}
            )

        return {"error": str(e)}

    finally:
        # Clean up workspace in all cases
        cleanup_workspace(workspace_dir)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
