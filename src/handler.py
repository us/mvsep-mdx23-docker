# coding: utf-8

import os
import warnings
warnings.filterwarnings("ignore")

import inspect
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import argparse
import soundfile as sf
from demucs.states import load_model
from demucs import pretrained
from demucs.apply import apply_model
import onnxruntime as ort
from time import time
import librosa
import hashlib
from scipy import signal
import gc
import yaml
from ml_collections import ConfigDict
import sys
import math
import pathlib
import warnings
import logging
from pathlib import Path
import runpod
import json
import traceback
import tempfile
import boto3
from moviepy import VideoFileClip
import static_ffmpeg
from pydub import AudioSegment
import shutil
import firebase_admin
from firebase_admin import credentials, messaging

from modules.tfc_tdf_v2 import Conv_TDF_net_trim_model
from modules.tfc_tdf_v3 import TFC_TDF_net, STFT
from modules.segm_models import Segm_Models_Net
from modules.bs_roformer import BSRoformer
from modules.bs_roformer import MelBandRoformer
from chords_tempo import analyze
import whisperx

# Initialize ffmpeg paths early
static_ffmpeg.add_paths()

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model instance
MODEL = None
# Firebase app instance
FIREBASE_APP = None

def get_models(name, device, load=True, vocals_model_type=0):
    if vocals_model_type == 2:
        model_vocals = Conv_TDF_net_trim_model(
            device=device,
            target_name='vocals',
            L=11,
            n_fft=7680,
            dim_f=3072
        )
    elif vocals_model_type == 3:
        model_vocals = Conv_TDF_net_trim_model(
            device=device,
            target_name='instrum',
            L=11,
            n_fft=5120,
            dim_f=2560
        )

    return [model_vocals]


def get_model_from_config(model_type, config_path):
    with open(config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
        if model_type == 'mdx23c':
            model = TFC_TDF_net(config)
        elif model_type == 'segm_models':
            model = Segm_Models_Net(config)
        elif model_type == 'bs_roformer':
            model = BSRoformer(
                **dict(config.model)
            )
        elif model_type == 'mel_band_roformer':
            model = MelBandRoformer(
                **dict(config.model)
            )
        else:
            print('Unknown model: {}'.format(model_type))
            model = None
    return model, config


def demix_new(model, mix, device, config, dim_t=256):
    mix = torch.tensor(mix)
    #N = options["overlap_BSRoformer"]
    N = 2 # overlap 50%
    batch_size = 1
    mdx_window_size = dim_t
    C = config.audio.hop_length * (mdx_window_size - 1)
    fade_size = C // 100
    step = int(C // N)
    border = C - step
    length_init = mix.shape[-1]
    #print(f"1: {mix.shape}")
    
    # Do pad from the beginning and end to account floating window results better
    if length_init > 2 * border and (border > 0):
        mix = nn.functional.pad(mix, (border, border), mode='reflect')
        
    
    # Prepare windows arrays (do 1 time for speed up). This trick repairs click problems on the edges of segment
    window_size = C
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window_start = torch.ones(window_size)
    window_middle = torch.ones(window_size)
    window_finish = torch.ones(window_size)
    window_start[-fade_size:] *= fadeout # First audio chunk, no fadein
    window_finish[:fade_size] *= fadein # Last audio chunk, no fadeout
    window_middle[-fade_size:] *= fadeout
    window_middle[:fade_size] *= fadein




    with torch.cuda.amp.autocast():
        with torch.inference_mode():
            if config.training.target_instrument is not None:
                req_shape = (1, ) + tuple(mix.shape)
            else:
                req_shape = (len(config.training.instruments),) + tuple(mix.shape)

            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
            i = 0
            batch_data = []
            batch_locations = []
            while i < mix.shape[1]:
                # print(i, i + C, mix.shape[1])
                part = mix[:, i:i + C].to(device)
                length = part.shape[-1]
                if length < C:
                    if length > C // 2 + 1:
                        part = nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
                    else:
                        part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step

                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)

                    window = window_middle
                    if i - step == 0:  # First audio chunk, no fadein
                        window = window_start
                    elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                        window = window_finish

                    for j in range(len(batch_locations)):
                        start, l = batch_locations[j]
                        result[..., start:start+l] += x[j][..., :l].cpu() * window[..., :l]
                        counter[..., start:start+l] += window[..., :l]

                    batch_data = []
                    batch_locations = []

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            if length_init > 2 * border and (border > 0):
                # Remove pad
                estimated_sources = estimated_sources[..., border:-border]

    if config.training.target_instrument is None:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources)}
    else:
        return {k: v for k, v in zip([config.training.target_instrument], estimated_sources)}


def demix_new_wrapper(mix, device, model, config, dim_t=256, bigshifts=1):
    if bigshifts <= 0:
        bigshifts = 1

    shift_in_samples = mix.shape[1] // bigshifts
    shifts = [x * shift_in_samples for x in range(bigshifts)]

    results = []

    for shift in tqdm(shifts, position=0):
        shifted_mix = np.concatenate((mix[:, -shift:], mix[:, :-shift]), axis=-1)
        sources = demix_new(model, shifted_mix, device, config, dim_t=dim_t)
        vocals = next(sources[key] for key in sources.keys() if key.lower() == "vocals")
        unshifted_vocals = np.concatenate((vocals[..., shift:], vocals[..., :shift]), axis=-1)  
        vocals *= 1 # 1.0005168 CHECK NEEDED! volume compensation
        
        results.append(unshifted_vocals)

    vocals = np.mean(results, axis=0)
    return vocals

def demix_vitlarge(model, mix, device):
    C = model.config.audio.hop_length * (2 * model.config.inference.dim_t - 1)
    N = 2
    step = C // N

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            if model.config.training.target_instrument is not None:
                req_shape = (1, ) + tuple(mix.shape)
            else:
                req_shape = (len(model.config.training.instruments),) + tuple(mix.shape)

            mix = mix.to(device)
            result = torch.zeros(req_shape, dtype=torch.float32).to(device)
            counter = torch.zeros(req_shape, dtype=torch.float32).to(device)
            i = 0

            while i < mix.shape[1]:
                part = mix[:, i:i + C]
                length = part.shape[-1]
                if length < C:
                    part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                x = model(part.unsqueeze(0))[0]
                result[..., i:i+length] += x[..., :length]
                counter[..., i:i+length] += 1.
                i += step
            estimated_sources = result / counter

    if model.config.training.target_instrument is None:
        return {k: v for k, v in zip(model.config.training.instruments, estimated_sources.cpu().numpy())}
    else:
        return {k: v for k, v in zip([model.config.training.target_instrument], estimated_sources.cpu().numpy())}


def demix_full_vitlarge(options, mix, device, model):
    if options["BigShifts"] <= 0:
        bigshifts = 1
    else:
        bigshifts = options["BigShifts"]
    shift_in_samples = mix.shape[1] // bigshifts
    shifts = [x * shift_in_samples for x in range(bigshifts)]

    results1 = []
    results2 = []
    mix = torch.from_numpy(mix).type('torch.FloatTensor').to(device)
    for shift in tqdm(shifts, position=0):
        shifted_mix = torch.cat((mix[:, -shift:], mix[:, :-shift]), dim=-1)
        sources = demix_vitlarge(model, shifted_mix, device)
        sources1 = sources["vocals"]
        sources2 = sources["other"]
        restored_sources1 = np.concatenate((sources1[..., shift:], sources1[..., :shift]), axis=-1)
        restored_sources2 = np.concatenate((sources2[..., shift:], sources2[..., :shift]), axis=-1)
        results1.append(restored_sources1)
        results2.append(restored_sources2)


    sources1 = np.mean(results1, axis=0)
    sources2 = np.mean(results2, axis=0)

    return sources1, sources2


def demix_wrapper(mix, device, models, infer_session, overlap=0.2, bigshifts=1, vc=1.0):
    if bigshifts <= 0:
        bigshifts = 1
    shift_in_samples = mix.shape[1] // bigshifts
    shifts = [x * shift_in_samples for x in range(bigshifts)]
    results = []
    
    for shift in tqdm(shifts, position=0):
        shifted_mix = np.concatenate((mix[:, -shift:], mix[:, :-shift]), axis=-1)
        sources = demix(shifted_mix, device, models, infer_session, overlap) * vc # 1.021 volume compensation
        restored_sources = np.concatenate((sources[..., shift:], sources[..., :shift]), axis=-1)
        results.append(restored_sources)
        
    sources = np.mean(results, axis=0)
    
    return sources

def demix(mix, device, models, infer_session, overlap=0.2):
    start_time = time()
    sources = []
    n_sample = mix.shape[1]
    n_fft = models[0].n_fft
    n_bins = n_fft//2+1
    trim = n_fft//2
    hop = models[0].hop
    dim_f = models[0].dim_f
    dim_t = models[0].dim_t # * 2
    chunk_size = hop * (dim_t -1)
    org_mix = mix
    tar_waves_ = []
    mdx_batch_size = 1
    overlap = overlap
    gen_size = chunk_size-2*trim
    pad = gen_size + trim - ((mix.shape[-1]) % gen_size)
    
    mixture = np.concatenate((np.zeros((2, trim), dtype='float32'), mix, np.zeros((2, pad), dtype='float32')), 1)

    step = int((1 - overlap) * chunk_size)
    result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
    divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
    total = 0
    total_chunks = (mixture.shape[-1] + step - 1) // step

    for i in range(0, mixture.shape[-1], step):
        total += 1
        start = i
        end = min(i + chunk_size, mixture.shape[-1])
        chunk_size_actual = end - start

        if overlap == 0:
            window = None
        else:
            window = np.hanning(chunk_size_actual)
            window = np.tile(window[None, None, :], (1, 2, 1))

        mix_part_ = mixture[:, start:end]
        if end != i + chunk_size:
            pad_size = (i + chunk_size) - end
            mix_part_ = np.concatenate((mix_part_, np.zeros((2, pad_size), dtype='float32')), axis=-1)
        
        
        mix_part = torch.tensor([mix_part_], dtype=torch.float32).to(device)
        mix_waves = mix_part.split(mdx_batch_size)
        
        with torch.no_grad():
            for mix_wave in mix_waves:
                _ort = infer_session
                stft_res = models[0].stft(mix_wave)
                stft_res[:, :, :3, :] *= 0 
                res = _ort.run(None, {'input': stft_res.cpu().numpy()})[0]
                ten = torch.tensor(res)
                tar_waves = models[0].istft(ten.to(device))
                tar_waves = tar_waves.cpu().detach().numpy()
                
                if window is not None:
                    tar_waves[..., :chunk_size_actual] *= window 
                    divider[..., start:end] += window
                else:
                    divider[..., start:end] += 1
                result[..., start:end] += tar_waves[..., :end-start]


    tar_waves = result / divider
    tar_waves_.append(tar_waves)
    tar_waves_ = np.vstack(tar_waves_)[:, :, trim:-trim]
    tar_waves = np.concatenate(tar_waves_, axis=-1)[:, :mix.shape[-1]]
    source = tar_waves[:,0:None]

    return source

class EnsembleDemucsMDXMusicSeparationModel:
    """
    Music separation model with preloaded models
    """
    def __init__(self, options):
        """
            options - user options
        """
        # Device setup
        if torch.cuda.is_available() and not options.get('cpu', False):
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        
        self.single_onnx = options.get('single_onnx', False)
        self.overlap_demucs = min(max(float(options['overlap_demucs']), 0.0), 0.99)
        self.overlap_MDX = min(max(float(options['overlap_VOCFT']), 0.0), 0.99)
        
        # Model folder
        self.model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')
        
        self.options = options

        # Execution providers for ONNX
        if self.device == 'cpu':
            self.providers = ["CPUExecutionProvider"]
        else:
            self.providers = ["CUDAExecutionProvider"]
        
        # Preloading ensemble models (if not vocals only)
        if not options.get('vocals_only', False):
            self.models = []
            self.weights_vocals = np.array([10, 1, 8, 9])
            self.weights_bass = np.array([19, 4, 5, 8])
            self.weights_drums = np.array([18, 2, 4, 9])
            self.weights_other = np.array([14, 2, 5, 10])
            
            model_names = ['htdemucs_ft', 'htdemucs', 'htdemucs_6s', 'hdemucs_mmi']
            for model_name in model_names:
                model = pretrained.get_model(model_name, repo=pathlib.Path(self.model_folder))
                model.to(self.device)
                self.models.append(model)

        # Preload vocal models based on options
        self.preload_vocal_models(options)

    def preload_vocal_models(self, options):
        """Preload vocal models based on configuration"""
        logger.info("Preloading vocal models...")
        
        if options.get("use_BSRoformer", False):
            logger.info('Preloading BSRoformer model')
            bs_model_name = "model_bs_roformer_ep_368_sdr_12.9628" if options["BSRoformer_model"] == "ep_368_1296" else "model_bs_roformer_ep_317_sdr_12.9755"
            self.model_bsrofo, self.config_bsrofo = self.load_model(bs_model_name, BSRoformer)

        if options.get("use_Kim_MelRoformer", False):
            logger.info('Preloading Kim_MelRoformer model')
            self.model_melrofo, self.config_melrofo = self.load_model('Kim_MelRoformer', MelBandRoformer)

        if options.get("use_InstVoc", False):
            logger.info('Preloading InstVoc model')
            self.model_mdxv3, self.config_mdxv3 = self.load_model('MDX23C-8KFFT-InstVoc_HQ', TFC_TDF_net)

        if options.get("use_VitLarge", False):
            logger.info('Preloading VitLarge model')
            self.model_vl, self.config_vl = self.load_model('model_vocals_segm_models_sdr_9.77', Segm_Models_Net)

        if options.get("use_VOCFT", False):
            logger.info('Preloading VOCFT model')
            self.infer_session1 = self.load_onnx_model('UVR-MDX-NET-Voc_FT')
            self.mdx_models1 = get_models('tdf_extra', load=False, device=self.device, vocals_model_type=2)

        if options.get("use_InstHQ4", False):
            logger.info('Preloading InstHQ4 model')
            self.infer_session2 = self.load_onnx_model('UVR-MDX-NET-Inst_HQ_4')
            self.mdx_models2 = get_models('tdf_extra', load=False, device=self.device, vocals_model_type=3)

        logger.info("Vocal models preloaded successfully")

    def load_model(self, model_name, model_class):
        """Loads a model from the models directory"""
        # Special handling for model file names that differ from model_name
        if model_name == 'Kim_MelRoformer':
            ckpt_path = os.path.join(self.model_folder, 'MelBandRoformer.ckpt')
            yaml_path = os.path.join(self.model_folder, 'config_vocals_mel_band_roformer_kj.yaml')
        elif model_name == 'BSRoformer':
            # Choose BSRoformer model version based on options
            bs_model_name = "model_bs_roformer_ep_368_sdr_12.9628" if self.options["BSRoformer_model"] == "ep_368_1296" else "model_bs_roformer_ep_317_sdr_12.9755"
            ckpt_path = os.path.join(self.model_folder, f'{bs_model_name}.ckpt')
            yaml_path = os.path.join(self.model_folder, f'{bs_model_name}.yaml')
        else:
            ckpt_path = os.path.join(self.model_folder, f'{model_name}.ckpt')
            # Special handling for models with different yaml names
            if model_name == 'MDX23C-8KFFT-InstVoc_HQ':
                yaml_path = os.path.join(self.model_folder, 'model_2_stem_full_band_8k.yaml')
            elif model_name == 'model_vocals_segm_models_sdr_9.77':
                yaml_path = os.path.join(self.model_folder, 'config_vocals_segm_models.yaml')
            else:
                yaml_path = os.path.join(self.model_folder, f'{model_name}.yaml')

        if not os.path.exists(ckpt_path) or not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Model files not found for {model_name}. Please run download_models.py first.\nLooking for:\n- {ckpt_path}\n- {yaml_path}")

        # Load configuration
        with open(yaml_path, 'r') as f:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

        # Get valid arguments for the model constructor
        model_args = inspect.signature(model_class.__init__).parameters
        valid_config = {key: value for key, value in dict(config.model).items() if key in model_args}

        # If the model requires a 'config' argument, pass the full config object
        if 'config' in model_args:
            model = model_class(config=config)
        else:
            model = model_class(**valid_config)

        model.load_state_dict(torch.load(ckpt_path))
        model = model.to(self.device)
        model.eval()

        return model, config

    def load_onnx_model(self, model_name):
        """Loads an ONNX model from the models directory"""
        model_path = os.path.join(self.model_folder, f'{model_name}.onnx')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model {model_name} not found. Please run download_models.py first.")
            
        return ort.InferenceSession(model_path, providers=self.providers, provider_options=[{"device_id": 0}])

    def initialize_model_if_needed(self, model_name, options):
        """Models are now preloaded in __init__, this is for compatibility"""
        pass

    @property
    def instruments(self):
        if not self.options.get('vocals_only', False):
            return ['bass', 'drums', 'other', 'vocals']
        else:
            return ['vocals']

    def separate_music_file(self, options, mixed_sound_array, sample_rate, current_file_number=0, total_files=0):
        """
        Implements the sound separation for a single sound file
        """
        separated_music_arrays = {}
        output_sample_rates = {}
        
        overlap_demucs = self.overlap_demucs
        overlap_MDX = self.overlap_MDX
        shifts = 0
        overlap = overlap_demucs

        vocals_model_names = [
            "BSRoformer",
            "Kim_MelRoformer",
            "InstVoc",
            "VitLarge",
            "VOCFT",
            "InstHQ4"
        ]

        vocals_model_outputs = []
        weights = []
        
        for model_name in vocals_model_names:
            if options[f"use_{model_name}"]:

                if model_name == "BSRoformer":
                    print(f'Processing vocals with {model_name} model...')
                    sources_bs = demix_new_wrapper(mixed_sound_array.T, self.device, self.model_bsrofo, self.config_bsrofo, dim_t=1101, bigshifts=options["BigShifts"])
                    vocals_bs = match_array_shapes(sources_bs, mixed_sound_array.T)
                    vocals_model_outputs.append(vocals_bs)
                    weights.append(options.get(f"weight_{model_name}"))

                elif model_name == "Kim_MelRoformer":
                    print(f'Processing vocals with {model_name} model...')
                    sources_mel = demix_new_wrapper(mixed_sound_array.T, self.device, self.model_melrofo, self.config_melrofo, dim_t=1101, bigshifts=options["BigShifts"])
                    vocals_mel = match_array_shapes(sources_mel, mixed_sound_array.T)
                    vocals_model_outputs.append(vocals_mel)
                    weights.append(options.get(f"weight_{model_name}"))

                elif model_name == "InstVoc":
                    print(f'Processing vocals with {model_name} model...')
                    sources3 = demix_new_wrapper(mixed_sound_array.T, self.device, self.model_mdxv3, self.config_mdxv3, dim_t=2048, bigshifts=options["BigShifts"])
                    vocals3 = match_array_shapes(sources3, mixed_sound_array.T)
                    vocals_model_outputs.append(vocals3)
                    weights.append(options.get(f"weight_{model_name}"))

                elif model_name == "VitLarge":
                    print(f'Processing vocals with {model_name} model...')
                    vocals4, instrum4 = demix_full_vitlarge(options, mixed_sound_array.T, self.device, self.model_vl)
                    vocals4 = match_array_shapes(vocals4, mixed_sound_array.T)
                    vocals_model_outputs.append(vocals4)
                    weights.append(options.get(f"weight_{model_name}"))

                elif model_name == "VOCFT":
                    print(f'Processing vocals with {model_name} model...')
                    overlap = overlap_MDX
                    vocals_mdxb1 = 0.5 * demix_wrapper(
                        mixed_sound_array.T,
                        self.device,
                        self.mdx_models1,
                        self.infer_session1,
                        overlap=overlap,
                        vc=1.021,
                        bigshifts=options['BigShifts'] // 3
                    )
                    vocals_mdxb1 += 0.5 * -demix_wrapper(
                        -mixed_sound_array.T,
                        self.device,
                        self.mdx_models1,
                        self.infer_session1,
                        overlap=overlap,
                        vc=1.021,
                        bigshifts=options['BigShifts'] // 3
                    )
                    vocals_model_outputs.append(vocals_mdxb1)
                    weights.append(options.get(f"weight_{model_name}"))

                elif model_name == "InstHQ4":
                    print(f'Processing vocals with {model_name} model...')
                    overlap = overlap_MDX
                    sources2 = 0.5 * demix_wrapper(
                        mixed_sound_array.T,
                        self.device,
                        self.mdx_models2,
                        self.infer_session2,
                        overlap=overlap,
                        vc=1.019,
                        bigshifts=options['BigShifts'] // 3
                    )
                    sources2 += 0.5 * -demix_wrapper(
                        -mixed_sound_array.T,
                        self.device,
                        self.mdx_models2,
                        self.infer_session2,
                        overlap=overlap,
                        vc=1.019,
                        bigshifts=options['BigShifts'] // 3
                    )
                    vocals_mdxb2 = mixed_sound_array.T - sources2
                    vocals_model_outputs.append(vocals_mdxb2)
                    weights.append(options.get(f"weight_{model_name}"))

                else:
                    # No more model to process or unknown one
                    pass

        print('Processing vocals: DONE!')
        
        vocals_combined = np.zeros_like(vocals_model_outputs[0])

        for output, weight in zip(vocals_model_outputs, weights):
            vocals_combined += output * weight

        vocals_combined /= np.sum(weights)
        del vocals_model_outputs

        if options['use_VOCFT']:
            vocals_low = lr_filter(vocals_combined.T, 12000, 'lowpass')
            vocals_high = lr_filter(vocals3.T, 12000, 'highpass')
            vocals = vocals_low + vocals_high
        else:
            vocals = vocals_combined.T

        if options['filter_vocals'] is True:
                vocals = lr_filter(vocals, 50, 'highpass', order=8)
        
        # Generate instrumental
        instrum = mixed_sound_array - vocals
        
        if options['vocals_only'] is False:
            
            audio = np.expand_dims(instrum.T, axis=0)
            audio = torch.from_numpy(audio).type('torch.FloatTensor').to(self.device)
            all_outs = []
            print('Processing with htdemucs_ft...')
            i = 0
            overlap = overlap_demucs
            model = pretrained.get_model('htdemucs_ft', repo=pathlib.Path(self.model_folder))
            model.to(self.device)
            out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
                  + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()
       
            out[0] = self.weights_drums[i] * out[0]
            out[1] = self.weights_bass[i] * out[1]
            out[2] = self.weights_other[i] * out[2]
            out[3] = self.weights_vocals[i] * out[3]
            all_outs.append(out)
            model = model.cpu()
            del model
            gc.collect()
            i = 1
            print('Processing with htdemucs...')
            overlap = overlap_demucs
            model = pretrained.get_model('htdemucs', repo=pathlib.Path(self.model_folder))
            model.to(self.device)
            out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
                  + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()
    
            out[0] = self.weights_drums[i] * out[0]
            out[1] = self.weights_bass[i] * out[1]
            out[2] = self.weights_other[i] * out[2]
            out[3] = self.weights_vocals[i] * out[3]
            all_outs.append(out)
            model = model.cpu()
            del model
            gc.collect()
            i = 2
            print('Processing with htdemucs_6s...')
            overlap = overlap_demucs
            model = pretrained.get_model('htdemucs_6s', repo=pathlib.Path(self.model_folder))
            model.to(self.device)
            out = apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()
       
            # More stems need to add
            out[2] = out[2] + out[4] + out[5]
            out = out[:4]
            out[0] = self.weights_drums[i] * out[0]
            out[1] = self.weights_bass[i] * out[1]
            out[2] = self.weights_other[i] * out[2]
            out[3] = self.weights_vocals[i] * out[3]
            all_outs.append(out)
            model = model.cpu()
            del model
            gc.collect()
            i = 3
            print('Processing with htdemucs_mmi...')
            model = pretrained.get_model('hdemucs_mmi', repo=pathlib.Path(self.model_folder))
            model.to(self.device)
            out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
                  + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()
       
            out[0] = self.weights_drums[i] * out[0]
            out[1] = self.weights_bass[i] * out[1]
            out[2] = self.weights_other[i] * out[2]
            out[3] = self.weights_vocals[i] * out[3]
            all_outs.append(out)
            model = model.cpu()
            del model
            gc.collect()
            out = np.array(all_outs).sum(axis=0)
            out[0] = out[0] / self.weights_drums.sum()
            out[1] = out[1] / self.weights_bass.sum()
            out[2] = out[2] / self.weights_other.sum()
            out[3] = out[3] / self.weights_vocals.sum()

            # other
            res = mixed_sound_array - vocals - out[0].T - out[1].T
            res = np.clip(res, -1, 1)
            separated_music_arrays['other'] = (2 * res + out[2].T) / 3.0
            output_sample_rates['other'] = sample_rate
    
            # drums
            res = mixed_sound_array - vocals - out[1].T - out[2].T
            res = np.clip(res, -1, 1)
            separated_music_arrays['drums'] = (res + 2 * out[0].T.copy()) / 3.0
            output_sample_rates['drums'] = sample_rate
    
            # bass
            res = mixed_sound_array - vocals - out[0].T - out[2].T
            res = np.clip(res, -1, 1)
            separated_music_arrays['bass'] = (res + 2 * out[1].T) / 3.0
            output_sample_rates['bass'] = sample_rate
    
            bass = separated_music_arrays['bass']
            drums = separated_music_arrays['drums']
            other = separated_music_arrays['other']
    
            separated_music_arrays['other'] = mixed_sound_array - vocals - bass - drums
            separated_music_arrays['drums'] = mixed_sound_array - vocals - bass - other
            separated_music_arrays['bass'] = mixed_sound_array - vocals - drums - other

        # vocals
        separated_music_arrays['vocals'] = vocals
        output_sample_rates['vocals'] = sample_rate

        # instrum
        separated_music_arrays['instrum'] = instrum

        return separated_music_arrays, output_sample_rates


def predict_with_model(options):
    """Compatibility function - now handled directly in handler"""
    pass


# Linkwitz-Riley filter
def lr_filter(audio, cutoff, filter_type, order=6, sr=44100):
    audio = audio.T
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order//2, normal_cutoff, btype=filter_type, analog=False)
    sos = signal.tf2sos(b, a)
    filtered_audio = signal.sosfiltfilt(sos, audio)
    return filtered_audio.T

def match_array_shapes(array_1:np.ndarray, array_2:np.ndarray):
    if array_1.shape[1] > array_2.shape[1]:
        array_1 = array_1[:,:array_2.shape[1]] 
    elif array_1.shape[1] < array_2.shape[1]:
        padding = array_2.shape[1] - array_1.shape[1]
        array_1 = np.pad(array_1, ((0,0), (0,padding)), 'constant', constant_values=0)
    return array_1

def dBgain(audio, volume_gain_dB):
    attenuation = 10 ** (volume_gain_dB / 20)
    gained_audio = audio * attenuation
    return gained_audio


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
        return True

    try:
        cred_path = None

        if credentials_s3_url and s3_client and workspace_dir:
            local_cred_path = os.path.join(workspace_dir, "firebase_credentials.json")
            if download_from_s3(s3_client, credentials_s3_url, local_cred_path):
                cred_path = local_cred_path
                logger.info(f"Downloaded Firebase credentials from {credentials_s3_url}")

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

    input_data = event.get("input", {})
    notification_config = input_data.get("notification", {})
    fcm_token = notification_config.get("fcm_token")
    enable_notifications = bool(fcm_token) and notification_config.get("enabled", False)
    firebase_creds_url = notification_config.get("credentials_url")

    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )

    if enable_notifications:
        init_firebase(s3_client, firebase_creds_url, workspace_dir)

    try:
        audio_url = input_data["audio_url"]
        job_id = event["id"]
        output_bucket = input_data.get("output_bucket")
        output_prefix = input_data.get("output_prefix", "")
        output_format = input_data.get("output_format", "FLOAT")
        options = input_data.get("options", {})

        song_name = os.path.basename(audio_url).split('.')[0]

        bucket_name, key = audio_url.replace("s3://", "").split("/", 1)
        input_path = os.path.join(workspace_dir, "input" + os.path.splitext(key)[1])
        logger.info(f"Downloading from S3: {audio_url}")
        s3_client.download_file(bucket_name, key, input_path)

        processed_path = process_audio_file(input_path, workspace_dir)

        audio_info = sf.info(processed_path)
        sample_rate = audio_info.samplerate

        options.update({
            "input_audio": [processed_path],
            "output_folder": workspace_dir,
            "output_format": output_format
        })

        logger.info("Initializing model and running inference")
        model = init_model(options)
        result = predict_with_model(options)

        analysis_result = None
        if input_data.get("enable_analysis", False):
            try:
                logger.info("Running chord/key/tempo analysis")

                output_extension = 'flac' if output_format == 'FLAC' else 'wav'
                vocals_filename = os.path.splitext(os.path.basename(processed_path))[0] + f'_vocals.{output_extension}'
                vocals_path = os.path.join(workspace_dir, vocals_filename)

                if os.path.exists(vocals_path):
                    analysis_result = analyze(vocals_path, rounding=2)
                    logger.info(f"Analysis complete: Key={analysis_result['key']}, Tempo={analysis_result['tempo']}, Chords={len(analysis_result['chords'])}")

                    analysis_filename = os.path.splitext(os.path.basename(processed_path))[0] + '_analysis.json'
                    analysis_path = os.path.join(workspace_dir, analysis_filename)
                    with open(analysis_path, 'w') as f:
                        json.dump(analysis_result, f, indent=2)
                    logger.info(f"Analysis saved to: {analysis_filename}")
                else:
                    logger.warning(f"Vocals file not found for analysis: {vocals_path}")

            except Exception as e:
                logger.error(f"Error during analysis: {str(e)}")
                logger.error(traceback.format_exc())

        transcription_result = None
        enable_lyrics = input_data.get("enable_lyrics", False)
        if enable_lyrics and os.path.exists(vocals_path if 'vocals_path' in locals() else ''):
            try:
                logger.info("Running WhisperX transcription with large-v3 model on GPU")

                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                batch_size = 16 if options.get("large_gpu", True) else 8

                model = whisperx.load_model("large-v3", device, compute_type=compute_type, download_root="/models/whisper")
                audio = whisperx.load_audio(vocals_path)

                result = model.transcribe(audio, batch_size=batch_size)

                del model
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

                model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
                result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

                del model_a
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

                transcription_result = result
                logger.info(f"Transcription complete: {len(result['segments'])} segments, language={result.get('language')}")

                transcription_filename = os.path.splitext(os.path.basename(processed_path))[0] + '_transcription.json'
                transcription_path = os.path.join(workspace_dir, transcription_filename)
                with open(transcription_path, 'w') as f:
                    json.dump(transcription_result, f, indent=2)
                logger.info(f"Transcription saved to: {transcription_filename}")

            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}")
                logger.error(traceback.format_exc())

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

        if transcription_result:
            response["transcription"] = {
                "language": transcription_result.get("language"),
                "segments": transcription_result.get("segments", [])
            }

        return response

    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        exec_error = traceback.format_exc()
        print(exec_error)

        try:
            audio_url = input_data.get("audio_url", "")
            song_name = os.path.basename(audio_url).split('.')[0]
        except:
            song_name = "Unknown"

        if enable_notifications:
            send_notification(
                fcm_token,
                f"Error: Separation failed",
                f"Separation failed for '{song_name}'. Please try again later ❌",
                {"job_id": event.get("id", "unknown"), "status": "error", "error": str(e)}
            )

        return {"error": str(e)}

    finally:
        cleanup_workspace(workspace_dir)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})