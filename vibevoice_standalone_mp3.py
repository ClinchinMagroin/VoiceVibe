#!/usr/bin/env python3
"""
VibeVoice Standalone with VibeVoice-Large-Q8 - MP3 SUPPORT VERSION
===========================================================
Direct implementation using HuggingFace Transformers with local model and tokenizer.
Supports both WAV and MP3 input/output!

Author: Your friendly neighborhood code wizard 🚀
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List
import logging
import warnings
import scipy.io.wavfile as wavfile
from pydub import AudioSegment
from scipy.signal import resample

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='🎙️ %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class VibeVoiceStandalone:
    """
    Standalone VibeVoice TTS using VibeVoice-Large-Q8 model
    Supports WAV and MP3 for both input and output!
    """

    def __init__(
        self,
        model_path: str = "models/vibevoice/VibeVoice-Large-Q8",
        tokenizer_path: str = "models/vibevoice/tokenizer",
        device: str = "cuda"
    ):
        """
        Initialize VibeVoice with local model and tokenizer.

        Args:
            model_path: Path to VibeVoice-Large-Q8 model directory
            tokenizer_path: Path to Qwen tokenizer directory
            device: "cuda" or "cpu" (GPU strongly recommended)
        """
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

        logger.info(f"🚀 Initializing VibeVoice-Large-Q8 on {self.device}")
        self._setup_model()

    def _check_dependencies(self):
        """Check if VibeVoice is available and import it with fallback installation"""
        try:
            import sys
            import os

            # Add vvembed to path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            vvembed_path = os.path.join(current_dir, 'vvembed')

            if vvembed_path not in sys.path:
                sys.path.insert(0, vvembed_path)

            # Import from embedded version
            from modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

            logger.info(f"Using embedded VibeVoice from {vvembed_path}")
            return None, VibeVoiceForConditionalGenerationInference

        except ImportError as e:
            logger.error(f"Embedded VibeVoice import failed: {e}")

            # Try fallback to installed version if available
            try:
                import vibevoice
                from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
                logger.warning("Falling back to system-installed VibeVoice")
                return vibevoice, VibeVoiceForConditionalGenerationInference
            except ImportError:
                pass

            raise Exception(
                "VibeVoice embedded module import failed. Please ensure the vvembed folder exists "
                "and transformers>=4.51.3 is installed."
            )

    def _setup_model(self):
        """Load the VibeVoice model and processor from local paths."""
        try:
            vibevoice, VibeVoiceInferenceModel = self._check_dependencies()

            # Verify paths exist
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. "
                    f"Please download VibeVoice-Large-Q8 model first."
                )

            if not self.tokenizer_path.exists():
                raise FileNotFoundError(
                    f"Tokenizer not found at {self.tokenizer_path}. "
                    f"Please download Qwen2.5-1.5B tokenizer first."
                )

            logger.info(f"📥 Loading model from: {self.model_path}")

            # Load model with automatic device mapping and quantization
            self.model = VibeVoiceInferenceModel.from_pretrained(
                str(self.model_path),
                device_map="auto",  # Auto device mapping with quantization support
                torch_dtype=torch.bfloat16,
            )

            logger.info(f"📥 Loading processor/tokenizer from: {self.model_path}")

            # Load processor (includes tokenizer)
            from processor.vibevoice_processor import VibeVoiceProcessor

            # Override thing for tokenizer path
            processor_kwargs = {
                "trust_remote_code": True,
                "language_model_pretrained_name": str(self.tokenizer_path),
                "cache_dir": str(self.model_path),
                "local_files_only": True
            }

            self.processor = VibeVoiceProcessor.from_pretrained(
                str(self.model_path),
                **processor_kwargs
            )

            logger.info(f"✨ Model loaded successfully! Ready to generate speech!")

        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise

    def generate_speech(
        self,
        text: str,
        voice_sample_path: Optional[str] = None,
        output_path: str = "output.wav",
        diffusion_steps: int = 20,
        cfg_scale: float = 1.3,
        seed: int = 42,
        use_sampling: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            voice_sample_path: Path to audio file for voice cloning (supports .wav, .mp3, .flac, .ogg)
            output_path: Where to save the output audio (supports .wav and .mp3)
            diffusion_steps: Number of denoising steps
            cfg_scale: Classifier-free guidance scale (1.0-3.0)
            seed: Random seed for reproducibility
            use_sampling: Enable sampling for variation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Path to generated audio file
        """
        logger.info(f"🎤 Generating speech for text: {text[:50]}...")

        try:
            # Set seeds for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            # Set diffusion steps
            self.model.set_ddpm_inference_steps(diffusion_steps)

            # Prepare voice samples FIRST (before processor call)
            if voice_sample_path and os.path.exists(voice_sample_path):
                logger.info(f"🎯 Using voice sample: {voice_sample_path}")
                # Load and process voice sample
                voice_np = self._load_voice_sample(voice_sample_path)
                voice_samples = [voice_np]
            else:
                logger.info("🎯 Using synthetic voice")
                voice_samples = [self._create_synthetic_voice()]

            logger.info(f"Voice samples: {len(voice_samples)} sample(s)")

            # Prepare inputs using processor WITH voice_samples
            formatted_text = f"Speaker 1: {text}"  # Single speaker format
            inputs = self.processor(
                [formatted_text],
                voice_samples=[voice_samples],  # Pass voice samples to processor!
                return_tensors="pt",
                return_attention_mask=True
            )

            # Debug inputs
            logger.info(f"Input tokens shape: {inputs['input_ids'].shape}")
            logger.info(f"Speech tensors in inputs: {inputs.get('speech_tensors') is not None}")

            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Generate with official parameters
            logger.info(f"⚙️ Generating audio with {diffusion_steps} diffusion steps...")
            with torch.no_grad():
                try:
                    # Pass inputs directly - unpack the dict to get individual tensors
                    generate_kwargs = {
                        **inputs,  # Unpack input tensors (input_ids, attention_mask, etc.)
                        "tokenizer": self.processor.tokenizer,
                        "cfg_scale": cfg_scale,
                        "max_new_tokens": None,
                        "do_sample": use_sampling,
                        "temperature": temperature if use_sampling else 1.0,
                        "top_p": top_p if use_sampling else 1.0,
                        "verbose": True,
                        "show_progress_bar": True
                    }

                    logger.info("Calling model.generate() with parameters...")
                    output = self.model.generate(**generate_kwargs)

                    logger.info(f"Model generate returned: {type(output)}")

                except Exception as gen_e:
                    logger.error(f"Model generation failed: {gen_e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise

            # Extract audio
            logger.info(f"💾 Extracting and saving audio...")
            if hasattr(output, 'speech_outputs'):
                speech_outputs = output.speech_outputs
                if speech_outputs is None:
                    logger.error("speech_outputs is None")
                    raise Exception("Model did not generate any audio output")

                logger.info(f"speech_outputs type: {type(speech_outputs)}")

                if isinstance(speech_outputs, list):
                    if len(speech_outputs) == 0:
                        raise Exception("speech_outputs list is empty")
                    elif len(speech_outputs) == 1:
                        audio_tensor = speech_outputs[0]
                    else:
                        audio_tensor = torch.cat(speech_outputs, dim=-1)
                else:
                    audio_tensor = speech_outputs

                logger.info(f"Audio tensor shape before processing: {audio_tensor.shape}")

                # Convert to numpy (handle different scenarios)
                if isinstance(audio_tensor, torch.Tensor):
                    # Convert to float32 first (numpy doesn't support bfloat16)
                    audio_tensor = audio_tensor.float()

                    # Ensure proper shape for audio (1, samples)
                    if audio_tensor.dim() == 2:
                        if audio_tensor.shape[0] == 1:
                            audio = audio_tensor.squeeze(0).cpu().numpy()
                        else:
                            audio = audio_tensor.mean(dim=0).cpu().numpy()  # Mix to mono
                    elif audio_tensor.dim() == 1:
                        audio = audio_tensor.cpu().numpy()
                    else:
                        logger.error(f"Unexpected audio tensor shape: {audio_tensor.shape}")
                        raise Exception(f"Cannot handle {audio_tensor.dim()}D audio tensor")

                else:
                    audio = np.array(audio_tensor)

                logger.info(f"Final audio shape: {audio.shape}")
                logger.info(f"Audio dtype: {audio.dtype}")
                logger.info(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

                # Save audio with format detection
                saved_path, audio_format = self._save_audio(audio, output_path)

                logger.info(f"✅ Audio saved to: {saved_path} (format: {audio_format})")
                return saved_path

            else:
                # Debug: print all attributes of output
                logger.error(f"Output has no speech_outputs attribute")
                logger.error(f"Output type: {type(output)}")
                logger.error(f"Output attributes: {[attr for attr in dir(output) if not attr.startswith('_')]}")
                raise Exception("Model output does not contain speech_outputs")

        except Exception as e:
            logger.error(f"❌ Error during generation: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _load_voice_sample(self, audio_path: str) -> np.ndarray:
        """Load and preprocess voice sample for cloning (supports WAV, MP3, FLAC, OGG)"""
        try:
            file_ext = Path(audio_path).suffix.lower()

            if file_ext == '.mp3':
                # Load MP3 using pydub
                audio = AudioSegment.from_mp3(audio_path)

                # Convert to numpy array (16-bit signed)
                voice_np = np.array(audio.get_array_of_samples(), dtype=np.float32)
                if audio.channels == 2:  # Stereo to mono
                    voice_np = voice_np.reshape((-1, 2)).mean(axis=1)

                sample_rate = audio.frame_rate

            elif file_ext in ['.wav', '.flac', '.ogg']:
                # Load with torchaudio (supports wav, flac, ogg)
                waveform, sample_rate = torchaudio.load(audio_path)

                # Ensure mono
                if waveform.dim() > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                voice_np = waveform.squeeze().numpy().astype(np.float32)

            else:
                raise ValueError(f"Unsupported audio format: {file_ext}")

            # Resample to 24kHz if needed
            if sample_rate != 24000:
                # Use scipy.signal.resample for resampling
                from scipy.signal import resample
                target_length = int(len(voice_np) * 24000 / sample_rate)
                voice_np = resample(voice_np, target_length)

            # Normalize to proper range [-1, 1]
            audio_max = np.abs(voice_np).max()
            if audio_max > 0:
                voice_np = voice_np / max(audio_max, 1.0)

            logger.info(f"Loaded voice sample: {len(voice_np)} samples at 24kHz")
            return voice_np

        except Exception as e:
            logger.warning(f"Failed to load voice sample: {e}, using synthetic voice")
            return self._create_synthetic_voice()

    def _create_synthetic_voice(self) -> np.ndarray:
        """Create a synthetic voice sample"""
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)

        t = np.linspace(0, duration, samples, False)

        # Create realistic voice-like characteristics
        base_freq = 180  # Male voice frequency
        formant1 = 800
        formant2 = 1200

        voice_sample = (
            0.6 * np.sin(2 * np.pi * base_freq * t) +
            0.25 * np.sin(2 * np.pi * base_freq * 2 * t) +
            0.15 * np.sin(2 * np.pi * base_freq * 3 * t) +
            0.1 * np.sin(2 * np.pi * formant1 * t) * np.exp(-t * 2) +
            0.05 * np.sin(2 * np.pi * formant2 * t) * np.exp(-t * 3) +
            0.02 * np.random.normal(0, 1, len(t))
        )

        envelope = np.exp(-t * 0.3) * (1 + 0.1 * np.sin(2 * np.pi * 4 * t))
        voice_sample *= envelope * 0.08

        return voice_sample.astype(np.float32)

    def _save_audio(self, audio: np.ndarray, output_path: str) -> tuple[str, str]:
        """Save audio in the appropriate format (WAV or MP3)"""

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        file_ext = Path(output_path).suffix.lower()

        sample_rate = 24000  # VibeVoice standard

        if file_ext == '.mp3':
            # Save as MP3 using pydub
            # Convert to 16-bit signed integer for MP3 (pydub needs integers)
            # Scale audio to [-32768, 32767] range
            audio_16bit = (audio * 32767).astype(np.int16)

            # Create AudioSegment from numpy array
            audio_segment = AudioSegment(
                audio_16bit.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # 16-bit = 2 bytes per sample
                channels=1
            )

            audio_segment.export(output_path, format='mp3')
            logger.info("Saved as MP3 format")
            return output_path, 'mp3'

        elif file_ext in ['.wav', '']:
            # Default to WAV
            wavfile.write(output_path, sample_rate, audio)
            logger.info("Saved as WAV format")
            return output_path, 'wav'

        else:
            # Default to WAV
            wav_path = str(Path(output_path).with_suffix('.wav'))
            wavfile.write(wav_path, sample_rate, audio)
            logger.warning(f"Unknown format '{file_ext}', saved as WAV: {wav_path}")
            return wav_path, 'wav'


def main():
    """Main CLI interface for the VibeVoice tool."""
    parser = argparse.ArgumentParser(
        description="🎙️ VibeVoice Standalone - TTS with MP3 Support!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎵 MP3 Support Examples:
-----------------------
# Generate MP3 output:
python vibevoice_standalone_mp3.py --text "Hello world!" --output audio.mp3

# Voice cloning with MP3 input:
python vibevoice_standalone_mp3.py --text "Wow!" --voice-sample voice.mp3 --output output.mp3

# Text-to-speech:
python vibevoice_standalone_mp3.py --text "VibeVoice is working!" --output speech.mp3

Supported formats: .wav, .mp3, .flac, .ogg input | .wav, .mp3 output
        """
    )

    # Input options
    parser.add_argument(
        "--text",
        type=str,
        help="Text to synthesize"
    )
    parser.add_argument(
        "--text-file",
        type=str,
        help="Load text from file"
    )

    # Voice cloning
    parser.add_argument(
        "--voice-sample",
        type=str,
        help="Audio file for voice cloning (.wav, .mp3, .flac, .ogg supported)"
    )

    # Generation parameters
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=20,
        help="Number of denoising steps (more = better quality, slower)"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.3,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--sampling",
        action="store_true",
        help="Enable sampling for variation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (only when sampling enabled)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter (only when sampling enabled)"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path (.wav or .mp3)"
    )

    # Model paths
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/vibevoice/VibeVoice-Large-Q8",
        help="Path to VibeVoice-Large-Q8 model directory"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="models/vibevoice/tokenizer",
        help="Path to Qwen tokenizer directory"
    )

    # Device
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU (slow)"
    )

    args = parser.parse_args()

    # Load text
    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        logger.error("❌ Please provide text with --text or --text-file")
        sys.exit(1)

    # Initialize VibeVoice
    device = "cpu" if args.cpu else "cuda"
    vv = VibeVoiceStandalone(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=device
    )

    # Generate speech
    try:
        output_path = vv.generate_speech(
            text=text,
            voice_sample_path=args.voice_sample,
            output_path=args.output,
            diffusion_steps=args.diffusion_steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            use_sampling=args.sampling,
            temperature=args.temperature,
            top_p=args.top_p
        )

        logger.info(f"🎉 Success! Audio generated: {output_path}")

    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()