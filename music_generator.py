#!/usr/bin/env python3
"""
AI Music Generator - Text-to-Music Tool
========================================
Generate background music from text descriptions using MusicGen
Powered by Meta's MusicGen model

Author: Your friendly neighborhood code wizard 🚀
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import argparse
from pathlib import Path
from typing import Optional
import logging
import warnings
from pydub import AudioSegment
import scipy.io.wavfile as wavfile

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='🎵 %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MusicGenerator:
    """
    AI Music Generator using MusicGen
    Generate background music from text descriptions!
    """

    MODELS = {
        "small": {
            "id": "facebook/musicgen-small",
            "size": "~1.5GB",
            "description": "Fast, good quality (300M params)"
        },
        "medium": {
            "id": "facebook/musicgen-medium", 
            "size": "~3.5GB",
            "description": "Balanced quality/speed (1.5B params)"
        },
        "large": {
            "id": "facebook/musicgen-large",
            "size": "~6GB",
            "description": "Best quality, slower (3.3B params)"
        },
        "melody": {
            "id": "facebook/musicgen-melody",
            "size": "~3.5GB",
            "description": "Can be conditioned on melody (1.5B params)"
        }
    }

    def __init__(
        self,
        model_name: str = "medium",
        device: str = "cuda"
    ):
        """
        Initialize Music Generator.

        Args:
            model_name: Model size (small, medium, large, melody)
            device: Device to use (cuda, cpu, mps)
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.processor = None
        
        logger.info(f"🎼 Initializing MusicGen-{model_name} on {self.device}")
        self._setup_model()

    def _get_device(self, requested_device: str) -> str:
        """Determine the best available device."""
        if requested_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif requested_device == "mps" and torch.backends.mps.is_available():
            return "mps"
        else:
            logger.warning("GPU not available, using CPU (will be slower)")
            return "cpu"

    def _setup_model(self):
        """Load MusicGen model."""
        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            
            model_id = self.MODELS[self.model_name]["id"]
            logger.info(f"📥 Loading {model_id}...")
            logger.info(f"   Size: {self.MODELS[self.model_name]['size']}")
            logger.info(f"   {self.MODELS[self.model_name]['description']}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # Load model
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.model.to(self.device)
            
            logger.info("✨ Model loaded successfully! Ready to generate music!")
            
        except ImportError:
            logger.error("❌ transformers not installed. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "accelerate"])
            # Retry
            self._setup_model()
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise

    def generate_music(
        self,
        description: str,
        duration: float = 10.0,
        output_path: str = "music.wav",
        guidance_scale: float = 3.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate music from text description.

        Args:
            description: Text description of the music (e.g., "upbeat electronic dance music")
            duration: Length of music in seconds (max 30)
            output_path: Where to save the generated music
            guidance_scale: Guidance scale for generation (higher = more aligned with text)
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            seed: Random seed for reproducibility

        Returns:
            Path to generated music file
        """
        logger.info(f"🎼 Generating music: '{description}'")
        logger.info(f"⏱️  Duration: {duration}s")

        try:
            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                logger.info(f"🎲 Using seed: {seed}")

            # Process text input
            inputs = self.processor(
                text=[description],
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Calculate max tokens for duration
            # MusicGen generates at 50Hz, so 50 tokens = 1 second
            max_new_tokens = int(duration * 50)
            
            logger.info(f"🎹 Generating with guidance_scale={guidance_scale}, temperature={temperature}")
            
            # Generate music
            with torch.no_grad():
                audio_values = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    guidance_scale=guidance_scale,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True
                )

            # Get sample rate from model config
            sample_rate = self.model.config.audio_encoder.sampling_rate
            
            # Convert to numpy
            audio_np = audio_values[0, 0].cpu().numpy()
            
            logger.info(f"🎵 Generated {len(audio_np)/sample_rate:.1f}s of music")
            logger.info(f"   Sample rate: {sample_rate}Hz")
            logger.info(f"   Audio range: [{audio_np.min():.3f}, {audio_np.max():.3f}]")

            # Save audio
            saved_path = self._save_audio(audio_np, sample_rate, output_path)
            
            logger.info(f"✅ Music saved to: {saved_path}")
            return saved_path

        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _save_audio(self, audio: np.ndarray, sample_rate: int, output_path: str) -> str:
        """Save audio to file (WAV or MP3)."""
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        file_ext = Path(output_path).suffix.lower()

        if file_ext == '.mp3':
            # Save as MP3
            audio_16bit = (audio * 32767).astype(np.int16)
            audio_segment = AudioSegment(
                audio_16bit.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )
            audio_segment.export(output_path, format='mp3', bitrate="320k")
            logger.info(f"💾 Saved as MP3 (320kbps)")
            return output_path
        else:
            # Save as WAV
            if file_ext != '.wav':
                output_path = str(Path(output_path).with_suffix('.wav'))
            
            wavfile.write(output_path, sample_rate, audio)
            logger.info(f"💾 Saved as WAV")
            return output_path


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="🎵 AI Music Generator - Create Background Music from Text!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎼 Music Generation Examples:
────────────────────────────
# Upbeat background music
python3 music_generator.py --text "upbeat electronic dance music with synths" --duration 15 --output music/background.mp3

# Calm ambient music
python3 music_generator.py --text "calm ambient atmospheric music with soft piano" --duration 30 --output music/calm.wav

# Cinematic soundtrack
python3 music_generator.py --text "epic cinematic orchestral music with drums" --duration 20 --output music/epic.mp3

# Jazz background
python3 music_generator.py --text "smooth jazz with saxophone and piano" --duration 25 --output music/jazz.mp3

🎹 Available Models:
──────────────────
- small  (1.5GB) - Fast, good quality
- medium (3.5GB) - Balanced (RECOMMENDED)
- large  (6GB)   - Best quality
- melody (3.5GB) - Can use melody conditioning

🎚️ Quality Tips:
────────────────
- Higher guidance_scale (3-5) = follows description more closely
- Lower temperature (0.8-1.0) = more coherent music
- Longer duration needs more VRAM
- Max duration: 30 seconds per generation
        """
    )

    # Text input
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Description of music to generate"
    )

    # Duration
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration in seconds (max 30)"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="generated_music.wav",
        help="Output file path (.wav or .mp3)"
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="medium",
        choices=["small", "medium", "large", "melody"],
        help="Model size to use"
    )

    # Generation parameters
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
        help="Guidance scale (higher = follows text more, 1-10)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random, 0.5-1.5)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=250,
        help="Top-k sampling parameter"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Nucleus sampling parameter"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to use"
    )

    args = parser.parse_args()

    # Validate duration
    if args.duration > 30:
        logger.warning("⚠️  Duration > 30s may cause memory issues. Capping at 30s.")
        args.duration = 30

    # Initialize generator
    try:
        generator = MusicGenerator(
            model_name=args.model,
            device=args.device
        )

        # Generate music
        output_path = generator.generate_music(
            description=args.text,
            duration=args.duration,
            output_path=args.output,
            guidance_scale=args.guidance_scale,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed
        )

        logger.info(f"🎉 Success! Music generated: {output_path}")
        logger.info("🎵 Enjoy your AI-generated music! 🎵")

    except KeyboardInterrupt:
        logger.info("\n⏹️  Generation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()