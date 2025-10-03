#!/usr/bin/env python3
"""
Audio Mixer - Combine Voice and Music
======================================
Mix VibeVoice speech with background music

Author: Your friendly neighborhood code wizard 🚀
"""

import os
import sys
import argparse
from pathlib import Path
import logging
import numpy as np
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO, format='🎧 %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def mix_audio(
    voice_path: str,
    music_path: str,
    output_path: str,
    voice_volume: float = 1.0,
    music_volume: float = 0.3,
    fade_in: float = 0.0,
    fade_out: float = 0.0
) -> str:
    """
    Mix voice and background music.

    Args:
        voice_path: Path to voice/speech audio file
        music_path: Path to background music file
        output_path: Output file path
        voice_volume: Voice volume multiplier (0.0-2.0)
        music_volume: Music volume multiplier (0.0-1.0)
        fade_in: Music fade-in duration (seconds)
        fade_out: Music fade-out duration (seconds)

    Returns:
        Path to mixed audio file
    """
    try:
        logger.info(f"🎙️ Loading voice: {voice_path}")
        voice = AudioSegment.from_file(voice_path)
        
        logger.info(f"🎵 Loading music: {music_path}")
        music = AudioSegment.from_file(music_path)

        # Adjust volumes
        logger.info(f"🎚️ Adjusting volumes (voice: {voice_volume}x, music: {music_volume}x)")
        voice = voice + (20 * np.log10(voice_volume) if voice_volume != 1.0 else 0)
        music = music + (20 * np.log10(music_volume) if music_volume != 1.0 else 0)

        # Apply fades to music
        if fade_in > 0:
            fade_in_ms = int(fade_in * 1000)
            music = music.fade_in(fade_in_ms)
            logger.info(f"🌅 Applied {fade_in}s fade-in to music")

        if fade_out > 0:
            fade_out_ms = int(fade_out * 1000)
            music = music.fade_out(fade_out_ms)
            logger.info(f"🌇 Applied {fade_out}s fade-out to music")

        # Ensure music is at least as long as voice
        voice_duration = len(voice)
        if len(music) < voice_duration:
            # Loop music to match voice duration
            repeats = (voice_duration // len(music)) + 1
            music = music * repeats
            logger.info(f"🔄 Looped music to match voice duration")

        # Trim music to voice duration
        music = music[:voice_duration]

        # Mix (overlay voice on top of music)
        logger.info("🎨 Mixing audio tracks...")
        mixed = music.overlay(voice, position=0)

        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Export with format detection
        file_ext = Path(output_path).suffix.lower()
        if file_ext == '.mp3':
            mixed.export(output_path, format='mp3', bitrate="320k")
            logger.info("💾 Exported as MP3 (320kbps)")
        else:
            mixed.export(output_path, format='wav')
            logger.info("💾 Exported as WAV")

        logger.info(f"✅ Mixed audio saved: {output_path}")
        logger.info(f"📊 Duration: {len(mixed)/1000:.1f}s")

        return output_path

    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Mixing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="🎧 Audio Mixer - Combine Voice and Background Music",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎼 Complete Workflow Examples:
─────────────────────────────

# 1. Generate voice
python3 vibevoice_standalone_mp3.py --text "Welcome to my podcast!" --output voice.mp3

# 2. Generate background music
python3 music_generator.py --text "upbeat electronic music" --duration 15 --output music.mp3

# 3. Mix them together
python3 mix_audio.py --voice voice.mp3 --music music.mp3 --output final.mp3 --music-volume 0.3

🎚️ Volume Guide:
────────────────
Voice volume: 1.0 = normal, 1.5 = louder, 0.8 = quieter
Music volume: 0.2 = soft bg, 0.3 = balanced, 0.5 = prominent

🎨 Advanced Mixing:
──────────────────
# Add fades for professional sound
python3 mix_audio.py --voice podcast.mp3 --music bg.mp3 --output final.mp3 \
    --music-volume 0.25 --fade-in 2 --fade-out 3
        """
    )

    parser.add_argument(
        "--voice",
        type=str,
        required=True,
        help="Voice/speech audio file"
    )

    parser.add_argument(
        "--music",
        type=str,
        required=True,
        help="Background music file"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (.mp3 or .wav)"
    )

    parser.add_argument(
        "--voice-volume",
        type=float,
        default=1.0,
        help="Voice volume multiplier (default: 1.0)"
    )

    parser.add_argument(
        "--music-volume",
        type=float,
        default=0.3,
        help="Music volume multiplier (default: 0.3)"
    )

    parser.add_argument(
        "--fade-in",
        type=float,
        default=0.0,
        help="Music fade-in duration in seconds"
    )

    parser.add_argument(
        "--fade-out",
        type=float,
        default=0.0,
        help="Music fade-out duration in seconds"
    )

    args = parser.parse_args()

    # Validate files exist
    if not os.path.exists(args.voice):
        logger.error(f"❌ Voice file not found: {args.voice}")
        sys.exit(1)

    if not os.path.exists(args.music):
        logger.error(f"❌ Music file not found: {args.music}")
        sys.exit(1)

    # Mix audio
    try:
        import numpy as np  # Import here to avoid dependency if not using
        
        output_path = mix_audio(
            voice_path=args.voice,
            music_path=args.music,
            output_path=args.output,
            voice_volume=args.voice_volume,
            music_volume=args.music_volume,
            fade_in=args.fade_in,
            fade_out=args.fade_out
        )

        logger.info("🎉 Success! Your podcast/video audio is ready!")

    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()