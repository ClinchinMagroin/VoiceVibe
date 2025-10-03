# 🎙️ VibeVoice Complete Audio Production Suite

**Voice Cloning + AI Music Generation + Audio Mixing - All in One!**

---

## 🚀 Quick Start (One Command!)

```bash
./setup.sh
```

This will:
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Download tokenizer
- ✅ Let you choose and download a model
- ✅ Create all necessary folders
- ✅ Set up helper scripts

**Then activate and test:**
```bash
source activate_vv.sh
./quick_test.sh
```

---

## 🎯 What's Included

### 1️⃣ **Voice Cloning** (`vibevoice_standalone_mp3.py`)
Generate speech with voice cloning capabilities!

```bash
# Basic speech
python3 vibevoice_standalone_mp3.py --text "Hello world!" --output speech.mp3

# Voice cloning
python3 vibevoice_standalone_mp3.py \
    --text "Clone my voice!" \
    --voice-sample voice_samples/my_voice.mp3 \
    --output cloned.mp3
```

**Features:**
- ✅ MP3 input/output support
- ✅ Voice cloning from samples
- ✅ Multiple format support (.wav, .mp3, .flac, .ogg)
- ✅ Adjustable quality (diffusion steps)
- ✅ 4 model options (5.4GB to 18.7GB)

### 2️⃣ **AI Music Generator** (`music_generator.py`)
Generate background music from text descriptions!

```bash
# Generate upbeat music
python3 music_generator.py \
    --text "upbeat electronic dance music with synths" \
    --duration 15 \
    --output music.mp3

# Calm ambient music
python3 music_generator.py \
    --text "calm ambient atmospheric music with soft piano" \
    --duration 30 \
    --output calm_music.mp3
```

**Features:**
- ✅ Text-to-music generation
- ✅ Multiple model sizes (small/medium/large/melody)
- ✅ Adjustable duration (up to 30s)
- ✅ MP3/WAV output

### 3️⃣ **Audio Mixer** (`mix_audio.py`)
Combine voice and music professionally!

```bash
# Mix voice with background music
python3 mix_audio.py \
    --voice speech.mp3 \
    --music background.mp3 \
    --output final.mp3 \
    --music-volume 0.3
```

**Features:**
- ✅ Mix voice + music
- ✅ Volume control
- ✅ Fade in/out support
- ✅ MP3/WAV support

### 4️⃣ **Complete Workflow** (`create_podcast.sh`)
Generate voice + music in ONE command!

```bash
# Complete podcast/video audio in one shot
./create_podcast.sh \
    --text "Welcome to my show!" \
    --music-desc "upbeat intro music" \
    --output podcast_episode_1.mp3
```

---

## 📦 Available Models

### Voice Models (VibeVoice)
| Model | Size | Best For | Download Command |
|-------|------|----------|------------------|
| **VibeVoice-Large-Q8** | 11.6GB | **Balanced** ⭐ | Auto via setup.sh |
| VibeVoice-1.5B | 5.4GB | Speed | Auto via setup.sh |
| VibeVoice-Large | 18.7GB | Quality | Auto via setup.sh |
| VibeVoice-Large-Q4 | 6.6GB | Low VRAM | Auto via setup.sh |

### Music Models (MusicGen)
| Model | Size | Best For |
|-------|------|----------|
| small | 1.5GB | Fast generation |
| **medium** | 3.5GB | **Balanced** ⭐ |
| large | 6GB | Best quality |
| melody | 3.5GB | Melody conditioning |

---

## 🎬 Complete Production Examples

### Example 1: Podcast Episode
```bash
# Voice with background music
./create_podcast.sh \
    --text-file scripts/episode1.txt \
    --voice-sample voice_samples/host.mp3 \
    --music-desc "calm podcast intro music" \
    --output podcast/episode1.mp3
```

### Example 2: Video Narration
```bash
# Step 1: Generate narration
python3 vibevoice_standalone_mp3.py \
    --text-file video_script.txt \
    --voice-sample narrator.mp3 \
    --output narration.mp3

# Step 2: Generate cinematic music
python3 music_generator.py \
    --text "epic cinematic orchestral music" \
    --duration 60 \
    --output cinematic.mp3

# Step 3: Mix
python3 mix_audio.py \
    --voice narration.mp3 \
    --music cinematic.mp3 \
    --output video_audio.mp3 \
    --music-volume 0.2 \
    --fade-in 2 \
    --fade-out 3
```

### Example 3: Audiobook Chapter
```bash
# Generate chapter with voice cloning
python3 vibevoice_standalone_mp3.py \
    --text-file book/chapter1.txt \
    --voice-sample narrator_voice.mp3 \
    --diffusion-steps 25 \
    --output audiobook_ch1.mp3
```

---

## ⚙️ Command Line Reference

### Voice Generation Options
```bash
python3 vibevoice_standalone_mp3.py \
    --text "Text to speak" \
    --voice-sample voice.mp3 \
    --output result.mp3 \
    --diffusion-steps 20 \
    --cfg-scale 1.3 \
    --seed 42
```

### Music Generation Options
```bash
python3 music_generator.py \
    --text "music description" \
    --duration 15 \
    --output music.mp3 \
    --model medium \
    --guidance-scale 3.0
```

### Mixing Options
```bash
python3 mix_audio.py \
    --voice speech.mp3 \
    --music background.mp3 \
    --output final.mp3 \
    --voice-volume 1.0 \
    --music-volume 0.3 \
    --fade-in 1 \
    --fade-out 2
```

---

## 📁 Project Structure

```
/workspace/voicevibe/
├── setup.sh ⭐                    # One-click installer
├── vibevoice_standalone_mp3.py   # Voice cloning
├── music_generator.py            # Music generation
├── mix_audio.py                  # Audio mixer
├── create_podcast.sh             # Complete workflow
├── activate_vv.sh                # Environment activator
├── quick_test.sh                 # Test installation
│
├── models/vibevoice/
│   ├── VibeVoice-Large-Q8/      # Voice model
│   └── tokenizer/                # Qwen tokenizer
│
├── vvembed/                      # Embedded VibeVoice code
│
├── voice_samples/                # Your voice files
├── outputs/                      # Generated audio
├── scripts/                      # Text scripts
└── vv_env/                       # Python environment
```

---

## 🎨 Supported Audio Formats

### Input Formats:
- `.mp3` - MP3 audio
- `.wav` - WAV audio
- `.flac` - FLAC audio
- `.ogg` - OGG Vorbis audio

### Output Formats:
- `.mp3` - MP3 (compressed, smaller file)
- `.wav` - WAV (lossless, larger file)

---

## ⚡ Performance Guide

### Voice Generation
| Diffusion Steps | Time (10s audio) | Quality |
|----------------|------------------|---------|
| 5 | ~10s | Good |
| 10 | ~15s | Very Good |
| 20 | ~30s | Excellent ⭐ |
| 30 | ~45s | Perfect |

### Music Generation
| Model | Time (10s music) | VRAM |
|-------|------------------|------|
| small | ~5s | 2GB |
| medium | ~10s | 4GB ⭐ |
| large | ~20s | 8GB |

---

## 🎯 Production Workflow

### Professional Podcast Production:

1. **Write script** → `scripts/episode.txt`

2. **Generate voice:**
```bash
python3 vibevoice_standalone_mp3.py \
    --text-file scripts/episode.txt \
    --voice-sample voice_samples/host.mp3 \
    --output voice.mp3
```

3. **Generate music:**
```bash
python3 music_generator.py \
    --text "upbeat podcast intro music" \
    --duration 15 \
    --output music.mp3
```

4. **Mix together:**
```bash
python3 mix_audio.py \
    --voice voice.mp3 \
    --music music.mp3 \
    --output episode_final.mp3 \
    --music-volume 0.25
```

**OR use the one-liner:**
```bash
./create_podcast.sh \
    --text-file scripts/episode.txt \
    --voice-sample voice_samples/host.mp3 \
    --music-desc "upbeat podcast intro" \
    --output episode_final.mp3
```

---

## 🐛 Troubleshooting

### "CUDA out of memory"
- Use smaller models
- Reduce diffusion steps to 5-10
- Generate shorter music (10-15s)

### "ffmpeg not found" (for MP3 export)
```bash
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS
```

### Voice cloning quality issues
- Use 30-60 second voice samples
- Ensure clean audio (no background noise)
- Increase diffusion steps (20-30)

### Music generation too slow
- Use `--model small` instead of medium
- Reduce duration
- Use CPU for quick tests

---

## 🎉 What Can You Create?

✨ **Podcasts** - Voice + background music
✨ **Audiobooks** - Professional narration
✨ **Video Narration** - YouTube, TikTok, etc.
✨ **Voice Assistants** - Custom AI voices
✨ **Educational Content** - Lessons with music
✨ **Audio Ads** - Commercial production
✨ **Gaming** - Character voices + soundtracks
✨ **Radio Shows** - Complete audio production

---

## 🎤 Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- 10GB disk space

**Recommended:**
- NVIDIA GPU (12GB+ VRAM)
- 32GB RAM
- 50GB disk space (for multiple models)

**Supported Platforms:**
- ✅ Linux (Ubuntu, Debian, etc.)
- ✅ Windows (WSL2 or native)
- ✅ macOS (Intel & Apple Silicon)

---

## 📚 Credits

- **VibeVoice:** Microsoft Research
- **MusicGen:** Meta AI Research
- **Implementation:** Complete standalone version with all features

---

## 🚀 Ready to Create!

1. Run `./setup.sh`
2. Activate environment: `source activate_vv.sh`
3. Start creating: `./create_podcast.sh --text "Your content!"`

**Happy creating! 🎙️🎵✨**