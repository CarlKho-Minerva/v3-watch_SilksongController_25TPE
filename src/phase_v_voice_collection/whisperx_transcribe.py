"""
WhisperX Transcription Script for Research-Grade Word Segmentation

This script uses WhisperX for word-specific research-grade word segmentation with:
- High-accuracy ASR models (Whisper large-v2 or large-v3)
- Forced alignment using wav2vec2 for stable per-word timings
- VAD (Voice Activity Detection) to trim silence and reduce boundary jitter
- Optional diarization for speaker segmentation
- Word-level timestamps with confidence scores
- JSON output format for programmatic analysis

WhisperX provides more stable per-word timings than Large-V3-Turbo's internal
alignment, especially on fast speech, code-switching, and noisy clips.

Usage:
    # Basic usage with defaults (large-v3 model)
    python whisperx_transcribe.py --audio session_01.wav

    # Specify model and enable diarization
    python whisperx_transcribe.py --audio session_01.wav --model large-v2 --diarize

    # Custom output location
    python whisperx_transcribe.py --audio session_01.wav --output results/

    # Use existing Whisper model from local path
    python whisperx_transcribe.py --audio session_01.wav --model-dir /path/to/model

For reproducible timing for studies or quantitative analysis, WhisperX's forced
alignment is the safer choice over real-time transcription methods.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import warnings

try:
    import whisperx
    import torch
except ImportError:
    print("ERROR: WhisperX is not installed!")
    print("Install with: pip install whisperx")
    print("See: https://github.com/m-bain/whisperX")
    sys.exit(1)

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("WARNING: librosa and soundfile not installed.")
    print("Install with: pip install librosa soundfile")
    print("Audio preprocessing will be limited.")
    librosa = None
    sf = None

# ANSI Color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def check_device():
    """Check available compute device"""
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        print(f"{Colors.GREEN}✓ Using CUDA GPU acceleration{Colors.RESET}")
    elif torch.backends.mps.is_available():
        device = "mps"
        compute_type = "float32"  # MPS doesn't support float16
        print(f"{Colors.GREEN}✓ Using Apple Silicon (MPS) acceleration{Colors.RESET}")
    else:
        device = "cpu"
        compute_type = "float32"
        print(f"{Colors.YELLOW}⚠ Using CPU (this will be slow){Colors.RESET}")

    return device, compute_type


def preprocess_audio(audio_path, output_path=None, target_sr=16000, normalize=True, vad=True):
    """
    Preprocess audio to optimize for WhisperX:
    - Normalize to 16 kHz mono WAV
    - Apply VAD to trim silence (optional)
    - Normalize volume (optional)

    Args:
        audio_path: Path to input audio file
        output_path: Path for preprocessed audio (None = use temp file)
        target_sr: Target sample rate (default: 16000 Hz)
        normalize: Apply volume normalization
        vad: Apply voice activity detection to trim silence

    Returns:
        Path to preprocessed audio file
    """
    if librosa is None or sf is None:
        print(f"{Colors.YELLOW}⚠ Audio preprocessing skipped (librosa/soundfile not installed){Colors.RESET}")
        return audio_path

    print(f"{Colors.CYAN}🎵 Preprocessing audio...{Colors.RESET}")

    # Load audio
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    print(f"  • Loaded audio: {len(audio)/sr:.2f}s @ {sr} Hz")

    # Apply VAD if requested
    if vad:
        # Simple energy-based VAD
        frame_length = 2048
        hop_length = 512

        # Calculate energy
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

        # Threshold: mean energy
        threshold = energy.mean() * 0.1

        # Find voice frames
        voice_frames = energy > threshold

        # Expand to sample indices
        voice_mask = librosa.util.frame(
            audio,
            frame_length=frame_length,
            hop_length=hop_length
        ).T
        voice_mask = voice_mask[voice_frames].flatten()

        # Only keep voiced segments if we detected any
        if len(voice_mask) > sr * 0.5:  # At least 0.5s of voice
            original_len = len(audio)
            # Simpler approach: just trim silence from start and end
            audio = librosa.effects.trim(audio, top_db=20)[0]
            trimmed = original_len - len(audio)
            print(f"  • VAD trimmed: {trimmed/sr:.2f}s of silence")

    # Normalize volume
    if normalize:
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95  # Normalize to -0.5 dB
            print(f"  • Normalized volume")

    # Save preprocessed audio
    if output_path is None:
        # Create temp file in same directory as input
        input_dir = Path(audio_path).parent
        output_path = input_dir / f"preprocessed_{Path(audio_path).name}"

    sf.write(output_path, audio, sr)
    print(f"{Colors.GREEN}✓ Preprocessed audio saved: {output_path}{Colors.RESET}")

    return str(output_path)


def transcribe_with_whisperx(
    audio_path,
    model_name="large-v3",
    model_dir=None,
    device=None,
    compute_type=None,
    batch_size=16,
    language=None
):
    """
    Transcribe audio using WhisperX

    Args:
        audio_path: Path to audio file
        model_name: Whisper model to use (large-v3, large-v2, etc.)
        model_dir: Optional custom model directory
        device: Compute device (cuda/cpu/mps)
        compute_type: Compute type (float16/float32)
        batch_size: Batch size for inference
        language: Language code (None = auto-detect)

    Returns:
        Transcription result with segments
    """
    if device is None or compute_type is None:
        device, compute_type = check_device()

    print(f"\n{Colors.CYAN}🤖 Loading WhisperX model...{Colors.RESET}")
    print(f"  • Model: {model_name}")
    print(f"  • Device: {device}")
    print(f"  • Compute type: {compute_type}")

    # Load model
    if model_dir:
        print(f"  • Model directory: {model_dir}")
        # Note: WhisperX doesn't directly support custom model directories
        # This is a placeholder for potential future support
        warnings.warn("Custom model directory support is limited in WhisperX")

    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=language
    )

    print(f"{Colors.GREEN}✓ Model loaded{Colors.RESET}")

    # Custom prompt for gesture command vocabulary
    initial_prompt = (
        "The following is a transcription of a person playing the video game Hollow Knight: Silksong. "
        "They are speaking their character's actions out loud. "
        "The key commands are: jump, punch, attack, turn, walk, walking, walk start, idle, rest, stop, noise. "
        "The speaker might say phrases like 'I'm gonna jump here', 'punch punch', 'let me walk over there', "
        "'okay, now idle', or 'that was noise'."
    )

    # Transcribe
    print(f"\n{Colors.CYAN}📝 Transcribing audio...{Colors.RESET}")
    print(f"  • Using custom prompt for gesture vocabulary")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size, initial_prompt=initial_prompt)

    print(f"{Colors.GREEN}✓ Transcription complete{Colors.RESET}")
    print(f"  • Detected language: {result.get('language', 'unknown')}")
    print(f"  • Segments: {len(result.get('segments', []))}")

    return result, audio


def align_whisperx(
    audio,
    transcription_result,
    device=None,
    language_code=None
):
    """
    Apply forced alignment to get word-level timestamps

    Args:
        audio: Audio array from whisperx.load_audio()
        transcription_result: Result from model.transcribe()
        device: Compute device
        language_code: Language code for alignment model

    Returns:
        Aligned result with word-level timestamps
    """
    if device is None:
        device, _ = check_device()

    if language_code is None:
        language_code = transcription_result.get('language', 'en')

    print(f"\n{Colors.CYAN}🎯 Applying forced alignment...{Colors.RESET}")
    print(f"  • Language: {language_code}")

    # Load alignment model
    model_a, metadata = whisperx.load_align_model(
        language_code=language_code,
        device=device
    )

    # Align
    result = whisperx.align(
        transcription_result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )

    print(f"{Colors.GREEN}✓ Alignment complete{Colors.RESET}")

    # Count words
    word_count = sum(len(seg.get('words', [])) for seg in result.get('segments', []))
    print(f"  • Word-level timestamps: {word_count} words")

    return result


def diarize_audio(audio_path, device=None, hf_token=None):
    """
    Apply speaker diarization

    Args:
        audio_path: Path to audio file
        device: Compute device
        hf_token: HuggingFace token for pyannote models

    Returns:
        Diarization result
    """
    if device is None:
        device, _ = check_device()

    print(f"\n{Colors.CYAN}👥 Applying speaker diarization...{Colors.RESET}")

    if hf_token is None:
        print(f"{Colors.YELLOW}⚠ No HuggingFace token provided{Colors.RESET}")
        print(f"  Diarization requires a HF token for pyannote models")
        print(f"  Get one at: https://huggingface.co/settings/tokens")
        print(f"  Pass with: --hf-token YOUR_TOKEN")
        return None

    # Load diarization pipeline
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hf_token,
        device=device
    )

    # Diarize
    diarize_segments = diarize_model(audio_path)

    print(f"{Colors.GREEN}✓ Diarization complete{Colors.RESET}")
    print(f"  • Speakers detected: {len(set(seg['speaker'] for seg in diarize_segments))}")

    return diarize_segments


def assign_speakers_to_words(aligned_result, diarize_segments):
    """
    Assign speaker labels to words based on diarization

    Args:
        aligned_result: Result from whisperx.align()
        diarize_segments: Result from diarization

    Returns:
        Result with speaker labels
    """
    if diarize_segments is None:
        return aligned_result

    print(f"\n{Colors.CYAN}🏷️  Assigning speakers to words...{Colors.RESET}")

    result = whisperx.assign_word_speakers(diarize_segments, aligned_result)

    print(f"{Colors.GREEN}✓ Speaker assignment complete{Colors.RESET}")

    return result


def save_results(result, output_path, format='json'):
    """
    Save transcription results

    Args:
        result: WhisperX result with word-level timestamps
        output_path: Output file path
        format: Output format (json, txt, srt)
    """
    print(f"\n{Colors.CYAN}💾 Saving results...{Colors.RESET}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        # Full detailed JSON output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"{Colors.GREEN}✓ Saved JSON: {output_path}{Colors.RESET}")

    elif format == 'txt':
        # Simple text transcript
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            for segment in result.get('segments', []):
                f.write(segment['text'] + '\n')
        print(f"{Colors.GREEN}✓ Saved TXT: {txt_path}{Colors.RESET}")

    elif format == 'srt':
        # SubRip subtitle format
        srt_path = output_path.with_suffix('.srt')
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result.get('segments', []), 1):
                start = segment['start']
                end = segment['end']
                text = segment['text'].strip()

                # Convert to SRT time format
                start_time = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d}"
                end_time = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}"

                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
        print(f"{Colors.GREEN}✓ Saved SRT: {srt_path}{Colors.RESET}")

    # Always save a summary
    summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("WhisperX Transcription Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Language: {result.get('language', 'unknown')}\n")
        f.write(f"Segments: {len(result.get('segments', []))}\n")

        # Count words
        word_count = sum(len(seg.get('words', [])) for seg in result.get('segments', []))
        f.write(f"Words: {word_count}\n\n")

        # Word statistics
        if word_count > 0:
            f.write("Word-Level Timestamp Statistics:\n")
            f.write("-" * 70 + "\n")

            all_words = []
            for seg in result.get('segments', []):
                all_words.extend(seg.get('words', []))

            # Show first 50 words with timestamps and confidence
            f.write("\nFirst 50 words:\n")
            for i, word_info in enumerate(all_words[:50], 1):
                word = word_info.get('word', '')
                start = word_info.get('start', 0)
                end = word_info.get('end', 0)
                score = word_info.get('score', 0)
                speaker = word_info.get('speaker', '')

                speaker_str = f" [Speaker: {speaker}]" if speaker else ""
                f.write(f"{i:3d}. {start:7.2f}s-{end:7.2f}s | {word:20s} | conf: {score:.3f}{speaker_str}\n")

            if len(all_words) > 50:
                f.write(f"\n... and {len(all_words) - 50} more words\n")

    print(f"{Colors.GREEN}✓ Saved summary: {summary_path}{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description='WhisperX transcription for research-grade word segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription with large-v3 model
  python whisperx_transcribe.py --audio session_01.wav

  # Use large-v2 model with diarization
  python whisperx_transcribe.py --audio session_01.wav --model large-v2 --diarize --hf-token TOKEN

  # Specify language and custom output
  python whisperx_transcribe.py --audio session_01.wav --language en --output results/transcript.json

  # Skip preprocessing
  python whisperx_transcribe.py --audio session_01.wav --no-preprocess

For research use, recommended setup:
  - Use large-v2 or large-v3 model for high accuracy
  - Enable preprocessing for audio normalization and VAD
  - Save as JSON for programmatic analysis
  - Add diarization if speaker segmentation is needed
        """
    )

    # Required arguments
    parser.add_argument(
        '--audio',
        required=True,
        help='Path to audio file to transcribe'
    )

    # Model arguments
    parser.add_argument(
        '--model',
        default='large-v3',
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
        help='Whisper model to use (default: large-v3)'
    )

    parser.add_argument(
        '--model-dir',
        help='Optional custom model directory'
    )

    parser.add_argument(
        '--language',
        help='Language code (e.g., en, es, fr). If not specified, will auto-detect.'
    )

    # Output arguments
    parser.add_argument(
        '--output',
        help='Output path for results (default: input_dir/input_name_whisperx.json)'
    )

    parser.add_argument(
        '--format',
        default='json',
        choices=['json', 'txt', 'srt', 'all'],
        help='Output format (default: json)'
    )

    # Processing arguments
    parser.add_argument(
        '--no-preprocess',
        action='store_true',
        help='Skip audio preprocessing (normalization, VAD)'
    )

    parser.add_argument(
        '--no-alignment',
        action='store_true',
        help='Skip forced alignment (faster but no word-level timestamps)'
    )

    parser.add_argument(
        '--diarize',
        action='store_true',
        help='Enable speaker diarization'
    )

    parser.add_argument(
        '--hf-token',
        help='HuggingFace token for diarization models'
    )

    # Performance arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for inference (default: 16)'
    )

    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu', 'mps'],
        help='Compute device (auto-detected if not specified)'
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.CYAN}WhisperX Research-Grade Word Segmentation{Colors.RESET}")
    print("=" * 70 + "\n")

    # Check if audio file exists
    if not os.path.exists(args.audio):
        print(f"{Colors.RED}ERROR: Audio file not found: {args.audio}{Colors.RESET}")
        sys.exit(1)

    print(f"Input audio: {args.audio}")
    print(f"Model: {args.model}")
    print(f"Language: {args.language or 'auto-detect'}")
    print()

    # Determine device
    device, compute_type = check_device()
    if args.device:
        device = args.device

    # Preprocess audio
    audio_path = args.audio
    if not args.no_preprocess:
        try:
            audio_path = preprocess_audio(
                args.audio,
                normalize=True,
                vad=True
            )
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ Preprocessing failed: {e}{Colors.RESET}")
            print(f"  Continuing with original audio file")
            audio_path = args.audio

    # Transcribe
    try:
        result, audio = transcribe_with_whsperx(
            audio_path,
            model_name=args.model,
            model_dir=args.model_dir,
            device=device,
            compute_type=compute_type,
            batch_size=args.batch_size,
            language=args.language
        )
    except Exception as e:
        print(f"\n{Colors.RED}ERROR during transcription: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Apply forced alignment
    if not args.no_alignment:
        try:
            result = align_whisperx(
                audio,
                result,
                device=device,
                language_code=args.language
            )
        except Exception as e:
            print(f"\n{Colors.YELLOW}⚠ Alignment failed: {e}{Colors.RESET}")
            print(f"  Continuing without word-level timestamps")

    # Apply diarization
    if args.diarize:
        try:
            diarize_segments = diarize_audio(
                audio_path,
                device=device,
                hf_token=args.hf_token
            )

            if diarize_segments:
                result = assign_speakers_to_words(result, diarize_segments)
        except Exception as e:
            print(f"\n{Colors.YELLOW}⚠ Diarization failed: {e}{Colors.RESET}")
            print(f"  Continuing without speaker labels")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        input_path = Path(args.audio)
        output_path = input_path.parent / f"{input_path.stem}_whisperx.json"

    # Save results
    try:
        if args.format == 'all':
            for fmt in ['json', 'txt', 'srt']:
                save_results(result, output_path, format=fmt)
        else:
            save_results(result, output_path, format=args.format)
    except Exception as e:
        print(f"\n{Colors.RED}ERROR saving results: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.GREEN}✅ WhisperX transcription complete!{Colors.RESET}")
    print("=" * 70)
    print(f"\nOutput: {output_path}")
    print(f"\nFor research analysis:")
    print(f"  • Word-level timestamps with confidence scores")
    print(f"  • Forced alignment for stable timing")
    print(f"  • JSON format for programmatic processing")
    print()


if __name__ == "__main__":
    try:
        import numpy as np
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠ Interrupted by user{Colors.RESET}")
        sys.exit(1)
