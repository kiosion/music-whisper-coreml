import argparse
import json
from pathlib import Path
import wave

import numpy as np

from music_whisper_coreml.artifacts import artifact_digest
from music_whisper_coreml.whisper_coreml import load_backend, validate_conversion


def main():
    """Print a caption and timing data for one PCM WAV excerpt."""
    parser = argparse.ArgumentParser(description="Caption one mono, 16 kHz, 16-bit PCM WAV excerpt of up to 30 seconds.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, required=True, help="original Music-Whisper checkpoint directory")
    parser.add_argument("--converted", dest="coreml_model", type=Path, required=True, help="version-4 conversion directory")
    parser.add_argument("--runner", dest="coreml_runner", type=Path, required=True, help="compiled native runner")
    parser.add_argument("--batch-size", type=int, choices=range(1, 5), default=1,
        help="converted model batch capacity (default: 1)")
    parser.add_argument("--compute-units", dest="coreml_compute_units", default="cpu-and-gpu",
        choices=("cpu-and-gpu", "cpu-and-neural-engine", "all", "cpu-only"),
        help="allowed Core ML processors (default: cpu-and-gpu)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="token limit per excerpt, 1-440 (default: 256)")
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 440:
        parser.error("max-new-tokens must be 1-440")
    try:
        with wave.open(str(args.input), "rb") as source:
            if (source.getnchannels(), source.getframerate(), source.getsampwidth()) != (1, 16000, 2):
                parser.error("input must be mono, 16 kHz, 16-bit PCM WAV; convert the excerpt before generation")
            if not 0 < source.getnframes() <= 30 * 16000:
                parser.error("input must contain 1-480000 frames; select an excerpt of at most 30 seconds")
            frames = source.readframes(source.getnframes())
            if len(frames) != source.getnframes() * 2:
                parser.error("input WAV is truncated; provide a complete excerpt")
            audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768
        args.model = args.model.resolve(strict=True)
        args.coreml_model = args.coreml_model.resolve(strict=True)
        args.coreml_runner = args.coreml_runner.resolve(strict=True)
        digest = artifact_digest(args.model, ("*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"))
        validate_conversion(args.coreml_model, digest, args.batch_size)
        _, infer = load_backend(args)
        try:
            print(json.dumps(infer([audio])[0], indent=2))
        finally:
            infer.close()
    except (OSError, ValueError, RuntimeError, wave.Error, EOFError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
