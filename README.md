# Music-Whisper Core ML

Convert [Music-Whisper](https://huggingface.co/laion/music-whisper) to FP16 Core
ML and generate music captions through a persistent Swift process. Uses
[WhisperKit tools](https://github.com/argmaxinc/whisperkittools) for conversion.
Requires Apple silicon, macOS 15+, Xcode and Python 3.12.

## Setup

Run from the repo root. Conversion and inference use separate environments
since their tested dependency versions differ.

```sh
python3.12 -m venv .venv-conversion
.venv-conversion/bin/python -m pip install '.[conversion]'
python3.12 -m venv .venv-runtime
.venv-runtime/bin/python -m pip install '.[runtime]'

.venv-conversion/bin/hf download laion/music-whisper \
  --revision f77998061ace193e308f0ecac9dfa844c60e2030 \
  --local-dir artifacts/checkpoint \
  --include '*.json' '*.safetensors' '*.model' '*.txt' '*.jinja'

.venv-conversion/bin/python -m music_whisper_coreml.whisper_conversion \
  --model artifacts/checkpoint --output-dir artifacts/coreml \
  --batch-size 2 --threads 2
swift build --package-path swift --scratch-path build \
  -c release --product analysis-captions-coreml -j 2
```

The supplied conversion dir must be new. It contains compiled encoder/decoder
models and `caption-coreml.json`. Model weights/build outputs are ignored.

## Generate a caption

Provide mono, 16 kHz, 16-bit PCM WAV, 30s at most.

```sh
mkdir -p generations
.venv-runtime/bin/python examples/caption_wav.py /path/to/excerpt.wav \
  --model artifacts/checkpoint --converted artifacts/coreml \
  --runner build/release/analysis-captions-coreml --batch-size 2 \
  --compute-units cpu-and-gpu > generations/caption.json
```

The JSON includes caption text, token counts, completion reason and timings.
The Python adapter also accepts batches of waveforms. Batch capacity is fixed
at conversion time (1-4); the runtime must use the same value.

This runtime is experimental.
- FP16 output may differ versus Torch
- Better throughput isn't guaranteed
- Initial model loading may take longer
- Captions may be inaccurate or truncated at/close to the token limit

## Tests

```sh
.venv-conversion/bin/python -m unittest discover -s tests -v
swift test --package-path swift --scratch-path build -j 2
```

Tests use synthetic inputs and require no downloaded checkpoint or music files.
Source licensed under [AGPLv3](LICENSE). Dependencies and model weights retain
their upstream licenses.
