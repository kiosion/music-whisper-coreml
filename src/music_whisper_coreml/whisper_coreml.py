import atexit
import base64
import json
from pathlib import Path
import subprocess
import time


class CaptionRuntimeError(RuntimeError):
    """A native inference failure that requires restarting the runtime."""


def validate_checkpoint(directory):
    """Reject decoding settings that the native caption runtime does not implement."""
    directory = Path(directory)
    settings = json.loads((directory / "generation_config.json").read_text())
    if not isinstance(settings, dict):
        raise ValueError("checkpoint generation settings must be an object")
    expected = {"decoder_start_token_id": 50258, "eos_token_id": 50257, "pad_token_id": 50257,
        "no_timestamps_token_id": 50363, "suppress_tokens": [], "begin_suppress_tokens": [220, 50257],
        "return_timestamps": False, "is_multilingual": True}
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint decoding settings are unsupported by the Core ML caption runtime")
    allowed = set(expected) | {"alignment_heads", "bos_token_id", "lang_to_id", "task_to_id",
        "max_initial_timestamp_index", "max_length", "prev_sot_token_id", "transformers_version"}
    for key in ("language", "task", "forced_decoder_ids"):
        if settings.get(key) is not None:
            raise ValueError("Core ML captions require automatic language detection without a task token")
        allowed.add(key)
    if set(settings) - allowed:
        raise ValueError("Core ML captions do not support generation setting: " + sorted(set(settings) - allowed)[0])
    config = json.loads((directory / "config.json").read_text())
    if not isinstance(config, dict) or config.get("vocab_size") != 51865:
        raise ValueError("Core ML captions require the 51865-token Whisper vocabulary")
    languages = settings.get("lang_to_id")
    if (not isinstance(languages, dict) or not languages
            or any(type(value) is not int or not 50259 <= value <= 50357 for value in languages.values())):
        raise ValueError("Core ML captions require valid Whisper language token IDs")
    if config.get("forced_decoder_ids") is not None:
        raise ValueError("Core ML captions do not support forced decoder tokens")


def validate_conversion(directory, checkpoint_digest, batch_size=1):
    """Validate converted artifacts against the selected checkpoint."""
    from .artifacts import artifact_digest

    directory = Path(directory)
    manifest = json.loads((directory / "caption-coreml.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("version") != 4 or manifest.get("runtime") != "coreml":
        raise ValueError("Core ML captions require a version 4 conversion; convert the checkpoint into a new directory")
    if type(manifest.get("batch_size")) is not int or not 1 <= manifest["batch_size"] <= 4:
        raise ValueError("Core ML conversion requires a batch size of 1-4")
    if manifest["batch_size"] != batch_size:
        raise ValueError("Core ML conversion batch size differs from --batch-size; select a matching conversion")
    if manifest.get("model_artifact_digest") != checkpoint_digest:
        raise ValueError("Core ML model was converted from a different checkpoint")
    if (manifest.get("precision"), manifest.get("max_target_positions"), manifest.get("num_mel_bins"),
            manifest.get("max_source_positions"), manifest.get("audio_cache_length")) != ("float16", 448, 80, 1500, 1504):
        raise ValueError("unsupported Core ML caption model dimensions or precision")
    if (type(manifest.get("decoder_layers")) is not int or not 1 <= manifest["decoder_layers"] <= 64
            or type(manifest.get("d_model")) is not int or not 1 <= manifest["d_model"] <= 4096):
        raise ValueError("unsupported Core ML caption decoder dimensions")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"AudioEncoder", "TextDecoder"}:
        raise ValueError("Core ML caption manifest requires encoder and decoder artifacts")
    for name, expected in artifacts.items():
        if artifact_digest(directory / (name + ".mlmodelc"), ("*",)) != expected:
            raise ValueError("Core ML caption artifact changed: " + name)
    return manifest


class CoreMLCaptions:
    """Generate captions through one persistent Core ML process."""

    def __init__(self, args, processor):
        self.processor = processor
        self.maximum = args.max_new_tokens
        self.batch_size = args.batch_size
        self.process = subprocess.Popen([str(args.coreml_runner), str(args.coreml_model), str(args.model), args.coreml_compute_units],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            if self._read() != {"ready": 3, "batchSize": self.batch_size, "computeUnits": args.coreml_compute_units}:
                raise ValueError("Core ML caption runner returned an unsupported handshake")
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _read(self):
        try:
            line = self.process.stdout.readline(1024 * 1024)
        except OSError as error:
            raise CaptionRuntimeError("Core ML caption runner response failed") from error
        if not line.endswith("\n"):
            raise CaptionRuntimeError("Core ML caption runner stopped or returned an invalid response")
        try:
            return json.loads(line)
        except json.JSONDecodeError as error:
            raise CaptionRuntimeError("Core ML caption runner returned invalid JSON") from error

    def __call__(self, waves):
        import numpy as np

        count = len(waves)
        if not 1 <= count <= self.batch_size:
            raise ValueError("Core ML caption batch exceeds the converted model capacity or is empty")
        started = time.monotonic()
        values = self.processor(waves, sampling_rate=16000, return_tensors="np", return_attention_mask=True)
        features = np.asarray(values.input_features, dtype="<f4")
        if features.shape != (count, 80, 3000) or not np.isfinite(features).all():
            raise ValueError("Core ML captions require finite 80-bin, 30-second features")
        prepared = time.monotonic()
        try:
            self.process.stdin.write(json.dumps({"features": base64.b64encode(features.tobytes()).decode("ascii"),
                "maxNewTokens": self.maximum, "batchSize": count}) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise CaptionRuntimeError("Core ML caption runner stopped") from error
        result = self._read()
        returned = time.monotonic()
        if not isinstance(result, dict):
            raise CaptionRuntimeError("Core ML caption runner returned an invalid result")
        rows = result.get("results")
        if not isinstance(rows, list) or len(rows) != count:
            raise CaptionRuntimeError("Core ML caption runner returned an unexpected batch size")
        timings = [result.get(key) for key in ("encodingSeconds", "generationSeconds")]
        decoding = [result.get(key) for key in ("predictionSeconds", "samplingSeconds")]
        if (any(type(value) not in (float, int) or not np.isfinite(value) or value < 0 for value in [*timings, *decoding])
                or sum(decoding) > timings[1] + 1e-6):
            raise CaptionRuntimeError("Core ML caption runner returned invalid timings")
        outputs = []
        for row in rows:
            if not isinstance(row, dict):
                raise CaptionRuntimeError("Core ML caption runner returned an invalid row")
            tokens, ended = row.get("tokens"), row.get("endedWithEOS")
            if (not isinstance(tokens, list) or not tokens or len(tokens) > self.maximum
                    or any(type(token) is not int or not 0 <= token < 51865 for token in tokens)
                    or type(ended) is not bool or ended != (tokens[-1] == 50257)
                    or not ended and len(tokens) != self.maximum or 50257 in tokens[:-1]):
                raise CaptionRuntimeError("Core ML caption runner returned invalid generated tokens")
            outputs.append({"caption": self.processor.decode(tokens, skip_special_tokens=True),
                "decoder_input_tokens": 3, "generated_tokens": len(tokens), "ended_with_eos": ended,
                "token_limit_reached": len(tokens) >= self.maximum and not ended,
                "finish_reason": "eos" if ended else "token_limit", "batch_size": count,
                "generation_timings_seconds": {"prediction": round(decoding[0] / count, 6), "sampling": round(decoding[1] / count, 6)}})
        shared = {"preprocessing": prepared - started, "audio_encoding": timings[0],
            "generation": timings[1], "runtime_transport": max(0, returned - prepared - sum(timings)),
            "text_decoding": time.monotonic() - returned}
        for output in outputs:
            output["timings_seconds"] = {key: value / count for key, value in shared.items()}
        return outputs

    def close(self):
        """Close the native process without leaving an inference worker running."""
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()


def load_backend(args):
    """Load the original feature processor and the converted caption runtime."""
    from transformers import AutoProcessor

    validate_checkpoint(args.model)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    if processor.feature_extractor.sampling_rate != 16000:
        raise ValueError("Core ML caption runtime requires 16 kHz audio")
    return 16000, CoreMLCaptions(args, processor)
