import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from music_whisper_coreml.artifacts import artifact_digest
from music_whisper_coreml.whisper_coreml import CaptionRuntimeError, CoreMLCaptions, validate_checkpoint, validate_conversion


class CoreMLCaptionTests(unittest.TestCase):
    def test_conversion_requires_matching_checkpoint_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifacts = {}
            for name in ("AudioEncoder", "TextDecoder"):
                path = directory / (name + ".mlmodelc")
                path.mkdir()
                (path / "weights.bin").write_bytes(name.encode())
                artifacts[name] = artifact_digest(path, ("*",))
            manifest = {"version": 4, "runtime": "coreml", "precision": "float16", "batch_size": 1,
                "max_target_positions": 448, "num_mel_bins": 80, "model_artifact_digest": "checkpoint",
                "max_source_positions": 1500, "audio_cache_length": 1504,
                "decoder_layers": 12, "d_model": 768, "artifacts": artifacts}
            (directory / "caption-coreml.json").write_text(json.dumps(manifest))
            self.assertEqual(validate_conversion(directory, "checkpoint"), manifest)
            with self.assertRaisesRegex(ValueError, "batch size differs"):
                validate_conversion(directory, "checkpoint", 2)
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                validate_conversion(directory, "other")
            (directory / "AudioEncoder.mlmodelc/weights.bin").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                validate_conversion(directory, "checkpoint")

    def test_stateless_conversion_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "caption-coreml.json").write_text('{"version": 1, "runtime": "whisperkit"}')
            with self.assertRaisesRegex(ValueError, "version 4 conversion"):
                validate_conversion(directory, "checkpoint")
            (directory / "caption-coreml.json").write_text('{"version": 3, "runtime": "coreml"}')
            with self.assertRaisesRegex(ValueError, "version 4 conversion"):
                validate_conversion(directory, "checkpoint")

    def test_caption_settings_reject_asr_prefix_and_decoding_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            settings = {"decoder_start_token_id": 50258, "eos_token_id": 50257, "pad_token_id": 50257,
                "no_timestamps_token_id": 50363, "suppress_tokens": [], "begin_suppress_tokens": [220, 50257],
                "return_timestamps": False, "is_multilingual": True, "lang_to_id": {"<|en|>": 50259}}
            (directory / "config.json").write_text('{"vocab_size": 51865}')
            target = directory / "generation_config.json"
            target.write_text(json.dumps(settings))
            validate_checkpoint(directory)
            for change in ({"task": "transcribe"}, {"suppress_tokens": [1]}, {"repetition_penalty": 1.2},
                    {"lang_to_id": {"<|en|>": 1}}):
                with self.subTest(change=change):
                    target.write_text(json.dumps({**settings, **change}))
                    with self.assertRaises(ValueError):
                        validate_checkpoint(directory)

    def infer_batch(self, result, count=1, capacity=1):
        result = {"predictionSeconds": 0.1, "samplingSeconds": 0.01, **result}
        instance = CoreMLCaptions.__new__(CoreMLCaptions)
        instance.maximum = 3
        instance.batch_size = capacity
        instance.process = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(json.dumps(result) + "\n"))

        class Processor:
            def __call__(self, waves, **kwargs):
                return SimpleNamespace(input_features=np.zeros((len(waves), 80, 3000), dtype=np.float32))

            def decode(self, tokens, **kwargs):
                return str(tokens)

        instance.processor = Processor()
        rows = instance([np.zeros(16000, dtype=np.float32) for _ in range(count)])
        request = json.loads(instance.process.stdin.getvalue())
        self.assertEqual(request["batchSize"], count)
        return rows

    def infer(self, result):
        row = {key: result.pop(key) for key in ("tokens", "endedWithEOS") if key in result}
        return self.infer_batch({**result, "results": [row]})[0]

    def test_token_limit_does_not_report_synthetic_eos(self):
        result = self.infer({"tokens": [1, 2, 3], "endedWithEOS": False,
            "encodingSeconds": 0.1, "generationSeconds": 0.2})
        self.assertEqual(result["generated_tokens"], 3)
        self.assertEqual(result["decoder_input_tokens"], 3)
        self.assertTrue(result["token_limit_reached"])
        self.assertFalse(result["ended_with_eos"])
        self.assertEqual(result["generation_timings_seconds"], {"prediction": 0.1, "sampling": 0.01})
        self.assertEqual(result["timings_seconds"]["generation"], 0.2)

    def test_eos_counts_as_generated_token(self):
        result = self.infer({"tokens": [1, 50257], "endedWithEOS": True,
            "encodingSeconds": 0.1, "generationSeconds": 0.2})
        self.assertEqual(result["generated_tokens"], 2)
        self.assertFalse(result["token_limit_reached"])
        self.assertTrue(result["ended_with_eos"])

    def test_batch_order_independent_endings_and_timing_allocation(self):
        rows = self.infer_batch({"results": [{"tokens": [7, 50257], "endedWithEOS": True},
            {"tokens": [8, 9, 10], "endedWithEOS": False}], "encodingSeconds": 0.1,
            "generationSeconds": 0.2}, count=2, capacity=2)
        self.assertEqual([row["caption"] for row in rows], ["[7, 50257]", "[8, 9, 10]"])
        self.assertEqual([row["finish_reason"] for row in rows], ["eos", "token_limit"])
        self.assertEqual([row["batch_size"] for row in rows], [2, 2])
        self.assertEqual(sum(row["timings_seconds"]["generation"] for row in rows), 0.2)
        self.assertEqual(sum(row["generation_timings_seconds"]["prediction"] for row in rows), 0.1)

    def test_singleton_final_batch_uses_actual_count(self):
        row = self.infer_batch({"results": [{"tokens": [7, 50257], "endedWithEOS": True}],
            "encodingSeconds": 0.1, "generationSeconds": 0.2}, capacity=4)[0]
        self.assertEqual(row["batch_size"], 1)
        self.assertEqual(row["timings_seconds"]["generation"], 0.2)

    def test_missing_extra_and_oversized_batch_rows_are_rejected(self):
        for rows in ([], [None], [{"tokens": [1, 50257], "endedWithEOS": True}] * 2):
            with self.subTest(rows=rows), self.assertRaises(CaptionRuntimeError):
                self.infer_batch({"results": rows, "encodingSeconds": 0.1, "generationSeconds": 0.2})
        for count in (0, 2):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "capacity or is empty"):
                self.infer_batch({}, count=count)

    def test_invalid_native_tokens_and_timings_are_rejected(self):
        valid = {"tokens": [1, 50257], "endedWithEOS": True,
            "encodingSeconds": 0.1, "generationSeconds": 0.2}
        for change in ({"tokens": [True]}, {"tokens": [1, 2]}, {"tokens": [50257, 1, 50257]},
                {"tokens": [51865]}, {"generationSeconds": float("nan")},
                {"predictionSeconds": float("nan")}, {"samplingSeconds": -1},
                {"predictionSeconds": 1},
                {"tokens": [1], "endedWithEOS": False}):
            with self.subTest(change=change):
                with self.assertRaises(CaptionRuntimeError):
                    self.infer({**valid, **change})

    def test_failed_handshake_closes_runner(self):
        args = SimpleNamespace(coreml_runner="runner", coreml_model="converted", model="model", max_new_tokens=256,
            batch_size=1, coreml_compute_units="cpu-and-gpu")
        with patch("music_whisper_coreml.whisper_coreml.subprocess.Popen") as process:
            process.return_value.stdin.closed = False
            process.return_value.stdout.readline.return_value = ""
            with self.assertRaises(RuntimeError):
                CoreMLCaptions(args, None)
            process.return_value.stdin.close.assert_called_once()
            process.return_value.wait.assert_called_once_with(timeout=5)

    def test_stateless_runner_handshake_is_rejected(self):
        args = SimpleNamespace(coreml_runner="runner", coreml_model="converted", model="model", max_new_tokens=256,
            batch_size=1, coreml_compute_units="cpu-and-gpu")
        with patch("music_whisper_coreml.whisper_coreml.subprocess.Popen") as process:
            process.return_value.stdin.closed = False
            process.return_value.stdout.readline.return_value = '{"ready": 1}\n'
            with self.assertRaisesRegex(ValueError, "unsupported handshake"):
                CoreMLCaptions(args, None)
            process.return_value.wait.assert_called_once_with(timeout=5)

    def test_runner_handshake_matches_compute_policy_and_batch(self):
        args = SimpleNamespace(coreml_runner="runner", coreml_model="converted", model="model", max_new_tokens=256,
            batch_size=2, coreml_compute_units="all")
        expected = {"ready": 3, "batchSize": 2, "computeUnits": "all"}
        for response in (expected, {**expected, "batchSize": 1}, {**expected, "computeUnits": "cpu-only"}):
            with self.subTest(response=response), patch("music_whisper_coreml.whisper_coreml.subprocess.Popen") as process:
                process.return_value.stdin.closed = False
                process.return_value.stdout.readline.return_value = json.dumps(response) + "\n"
                if response == expected:
                    instance = CoreMLCaptions(args, None)
                    instance.close()
                else:
                    with self.assertRaisesRegex(ValueError, "unsupported handshake"):
                        CoreMLCaptions(args, None)
                self.assertEqual(process.call_args.args[0], ["runner", "converted", "model", "all"])

    def test_broken_pipe_during_close_still_reaps_runner(self):
        instance = CoreMLCaptions.__new__(CoreMLCaptions)
        with patch("music_whisper_coreml.whisper_coreml.subprocess.Popen") as process:
            instance.process = process.return_value
            instance.process.stdin.closed = False
            instance.process.stdin.close.side_effect = BrokenPipeError()
            instance.close()
            instance.process.wait.assert_called_once_with(timeout=5)
            instance.process.stdout.close.assert_called_once()

    def test_response_read_failure_is_fatal(self):
        instance = CoreMLCaptions.__new__(CoreMLCaptions)
        with patch("music_whisper_coreml.whisper_coreml.subprocess.Popen") as process:
            instance.process = process.return_value
            instance.process.stdout.readline.side_effect = OSError("pipe closed")
            with self.assertRaises(CaptionRuntimeError):
                instance._read()
