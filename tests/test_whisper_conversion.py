from copy import deepcopy
import importlib.util
import sys
import unittest


@unittest.skipUnless(importlib.util.find_spec("whisperkit"), "requires the Core ML conversion environment")
class StatefulConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import WhisperConfig, WhisperForConditionalGeneration
        from whisperkit import audio_encoder, text_decoder
        from music_whisper_coreml.whisper_conversion import ProjectedAudioEncoder, StatefulTextDecoder

        torch.set_num_threads(1)
        torch.manual_seed(41)
        cls.torch = torch
        cls.config = WhisperConfig(d_model=16, encoder_layers=2, decoder_layers=2,
            encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
            vocab_size=64, max_source_positions=8, max_target_positions=12,
            pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=3)
        hf = WhisperForConditionalGeneration(cls.config).eval()
        encoder = audio_encoder.WhisperAudioEncoder(cls.config).eval()
        encoder.load_state_dict(hf.model.encoder.state_dict())
        decoder = text_decoder.WhisperTextDecoder(cls.config).eval()
        decoder.load_state_dict(hf.model.decoder.state_dict())
        cls.reference_encoder = deepcopy(encoder)
        cls.reference_decoder = deepcopy(decoder)
        cls.encoder = ProjectedAudioEncoder(encoder, decoder).eval()
        cls.decoder = StatefulTextDecoder(decoder).eval()

    def prepare(self, features):
        projections = self.encoder(features)
        for name, buffer in self.decoder.named_buffers():
            buffer.zero_()
        for index in range(self.config.decoder_layers):
            for offset, kind in enumerate(("key", "value")):
                getattr(self.decoder, f"audio_{kind}_{index}").copy_(projections[index * 2 + offset])

    def sequence(self, features):
        torch = self.torch
        self.prepare(features)
        encoded = self.reference_encoder(features)
        keys = torch.zeros(1, self.config.d_model * self.config.decoder_layers, 1, 12)
        values = torch.zeros_like(keys)
        mask = torch.full((1, 12), -10000.0)
        result = []
        for position, token in enumerate((3, 4, 8, 6, 7, 14, 23, 17)):
            mask[:, position] = 0
            update = torch.zeros_like(mask)
            update[:, position] = 1
            ids = torch.tensor([token])
            length = torch.tensor([position])
            expected, key, value = self.reference_decoder(ids, length, keys, values, update, encoded, mask)
            actual = self.decoder(ids, length, mask, torch.ones(1, dtype=torch.int64))
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            keys[:, :, :, position:position + 1] = key
            values[:, :, :, position:position + 1] = value
            for index in range(self.config.decoder_layers):
                torch.testing.assert_close(getattr(self.decoder, f"self_key_{index}"), keys[:, index * 16:(index + 1) * 16])
                torch.testing.assert_close(getattr(self.decoder, f"self_value_{index}"), values[:, index * 16:(index + 1) * 16])
            result.append(actual.clone())
        return result

    def test_logits_cache_and_excerpt_reset_match_stateless_decoder(self):
        torch = self.torch
        with torch.inference_mode():
            first = torch.randn(1, 80, 1, 16)
            second = torch.randn_like(first)
            before = self.sequence(first)
            different = self.sequence(second)
            after = self.sequence(first)
            for left, right in zip(before, after):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertFalse(torch.equal(before[0], different[0]))

    def test_cross_projections_are_computed_once_per_excerpt(self):
        torch = self.torch
        calls = []
        handles = [module.register_forward_hook(lambda *args: calls.append(1))
            for module in (*self.encoder.keys, *self.encoder.values)]
        try:
            with torch.inference_mode():
                self.prepare(torch.randn(1, 80, 1, 16))
                for name, buffer in self.decoder.named_buffers():
                    if name.startswith("audio_"):
                        self.assertEqual(buffer.shape[-1] % 32, 0)
                        self.assertEqual(torch.count_nonzero(buffer[:, :, :, self.config.max_source_positions:]).item(), 0)
                for position in range(8):
                    mask = torch.full((1, 12), -10000.0)
                    mask[:, :position + 1] = 0
                    self.decoder(torch.tensor([3]), torch.tensor([position]), mask, torch.ones(1, dtype=torch.int64))
            self.assertEqual(len(calls), 4)
            self.assertFalse(any("encoder_attn.k_proj" in name or "encoder_attn.v_proj" in name
                for name, _ in self.decoder.named_parameters()))
        finally:
            for handle in handles:
                handle.remove()

    @unittest.skipUnless(sys.platform == "darwin", "Core ML execution requires macOS")
    def test_conversion_retains_state_writes_and_dynamic_positions(self):
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                self.check_conversion(batch_size)

    def test_assignment_pass_preserves_partial_reversed_squeezed_and_input_updates(self):
        import coremltools as ct
        from coremltools.converters.mil.mil import Builder as mb, types
        from music_whisper_coreml.whisper_conversion import remove_whole_state_assignments

        cases = (
            ((1, 4), [0, 0], [1, 4], [1, 1], [False, False], True, 0),
            ((1, 2), [0, 0], [1, 2], [1, 1], [False, False], True, 1),
            ((1, 4), [0, 3], [1, -5], [1, -1], [False, False], True, 1),
            ((4,), [0, 0], [1, 4], [1, 1], [True, False], True, 1),
            ((1, 4), [0, 0], [1, 4], [1, 1], [False, False], False, 1),
        )
        for shape, begin, end, stride, squeeze, state, remaining in cases:
            with self.subTest(shape=shape, stride=stride, state=state):
                @mb.program(input_specs=[mb.StateTensorSpec((1, 4), dtype=types.fp16) if state
                        else mb.TensorSpec((1, 4), dtype=types.fp16), mb.TensorSpec(shape, dtype=types.fp16)],
                        opset_version=ct.target.iOS18)
                def program(value, update):
                    source = mb.read_state(input=value) if state else value
                    assigned = mb.slice_update(x=source, update=update, begin=begin, end=end, stride=stride,
                        begin_mask=[False, False], end_mask=[False, stride[-1] < 0], squeeze_mask=squeeze)
                    return mb.identity(x=assigned)

                remove_whole_state_assignments().apply(program)
                self.assertEqual(sum(op.op_type == "slice_update" for op in program.functions["main"].operations), remaining)

    def check_conversion(self, batch_size):
        import coremltools as ct
        import numpy as np
        from music_whisper_coreml.whisper_conversion import StatefulTextDecoder, decoder_conversion_pipeline

        torch = self.torch
        decoder = StatefulTextDecoder(deepcopy(self.reference_decoder), batch_size).eval()
        inputs = (torch.full((batch_size,), 3), torch.zeros(batch_size, dtype=torch.int64),
            torch.zeros(batch_size, 12), torch.ones(batch_size, dtype=torch.int64))
        with torch.inference_mode():
            for _, buffer in decoder.named_buffers():
                buffer.zero_()
            traced = torch.jit.trace(decoder, inputs, check_trace=False)
            traced(*inputs)
            traced(torch.full((batch_size,), 4), torch.ones(batch_size, dtype=torch.int64), inputs[2], inputs[3])
            self.assertTrue(traced.self_key_0[:, :, :, 1].abs().sum() > 0)
            model = ct.convert(traced,
                inputs=[ct.TensorType(name="input_ids", shape=(batch_size,), dtype=np.int32),
                    ct.TensorType(name="cache_length", shape=(batch_size,), dtype=np.int32),
                    ct.TensorType(name="decoder_key_padding_mask", shape=(batch_size, 12), dtype=np.float16),
                    ct.TensorType(name="active_rows", shape=(batch_size,), dtype=np.int32)],
                outputs=[ct.TensorType(name="logits", dtype=np.float16)],
                states=[ct.StateType(name=name, wrapped_type=ct.TensorType(shape=value.shape, dtype=np.float16))
                    for name, value in decoder.named_buffers()],
                minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_ONLY, pass_pipeline=decoder_conversion_pipeline())
        spec = model.get_spec()
        self.assertEqual(len(spec.description.state), 8)
        self.assertEqual({value.name for value in spec.description.input},
            {"input_ids", "cache_length", "decoder_key_padding_mask", "active_rows"})
        operations = model._mil_program.functions["main"].operations
        self.assertEqual(sum(op.op_type == "coreml_update_state" for op in operations), 4)
        self.assertFalse(any(op.op_type == "slice_update" for op in operations))
        self.assertTrue(all(not op.outputs[0].child_ops for op in operations if op.op_type == "coreml_update_state"))
        with torch.inference_mode():
            features = torch.randn(batch_size, 80, 1, 16)
            outputs = []
            for excerpt in (features, torch.randn_like(features), features):
                for name, buffer in decoder.named_buffers():
                    buffer.zero_()
                projections = self.encoder(excerpt)
                for index in range(self.config.decoder_layers):
                    for offset, kind in enumerate(("key", "value")):
                        getattr(decoder, f"audio_{kind}_{index}").copy_(projections[index * 2 + offset])
                state = model.make_state()
                for name, value in decoder.named_buffers():
                    if name.startswith("audio_"):
                        state.write_state(name, np.ascontiguousarray(value.numpy(), dtype=np.float32))
                current = []
                for position, token in ((0, 3), (1, 4), (2, 8), (11, 17)):
                    mask = torch.full((batch_size, 12), -10000.0)
                    mask[:, :position + 1] = 0
                    tokens = torch.tensor([token + row for row in range(batch_size)])
                    positions = torch.tensor([max(0, position - row) for row in range(batch_size)])
                    active = torch.tensor([int(row == 0 or position < 2) for row in range(batch_size)])
                    expected = decoder(tokens, positions, mask, active)
                    actual = model.predict({"input_ids": tokens.numpy().astype(np.int32),
                        "cache_length": positions.numpy().astype(np.int32),
                        "decoder_key_padding_mask": mask.numpy().astype(np.float16),
                        "active_rows": active.numpy().astype(np.int32)}, state=state)["logits"]
                    np.testing.assert_allclose(actual, expected.numpy(), atol=0.0006, rtol=0.01)
                    for name, value in decoder.named_buffers():
                        np.testing.assert_allclose(state.read_state(name), value.numpy(), atol=0.002, rtol=0.01)
                    current.append(actual.copy())
                outputs.append(current)
            for before, after in zip(outputs[0], outputs[2]):
                np.testing.assert_array_equal(before, after)
            state = model.make_state()
            for _, value in decoder.named_buffers():
                value.zero_()
            if batch_size > 1:
                active = np.array([1] + [0] * (batch_size - 1), dtype=np.int32)
                model.predict({"input_ids": np.full(batch_size, 3, dtype=np.int32),
                    "cache_length": np.zeros(batch_size, dtype=np.int32),
                    "decoder_key_padding_mask": np.zeros((batch_size, 12), dtype=np.float16),
                    "active_rows": active}, state=state)
                for name, _ in decoder.named_buffers():
                    self.assertEqual(np.count_nonzero(state.read_state(name)[1:]), 0)
