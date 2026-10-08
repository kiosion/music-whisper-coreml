import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

import torch
from argmaxtools.nn import AttentionType
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

STATE_ROW_ALIGNMENT = 32


@register_pass(namespace="music_whisper_coreml")
class remove_whole_state_assignments(AbstractGraphPass):
    """Remove full-buffer assignments introduced by TorchScript's required slice syntax."""

    def apply(self, program):
        for function in program.functions.values():
            for operation in list(function.operations):
                if operation.op_type != "slice_update" or operation.x.op is None or operation.x.op.op_type != "read_state":
                    continue
                if operation.outputs[0] in function.outputs:
                    continue
                if operation.x.sym_type != operation.update.sym_type or operation.outputs[0].sym_type != operation.update.sym_type:
                    continue
                bounds = [getattr(getattr(operation, name), "val", None) for name in
                    ("begin", "end", "stride", "begin_mask", "end_mask", "squeeze_mask")]
                if any(value is None for value in bounds):
                    continue
                begin, end, stride, begin_mask, end_mask, squeeze = bounds
                if any(stride != 1) or any(squeeze):
                    continue
                if any(not masked and value != 0 for value, masked in zip(begin, begin_mask)):
                    continue
                if any(not masked and value != size for value, size, masked in zip(end, operation.x.shape, end_mask)):
                    continue
                function.replace_uses_of_var_after_op(operation, operation.outputs[0], operation.update)
                function.remove_ops([operation])


def decoder_conversion_pipeline():
    """Retain direct cache values while removing redundant full-buffer assignment operations."""
    import coremltools as ct

    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes({"common::prefer_state_in_downstream"})
    pipeline.append_pass("music_whisper_coreml::remove_whole_state_assignments")
    pipeline.append_pass("common::dead_code_elimination")
    return pipeline


class ProjectedAudioEncoder(torch.nn.Module):
    """Encode audio and compute each decoder layer's fixed attention projections."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        # Unaligned audio state produces incorrect Neural Engine attention on macOS 26.6.
        self.padding = (-decoder.config.max_source_positions) % STATE_ROW_ALIGNMENT
        self.keys = torch.nn.ModuleList([layer.encoder_attn.k_proj for layer in decoder.layers])
        self.values = torch.nn.ModuleList([layer.encoder_attn.v_proj for layer in decoder.layers])

    def forward(self, melspectrogram_features):
        encoded = self.encoder(melspectrogram_features)
        return tuple(torch.nn.functional.pad(projection(encoded), (0, self.padding))
            for pair in zip(self.keys, self.values) for projection in pair)


class StatefulTextDecoder(torch.nn.Module):
    """Decode one token while retaining self-attention and audio caches in Core ML state."""

    def __init__(self, decoder, batch_size=1):
        super().__init__()
        self.decoder = decoder
        config = decoder.config
        audio_length = config.max_source_positions + (-config.max_source_positions) % STATE_ROW_ALIGNMENT
        for index, layer in enumerate(decoder.layers):
            for kind, length in (("self", config.max_target_positions), ("audio", audio_length)):
                for projection in ("key", "value"):
                    self.register_buffer(f"{kind}_{projection}_{index}", torch.zeros(batch_size, config.d_model, 1, length))
            layer.encoder_attn.attention_type = AttentionType.KVCachedEncoderDecoderCrossAttention
            del layer.encoder_attn.k_proj
            del layer.encoder_attn.v_proj

    def forward(self, input_ids, cache_length, decoder_key_padding_mask, active_rows):
        hidden = self.decoder.embed_tokens(input_ids) + self.decoder.embed_positions(cache_length)
        hidden = hidden[:, :, None, None]
        update = ((torch.arange(self.decoder.config.max_target_positions)[None, :] == cache_length[:, None])
            & (active_rows[:, None] != 0))[:, None, None, :]
        for index, layer in enumerate(self.decoder.layers):
            normalized = layer.self_attn_layer_norm(hidden)
            query = layer.self_attn.q_proj(normalized)
            key = getattr(self, f"self_key_{index}")
            value = getattr(self, f"self_value_{index}")
            # Core ML on macOS 26.6 overwrites column zero for converted runtime slice bounds.
            updated_key = torch.where(update, layer.self_attn.k_proj(normalized), key)
            updated_value = torch.where(update, layer.self_attn.v_proj(normalized), value)
            # Core ML Tools 9 requires slice syntax for TorchScript state assignment.
            key[:] = updated_key
            value[:] = updated_value
            attention = layer.self_attn.sdpa_implementation.sdpa(
                query, updated_key, updated_value, decoder_key_padding_mask, causal=False)
            hidden = hidden + layer.self_attn.o_proj(attention)
            hidden = hidden + layer.encoder_attn(layer.encoder_attn_layer_norm(hidden),
                key_cache=getattr(self, f"audio_key_{index}")[:, :, :, :self.decoder.config.max_source_positions],
                value_cache=getattr(self, f"audio_value_{index}")[:, :, :, :self.decoder.config.max_source_positions])[0]
            hidden = hidden + layer.fc2(layer.act_fn(layer.fc1(layer.final_layer_norm(hidden))))
        hidden = self.decoder.layer_norm(hidden)
        return torch.nn.functional.linear(hidden.squeeze(2).transpose(1, 2), self.decoder.embed_tokens.weight)


def main():
    """Convert a local Whisper caption checkpoint for the native Core ML runtime."""
    import coremltools as ct
    import numpy as np
    from argmaxtools import _sdpa
    from transformers import WhisperForConditionalGeneration
    from whisperkit import audio_encoder, text_decoder

    from .artifacts import artifact_digest
    from .whisper_coreml import validate_checkpoint

    parser = argparse.ArgumentParser(description="Convert a local Music-Whisper checkpoint to Core ML.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, choices=range(1, 5), default=1,
        help="fixed excerpt capacity of the converted model, 1-4 (default: 1)")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists; select a new directory")
    if args.output_dir.resolve().is_relative_to(args.model.resolve()):
        parser.error("output directory must be outside the source model directory")
    if not 1 <= args.threads <= 64:
        parser.error("threads must be 1-64")
    torch.set_num_threads(args.threads)
    validate_checkpoint(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(args.model), local_files_only=True, torch_dtype=torch.float32,
        use_safetensors=True, attn_implementation="eager").eval()
    config = model.config
    if not torch.equal(model.proj_out.weight, model.model.decoder.embed_tokens.weight):
        raise ValueError("WhisperKit conversion requires tied output and token embeddings")
    if config.num_mel_bins != 80 or config.max_source_positions != 1500:
        raise ValueError("caption runtime requires 80-bin, 30-second Whisper features")
    if config.max_target_positions != 448:
        raise ValueError("caption runtime requires the 448-token Whisper context")
    audio_encoder.SDPA_IMPL = _sdpa.SplitHeadsQ
    encoder = audio_encoder.WhisperAudioEncoder(config).eval()
    encoder.load_state_dict(model.model.encoder.state_dict())
    decoder = text_decoder.WhisperTextDecoder(config).eval()
    decoder.load_state_dict(model.model.decoder.state_dict())
    encoder = ProjectedAudioEncoder(encoder, decoder).eval()
    decoder = StatefulTextDecoder(decoder, args.batch_size).eval()
    context = config.max_target_positions
    inputs = {
        "input_ids": torch.full((args.batch_size,), config.decoder_start_token_id, dtype=torch.int64),
        "cache_length": torch.zeros(args.batch_size, dtype=torch.int64),
        "decoder_key_padding_mask": torch.full((args.batch_size, context), -10000.0),
        "active_rows": torch.ones(args.batch_size, dtype=torch.int64),
    }
    inputs["decoder_key_padding_mask"][:, 0] = 0
    args.output_dir.mkdir(parents=True)
    components = (
        ("AudioEncoder", encoder, {"melspectrogram_features": torch.zeros((args.batch_size, 80, 1, 3000))},
            [f"audio_{kind}_{index}" for index in range(config.decoder_layers) for kind in ("key", "value")]),
        ("TextDecoder", decoder, inputs, ["logits"]),
    )
    with torch.inference_mode():
        for name, module, values, outputs in components:
            print("Converting " + name, flush=True)
            traced = torch.jit.trace(module, tuple(values.values()), check_trace=False)
            states = [ct.StateType(name=key, wrapped_type=ct.TensorType(shape=value.shape, dtype=np.float16))
                for key, value in module.named_buffers()] if name == "TextDecoder" else []
            converted = ct.convert(traced,
                inputs=[ct.TensorType(name=key, shape=value.shape,
                    dtype=np.int32 if value.dtype == torch.int64 else np.float16)
                    for key, value in values.items()],
                outputs=[ct.TensorType(name=key, dtype=np.float16) for key in outputs],
                states=states,
                minimum_deployment_target=ct.target.macOS15,
                compute_precision=ct.precision.FLOAT16, skip_model_load=True,
                pass_pipeline=decoder_conversion_pipeline() if name == "TextDecoder" else None)
            package = args.output_dir / (name + ".mlpackage")
            converted.save(str(package))
            subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(args.output_dir)], check=True)
    manifest = {
        "version": 4, "runtime": "coreml", "precision": "float16", "batch_size": args.batch_size,
        "model_artifact_digest": artifact_digest(args.model, ("*.json", "*.safetensors", "*.model", "*.txt", "*.jinja")),
        "max_target_positions": context, "num_mel_bins": config.num_mel_bins,
        "max_source_positions": config.max_source_positions,
        "audio_cache_length": decoder.audio_key_0.shape[-1],
        "decoder_layers": config.decoder_layers, "d_model": config.d_model,
        "conversion_versions": {name: importlib.metadata.version(name)
            for name in ("whisperkit", "argmaxtools", "coremltools", "torch", "transformers")},
        "artifacts": {name: artifact_digest(args.output_dir / (name + ".mlmodelc"), ("*",))
            for name in ("AudioEncoder", "TextDecoder")},
    }
    (args.output_dir / "caption-coreml.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
