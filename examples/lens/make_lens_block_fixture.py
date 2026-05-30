import argparse
import importlib.util
import struct
from pathlib import Path

import torch


def load_lens_transformer_module(lens_src: str):
    transformer_path = Path(lens_src) / "transformer.py"
    spec = importlib.util.spec_from_file_location("lens_transformer_direct", transformer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_tensor(out, name: str, tensor: torch.Tensor):
    tensor = tensor.detach().cpu().contiguous().to(torch.float32)
    encoded = name.encode("utf-8")
    out.write(struct.pack("<II", len(encoded), tensor.ndim))
    out.write(encoded)
    for dim in tensor.shape:
        out.write(struct.pack("<q", int(dim)))
    out.write(tensor.numpy().tobytes(order="C"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-src", default=r"F:\Paralol\local\Lens\lens")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(1234)
    module = load_lens_transformer_module(args.lens_src)
    block = module.LensTransformerBlock(
        dim=12,
        num_attention_heads=3,
        attention_head_dim=4,
        rms_norm=True,
        gate_mlp=True,
    ).eval()

    hidden = torch.randn(1, 4, 12, dtype=torch.float32) * 0.25
    encoder = torch.randn(1, 3, 12, dtype=torch.float32) * 0.25
    temb = torch.randn(1, 12, dtype=torch.float32) * 0.25
    img_freqs = torch.polar(torch.ones(4, 2), torch.randn(4, 2) * 0.25)
    txt_freqs = torch.polar(torch.ones(3, 2), torch.randn(3, 2) * 0.25)
    attention_mask = torch.zeros(1, 1, 1, 7, dtype=torch.float32)
    attention_mask[:, :, :, -1] = float("-inf")

    with torch.no_grad():
        expected_encoder, expected_hidden = block(
            hidden_states=hidden,
            encoder_hidden_states=encoder,
            temb=temb,
            image_rotary_emb=(img_freqs, txt_freqs),
            attention_mask=attention_mask,
        )

    tensors = {
        "input.hidden": hidden,
        "input.encoder": encoder,
        "input.temb": temb,
        "input.img_freqs": torch.view_as_real(img_freqs),
        "input.txt_freqs": torch.view_as_real(txt_freqs),
        "input.attention_mask": attention_mask.reshape(1, 7),
        "expected.encoder": expected_encoder,
        "expected.hidden": expected_hidden,
    }
    tensors.update({f"state.{name}": value for name, value in block.state_dict().items()})

    torch.manual_seed(4321)
    model = module.LensTransformer2DModel(
        patch_size=2,
        in_channels=8,
        out_channels=2,
        num_layers=2,
        attention_head_dim=4,
        num_attention_heads=3,
        enc_hidden_dim=5,
        axes_dims_rope=(0, 2, 2),
        gate_mlp=True,
        rms_norm=True,
        multi_layer_encoder_feature=True,
        selected_layer_index=(0, 1, 2, 3),
    ).eval()
    model_hidden = torch.randn(1, 4, 8, dtype=torch.float32) * 0.25
    model_features = [torch.randn(1, 3, 5, dtype=torch.float32) * 0.25 for _ in range(4)]
    model_mask = torch.tensor([[True, True, False]])
    model_timestep = torch.tensor([1.0], dtype=torch.float32)
    with torch.no_grad():
        model_hidden_proj = model.img_in(model_hidden)
        model_temb = model.time_text_embed(model_timestep.to(model_hidden_proj.dtype), model_hidden_proj)
        model_img_freqs, model_txt_freqs = model.pos_embed([(1, 2, 2)], [3], device=model_hidden.device)
        model_attention_mask = model._build_joint_attention_mask(model_mask, model_hidden.shape[1])
        expected_model = model(
            hidden_states=model_hidden,
            encoder_hidden_states=model_features,
            encoder_hidden_states_mask=model_mask,
            timestep=model_timestep,
            img_shapes=[(1, 2, 2)],
        )
    tensors.update(
        {
            "full.input.hidden": model_hidden,
            "full.input.temb": model_temb,
            "full.input.img_freqs": torch.view_as_real(model_img_freqs),
            "full.input.txt_freqs": torch.view_as_real(model_txt_freqs),
            "full.input.attention_mask": model_attention_mask.reshape(1, 7),
            "full.expected.output": expected_model,
        }
    )
    for i, feature in enumerate(model_features):
        tensors[f"full.input.feature_{i}"] = feature
    tensors.update({f"full.state.{name}": value for name, value in model.state_dict().items()})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as out:
        out.write(b"LENSBLK1")
        out.write(struct.pack("<I", len(tensors)))
        for name, tensor in tensors.items():
            write_tensor(out, name, tensor)
    print(f"wrote {output} tensors={len(tensors)}")


if __name__ == "__main__":
    main()
