# CanViT (Canvas Vision Transformer) -- PyTorch Reference Implementation

Reference PyTorch implementation of CanViT (Canvas Vision Transformer).

_This is an early release. For details, a preprint version of our manuscript "CanViT: Toward Active Vision Foundation Models" will be available in the coming weeks._

---

CanViT is a scalable recurrent architecture for fine-grained vision, and the first **Active Vision Foundation Model (AVFM)**: a foundation model for active vision that is both task-agnostic and policy-agnostic.

CanViT processes scenes through sequences of localized glimpses, integrating observations over time into a persistent scene-wide latent workspace — the **canvas** — via **Canvas Attention**, an efficient asymmetric cross-attention mechanism which is based on Scene-Relative Rotary Position Embeddings and eliminates canvas-side QKVO projections.

CanViT-B is pretrained on 1 billion glimpses taken from 13.5 million ImageNet-21k scenes, via **passive-to-active dense distillation** from a frozen DINOv3 ViT-B teacher, without human annotations.

## Quickstart

```bash
uv run demos/basic.py
```

## Pretrained checkpoints

We release checkpoints on HuggingFace under the [`canvit`](https://huggingface.co/canvit) namespace.

The following checkpoints are currently available:

- [`canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02`](https://huggingface.co/canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02)


## Citation

If you use this work, please cite this repository.
An updated citation will be available upon preprint release.

```bibtex
@misc{berreby2026canvit,
  title={CanViT: Toward Active Vision Foundation Models},
  author={Berreby, Yoha{\"i}-Eliel and Du, Sabrina and Durand, Audrey and Krishna, Suresh},
  year={2026},
  howpublished={\url{https://github.com/m2b3/CanViT-PyTorch}}
}
```

## License

MIT. See [LICENSE.md](LICENSE.md) for details.
