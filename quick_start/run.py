"""Load a released fold and run random inputs; this does not measure accuracy."""
import argparse
from pathlib import Path

import torch

from qjm import load_fold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        type=Path,
        default=Path(__file__).resolve().parent
        / "weights/own_dataset/image_train/seed2026/S01",
        help="Fold directory; relative paths are resolved from the current directory.",
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(2026)
    model = load_fold(args.fold, args.device)
    config = model.config
    eeg = torch.randn(2, *config["eeg_input_shape"], device=args.device)
    eog = torch.randn(2, 6, config["samples"], device=args.device)
    context = torch.randn(2, *config["context_shape"], device=args.device)
    logits = model(eeg, eog, context)

    if not torch.isfinite(logits).all():
        raise RuntimeError("Inference produced non-finite logits.")
    print("Fold:", args.fold.resolve())
    print("logits shape:", tuple(logits.shape))
    print("Random-input predictions (not accuracy):", logits.argmax(1).tolist())
    print("Class order:", config["class_names"])


if __name__ == "__main__":
    main()
