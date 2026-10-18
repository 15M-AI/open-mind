import argparse

from .trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local Open Mind model and save OMF checkpoints")
    parser.add_argument("--model", default="15M", choices=("15M", "tiny"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--dataset", default="data.txt")
    parser.add_argument("--output", default="pico.omf")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dtype", choices=("fp16", "q4_0", "q8_0"), default="fp16")
    parser.add_argument("--optimizer", choices=("sgd8bit", "adamw"), default="sgd8bit")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=500)
    args = parser.parse_args()
    Trainer(
        model=args.model,
        device=args.device,
        dataset=args.dataset,
        output=args.output,
        steps=args.steps,
        checkpoint_every=args.checkpoint_every,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        dtype=args.dtype,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        warmup_steps=args.warmup_steps,
    ).train()
