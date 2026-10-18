import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "open-mind-train" / "src"))
sys.path.insert(0, str(ROOT.parent / "open-mind-format" / "python" / "src"))

from open_mind import Trainer
from open_mind.trainer import PRESETS
from prepare_data import inspect_corpus


class PicoRecipeTests(unittest.TestCase):
    def test_config_matches_exact_15m_model_preset(self):
        recipe = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        architecture = recipe["architecture"]
        self.assertEqual(
            {key: value for key, value in architecture.items() if key != "tokenizer"},
            PRESETS["Pico"],
        )
        self.assertEqual(architecture["tokenizer"], "byte")
        hidden = architecture["hidden_size"]
        layers = architecture["layers"]
        vocab = architecture["vocab"]
        context = architecture["context"]
        feed_forward = architecture["intermediate_size"]
        layer_norms = 9 * hidden + feed_forward
        attention_and_mlp = 4 * hidden * hidden + 2 * hidden * feed_forward
        per_layer = layer_norms + attention_and_mlp
        parameters = vocab * hidden + context * hidden + 2 * hidden + layers * per_layer
        self.assertEqual(parameters, recipe["parameter_count"])
        self.assertEqual(parameters, 14_983_680)

    def test_trainer_config_can_be_constructed_without_starting_training(self):
        recipe = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        settings = recipe["training"]
        trainer = Trainer(
            model="Pico",
            device="cpu",
            dataset=ROOT / "not-opened-by-constructor.txt",
            steps=settings["steps"],
            checkpoint_every=settings["checkpoint_every"],
            batch_size=settings["batch_size"],
            seq_len=settings["seq_len"],
            gradient_accumulation_steps=settings["gradient_accumulation_steps"],
            learning_rate=settings["learning_rate"],
            optimizer=settings["optimizer"],
            weight_decay=settings["weight_decay"],
            validation_fraction=settings["validation_fraction"],
            eval_interval=settings["eval_interval"],
            eval_batches=settings["eval_batches"],
            warmup_steps=settings["warmup_steps"],
        )
        self.assertEqual(trainer.model_name, "Pico")
        self.assertEqual(trainer.optimizer_name, "adamw")
        self.assertEqual(trainer.steps * trainer.batch_size * trainer.seq_len * trainer.gradient_accumulation_steps, 819_200_000)

    def test_corpus_validation_hashes_utf8_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.txt"
            content = ("Pico local training corpus. " * 40).encode("utf-8")
            corpus.write_bytes(content)
            size, digest = inspect_corpus(corpus, minimum_bytes=100)
            self.assertEqual(size, len(content))
            self.assertEqual(len(digest), 64)
            corpus.write_bytes(b"\xff")
            with self.assertRaises(UnicodeDecodeError):
                inspect_corpus(corpus, minimum_bytes=0)


if __name__ == "__main__":
    unittest.main()
