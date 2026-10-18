import io
import tempfile
import unittest
from pathlib import Path

import torch

from open_mind_format import read_omf, write_omf


class OMFReaderTests(unittest.TestCase):
    def test_python_writer_round_trips_each_omf_dtype(self):
        tensors = {
            "half": torch.tensor([-1.0, 0.0, 1.0]),
            "four": torch.tensor([-1.0, 0.0, 1.0]),
            "eight": torch.tensor([-1.0, 0.0, 1.0]),
        }
        with tempfile.TemporaryDirectory() as directory:
            for dtype in ("fp16", "q4_0", "q8_0"):
                path = Path(directory) / f"{dtype}.omf"
                write_omf(path, tensors, {"hidden_size": 1}, metadata={"step": 5}, dtype=dtype)
                model = read_omf(path)
                self.assertEqual(model.metadata["step"], 5)
                self.assertEqual(model.config["hidden_size"], 1)
                self.assertEqual(set(model.tensors), set(tensors))
                self.assertAlmostEqual(float(model.tensors["half"][2]), 1.0, places=3)
                self.assertAlmostEqual(float(model.tensors["four"][0]), -1.0, places=2)
                self.assertAlmostEqual(float(model.tensors["eight"][0]), -1.0, places=2)

    def test_reader_rejects_truncated_header(self):
        with self.assertRaisesRegex(ValueError, "truncated"):
            from open_mind_format.reader import _read_exact

            _read_exact(io.BytesIO(b"OM"), 3)

    def test_reader_rejects_bad_magic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.omf"
            path.write_bytes(b"BAD")
            with self.assertRaisesRegex(ValueError, "magic"):
                read_omf(path)


if __name__ == "__main__":
    unittest.main()
