#!/usr/bin/env python3
"""Validate a Pico UTF-8 corpus without loading it all into memory."""

from __future__ import annotations

import argparse
import hashlib
import codecs
from pathlib import Path


MINIMUM_BYTES = 20 * (256 + 1)


def inspect_corpus(path: Path, minimum_bytes: int = MINIMUM_BYTES) -> tuple[int, str]:
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    total_bytes = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            total_bytes += len(chunk)
            digest.update(chunk)
            decoder.decode(chunk)
        decoder.decode(b"", final=True)
    if total_bytes < minimum_bytes:
        raise ValueError(
            f"corpus has {total_bytes} bytes; at least {minimum_bytes} bytes are "
            "required for the default context and validation split"
        )
    return total_bytes, digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--minimum-bytes", type=int, default=MINIMUM_BYTES)
    args = parser.parse_args()
    try:
        size, sha256 = inspect_corpus(args.corpus, args.minimum_bytes)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        parser.error(str(error))
    print(f"path: {args.corpus.resolve()}")
    print(f"bytes: {size}")
    print(f"sha256: {sha256}")
    print("encoding: UTF-8")


if __name__ == "__main__":
    main()
