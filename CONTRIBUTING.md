# Contributing

Thanks for helping improve Open Mind. The project is at experimental v0.1.
Changes must preserve compatibility between the OMF specification in
`open-mind-format/OMF_SPEC.md`, the Python writer, and the Rust reader.

## Changes

- Keep code and test data local; do not add model weights, credentials, private
  data, or corpus files without permission.
- For format changes, update the specification and both sides of the
  interchange; add a round-trip test or a test that rejects invalid data.
- For Rust changes, run `cargo test --workspace`.
- For Python changes, install in a virtual environment and run the training
  smoke test with the `tiny` preset and a local test corpus.
- Describe observable behavior and any new limitations.

## Reporting issues

Include the operating system, Rust/Python versions, command you ran, and a
small example that contains no private data or model checkpoints.
