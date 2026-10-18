# Security

Open Mind v0.1 is intended for local use. OMA listens on `127.0.0.1:8080` by
default and does not implement authentication, authorization, or TLS. Do not
expose it to an untrusted network. Do not send sensitive paths or prompts to
instances you do not control.

OMF files are parsed as untrusted input. Use models from trusted sources and
report parser failures or excessive allocations with a reproducible case that
does not contain private model weights.

To report a vulnerability, contact the maintainers privately before publishing
exploitable details. Do not include credentials, private models, or personal
data in reports.
