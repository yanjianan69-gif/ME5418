# PyTorch binary used for RTX 50-series reproduction

Included file: `torch-2.3.0a0+git63d5e92-cp38-cp38-linux_x86_64.whl` (approximately 643 MiB).

- Package version: `2.3.0a0+git63d5e92`
- Platform / ABI: Linux x86_64 / CPython 3.8
- Reported CUDA build: 12.8
- SHA-256: `6a9707962e83ededded35df45df0149951b8276df37035761ba61173428c1512`
- Artifact identity: the unmodified wheel is identified by the SHA-256 above.

The original upstream download URL and complete build recipe were not recorded
in the available artifact metadata. No upstream source or reproducible source-build claim
is inferred from the filename. Bundling the actual artifact makes binary
reproduction possible without depending on the author's local wheel path.
PyTorch license notices are retained inside the wheel's dist-info directory.

This is NOT the public `torch==2.3.1` release. Do not replace it with an arbitrary
PyPI torch version when using an RTX 50-series GPU and Isaac Gym's Python 3.8 ABI.
It dynamically links system CUDA and OpenMPI libraries; it is NOT a self-contained
CUDA runtime. Follow the external dependency instructions in the main README.

For a smaller submission targeting only RTX 40-series machines, this binary is
not required by `environment-rtx40.yml`. Keep it when distributing the exact
RTX 50-series environment described by `environment.yml`.
