"""Isolated MLX DeepFilterNet process.

Keeping this in a short-lived process makes GPU memory release predictable and
prevents a failed enhancement from taking down the editor service.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: deepfilter_worker.py INPUT.wav OUTPUT.wav MODEL_DIR", file=sys.stderr)
        return 2
    source, output, model_dir = map(Path, sys.argv[1:])
    from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel

    model = DeepFilterNetModel.from_pretrained(str(model_dir), subfolder="v3")
    model.enhance_file_streaming(str(source), str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

