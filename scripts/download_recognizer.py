"""Download pinned Microsoft TrOCR files for fully offline handwriting readback."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "b4648cfa171985a6745f37ddd637e98c0da958ac"
BASE = f"https://huggingface.co/microsoft/trocr-small-handwritten/resolve/{REVISION}"
FILES = {
    "config.json": "2b6e06edec319984eca99c932d940bfa1f308db3f74d6279440b906c373bf727",
    "generation_config.json": "41149cdcffec4d657f32dfcddd9b208037f01286c9e07945c724908c58ed0193",
    "preprocessor_config.json": "eec0c686d2e2ea6d5887f87e1c6a34ef5bf1ba5b2ac98b8675e131da646fa1d2",
    "pytorch_model.bin": "1b83102cbc1520dee1c3937ac334da83da18cf4683d46ebd1bd4e93ebe584dd7",
    "sentencepiece.bpe.model": "6f5e2fefcf793761a76a6bfb8ad35489f9c203b25557673284b6d032f41043f4",
    "special_tokens_map.json": "58dc0a3d15e3bad4861bcf2dee770ea327344d185182d4c5d2bf253285e53a83",
    "tokenizer_config.json": "e098c98c6834fbcd3f3e75215e3e8b8868f0fe5bf07aef93d28e4f7196db293c",
}


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    folder = ROOT / "models/trocr-small-handwritten"
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = folder / name
        if target.is_file() and digest(target) == expected:
            print(f"Verified {name}")
            continue
        temporary = folder / (name + ".download")
        print(f"Downloading {name}", flush=True)
        request = urllib.request.Request(f"{BASE}/{name}", headers={"User-Agent": "InkForm/0.1"})
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as output:
            while chunk := source.read(256 * 1024):
                output.write(chunk)
        if digest(temporary) != expected:
            raise RuntimeError(f"Checksum mismatch for {name}; the incomplete download was not installed.")
        temporary.replace(target)
        print(f"Verified {name}")


if __name__ == "__main__":
    main()
