"""Download only pinned, checksum-verified pretrained inference files."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "899e9432dfe0998ba2ef31582d11c2d5f24a5185"
BASE = f"https://raw.githubusercontent.com/X-rayLaser/pytorch-handwriting-synthesis-toolkit/{REVISION}/checkpoints"
CHECKSUMS = {
    "46": {"model.pt": "34e3d0abbef3331f0ac76155048ce4b4f12301f91681d3335a423a2e0462ec2f",
           "meta.json": "00c383051895af5a091743652597dc4a9c456c70d07ec7ca51a96fcad6e6f70c"},
    "50": {"model.pt": "a8e27c995c8f71150d141898fbd6369113f8cb389ff359fef5e1c1d86eeb5755",
           "meta.json": "00c383051895af5a091743652597dc4a9c456c70d07ec7ca51a96fcad6e6f70c"},
    "52": {"model.pt": "29dabc00f41ee0a97aea5cbf29c2a6b1ae73cc4b83864babb4fa0bd7385be73b",
           "meta.json": "00c383051895af5a091743652597dc4a9c456c70d07ec7ca51a96fcad6e6f70c"},
    "56": {"model.pt": "e5eddd9f771ef1f0414909e61f173e88f293a35cb4c99cf359ac296552349e43",
           "meta.json": "00c383051895af5a091743652597dc4a9c456c70d07ec7ca51a96fcad6e6f70c"},
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(epoch):
    folder = ROOT / "models/graves" / f"Epoch_{epoch}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in CHECKSUMS[epoch].items():
        path = folder / name
        if path.is_file() and sha256(path) == expected:
            print(f"Verified {path.relative_to(ROOT)}")
            continue
        temporary = path.with_suffix(path.suffix + ".download")
        request = urllib.request.Request(f"{BASE}/Epoch_{epoch}/{name}", headers={"User-Agent": "InkForm/0.1"})
        print(f"Downloading Epoch_{epoch}/{name}", flush=True)
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 256):
                output.write(chunk)
        if sha256(temporary) != expected:
            raise RuntimeError(f"Checksum mismatch: {name}. The incomplete file remains at {temporary}.")
        temporary.replace(path)
        print(f"Verified {path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", choices=(*CHECKSUMS, "all"), default="52")
    args = parser.parse_args()
    for epoch in CHECKSUMS if args.epoch == "all" else (args.epoch,):
        download(epoch)
