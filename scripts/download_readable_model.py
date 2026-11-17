"""Fetch pinned pretrained stroke weights and a fixed writer's real pen sample."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "61544f003893c99cbcb9013b33d0a437bbe1ac6f"
BASE = f"https://raw.githubusercontent.com/niccolodevries/svg-handwriting-generator/{REVISION}/synth"
FILES = {
    "styles/style-2-chars.npy": ("styles/style-2-chars.npy", "7b1f82d45c1e217026d355f9f5a4f63b3204181d4f5eccab7805f3d666e94325"),
    "styles/style-2-strokes.npy": ("styles/style-2-strokes.npy", "565d09ed7fd62a28aff9a58ea6dd7bdab2e60bd1b5f4eabfbe228abd1a0425a0"),
    "model.pt": ("checkpoints/model.pt", "47ede618fa3772e9a0b2c531397546f2c0b3a23776bdfb6f4918114f45304571"),
    "styles/style-0-chars.npy": ("styles/style-0-chars.npy", "b17a762c2b4424d6bcf386b91e155696b316e418a82e3b0925e3bdb81975a2a9"),
    "styles/style-0-strokes.npy": ("styles/style-0-strokes.npy", "4c6761c223bcebcf642d9658040d2d15cecc03970aa9fae0f7665c9f70961439"),
    "styles/style-3-chars.npy": ("styles/style-3-chars.npy", "d974b6ac1f894f813925e731c3942b2e90196a8d60a3591a44d4cf3dfa66907b"),
    "styles/style-3-strokes.npy": ("styles/style-3-strokes.npy", "b4f606447e3f757394c9d9f1fd8677b95efc84aef7f104ccd4af490174bd5659"),
}


def checksum(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source,"sha256").hexdigest()


def main():
    folder=ROOT/"models/vasquez"
    for filename,(remote,expected) in FILES.items():
        target=folder/filename
        if target.is_file() and checksum(target)==expected:
            print(f"Verified {filename}")
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+".download")
        print(f"Downloading {filename}",flush=True)
        request=urllib.request.Request(f"{BASE}/{remote}",headers={"User-Agent":"InkForm/0.1"})
        with urllib.request.urlopen(request,timeout=120) as response,temporary.open("wb") as output:
            while data:=response.read(256*1024):
                output.write(data)
        if checksum(temporary)!=expected:
            raise RuntimeError(f"Checksum mismatch: {filename}. Incomplete data was not installed.")
        temporary.replace(target)
        print(f"Verified {filename}")


if __name__=="__main__":
    main()
