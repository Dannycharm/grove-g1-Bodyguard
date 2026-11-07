from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parents[1]
MANIFEST = PACKAGE / 'configs/gait_reference.json'

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_contract(verify=True):
    c = json.loads(MANIFEST.read_text())
    if verify:
        for name, digest in c['files'].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f'Reference identity changed: {name}')
    return c
