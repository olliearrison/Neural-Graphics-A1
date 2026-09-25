"""Download pinned, publicly reusable P8 textures; no scikit-image dependency."""
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCES = {
    "brick_photo": ("brick", "CC0Textures Bricks25; transformed by scikit-image", "https://ambientcg.com/view?id=Bricks025"),
    "grass": ("grass", "linolafett, Grass 01; cropped by scikit-image", "https://www.deviantart.com/linolafett/art/Grass-01-434853879"),
    "gravel": ("gravel", "CC0Textures Gravel04; cropped by scikit-image", "https://ambientcg.com/view?id=Gravel004"),
}


def main():
    output = Path(__file__).resolve().parents[1] / "data" / "own"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (source_name, credit, original_url) in SOURCES.items():
        url = f"https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/{source_name}.png"
        path = output / f"{name}.png"
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        records.append({"name": name, "file": str(path.relative_to(output.parent)), "download_url": url,
                        "original_url": original_url, "credit": credit, "license": "CC0-1.0",
                        "license_evidence": f"https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.{source_name}",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "preprocessing": "512x512 source used without resizing; grayscale replicated into three RGB channels at load time."})
        print(name, records[-1]["sha256"])
    (output / "sources.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
