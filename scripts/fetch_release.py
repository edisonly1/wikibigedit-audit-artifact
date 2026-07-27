"""Download the released WikiBigEdit interval files into data/raw/.

The eight interval files (~368 MB total) are the original benchmark and are not
shipped with this artifact; they are fetched from the public HuggingFace dataset.
Run this once before the scripts that read data/raw/.
"""
from __future__ import annotations

import os
import urllib.request

BASE = "https://huggingface.co/datasets/lukasthede/WikiBigEdit/resolve/main"
FILES = [
    "wiki_big_edit_20240201_20240220.json",
    "wiki_big_edit_20240220_20240301.json",
    "wiki_big_edit_20240301_20240320.json",
    "wiki_big_edit_20240320_20240401.json",
    "wiki_big_edit_20240401_20240501.json",
    "wiki_big_edit_20240501_20240601.json",
    "wiki_big_edit_20240601_20240620.json",
    "wiki_big_edit_20240620_20240701.json",
]


def main() -> None:
    os.makedirs("data/raw", exist_ok=True)
    for f in FILES:
        dest = os.path.join("data", "raw", f)
        if os.path.exists(dest):
            print(f"have  {f}")
            continue
        print(f"fetch {f} ...", flush=True)
        urllib.request.urlretrieve(f"{BASE}/{f}", dest)
        print(f"      {os.path.getsize(dest) / 1e6:.1f} MB")
    print("done. data/raw/ is ready.")


if __name__ == "__main__":
    main()
