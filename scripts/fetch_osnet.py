"""
scripts/fetch_osnet.py
───────────────────────
Download re-ID-trained OSNet weights for SmartDetect body re-identification.

Weights: osnet_x1_0, multi-source domain-generalization checkpoint
(MSMT17+Duke+CUHK03, cosine distance) from the deep-person-reid Model Zoo —
best choice for cameras the model has never seen.

Saved to models/osnet_x1_0_reid.pth. recognition/reid_model.py picks the file
up automatically on next start.

Usage:  python scripts/fetch_osnet.py
"""

from pathlib import Path

# This machine's HTTPS is intercepted (AV/proxy); only the Windows cert store
# trusts the interception cert — route Python TLS through it.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import gdown

# deep-person-reid MODEL_ZOO.md → "Multi-source domain generalization",
# osnet_x1_0 (MS+D+C), cosine distance
WEIGHTS_URL  = "https://drive.google.com/uc?id=1tuYY1vQXReEd8N8_npUkc7npPDDmjNCV"
WEIGHTS_FILE = Path("models") / "osnet_x1_0_reid.pth"


def main() -> None:
    WEIGHTS_FILE.parent.mkdir(exist_ok=True)
    if WEIGHTS_FILE.exists():
        print(f"Already present: {WEIGHTS_FILE}")
        return
    print("Downloading OSNet re-ID weights…")
    gdown.download(WEIGHTS_URL, str(WEIGHTS_FILE), quiet=False)
    print(f"Saved: {WEIGHTS_FILE} ({WEIGHTS_FILE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
