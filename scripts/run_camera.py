"""
scripts/run_camera.py
──────────────────────
Convenience launcher for CameraProcessor.

Usage:
    python -m scripts.run_camera --camera CAM-001 --station STA-001 --source 0
    python -m scripts.run_camera --camera CAM-002 --station STA-002 --source rtsp://192.168.1.10/stream
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cameras.camera_processor import CameraProcessor


def main():
    parser = argparse.ArgumentParser(description="SmartDetect Camera Processor")
    parser.add_argument("--camera",  required=True, help="Camera ID, e.g. CAM-001")
    parser.add_argument("--station", required=True, help="Location ID, e.g. LOC-001")
    parser.add_argument("--source",  default="0",  help="Webcam index (0,1,…) or RTSP URL")
    parser.add_argument("--api",     default="http://localhost:8000", help="Backend API base URL")
    parser.add_argument("--no-debug", action="store_true", help="Disable OpenCV preview window")
    args = parser.parse_args()

    # Convert numeric source to int
    source = int(args.source) if args.source.isdigit() else args.source

    processor = CameraProcessor(
        camera_id=args.camera,
        location_id=args.station,
        video_source=source,
        api_base_url=args.api,
        debug=not args.no_debug,
    )

    print(f"Starting Camera {args.camera} at Location {args.station} (source={source})")
    print("Press Q in the OpenCV window to stop.\n")

    try:
        processor.start()
    except KeyboardInterrupt:
        processor.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
