import sys

results = {}

try:
    import insightface
    results['InsightFace'] = f'OK  v{insightface.__version__}'
except Exception as e:
    results['InsightFace'] = f'FAIL  {e}'

try:
    import ultralytics
    results['YOLOv8 (ultralytics)'] = f'OK  v{ultralytics.__version__}'
except Exception as e:
    results['YOLOv8 (ultralytics)'] = f'FAIL  {e}'

try:
    import torch
    cuda = torch.cuda.is_available()
    results['PyTorch'] = f'OK  v{torch.__version__}  CUDA={cuda}'
except Exception as e:
    results['PyTorch'] = f'FAIL  {e}'

try:
    import torchvision
    results['torchvision'] = f'OK  v{torchvision.__version__}'
except Exception as e:
    results['torchvision'] = f'FAIL  {e}'

try:
    import torchreid
    results['torchreid'] = 'OK'
except Exception as e:
    results['torchreid'] = f'FAIL  {e}'

try:
    import cv2
    results['OpenCV'] = f'OK  v{cv2.__version__}'
except Exception as e:
    results['OpenCV'] = f'FAIL  {e}'

try:
    import onnxruntime
    results['onnxruntime'] = f'OK  v{onnxruntime.__version__}'
except Exception as e:
    results['onnxruntime'] = f'FAIL  {e}'

try:
    import numpy
    results['numpy'] = f'OK  v{numpy.__version__}'
except Exception as e:
    results['numpy'] = f'FAIL  {e}'

print("\n=== ML Dependency Check ===\n")
for name, status in results.items():
    icon = "[OK]" if status.startswith("OK") else "[!!]"
    print(f"  {icon}  {name}: {status}")
print()
