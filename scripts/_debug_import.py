import sys, traceback
sys.path.insert(0, r'C:\Users\lalit\OneDrive\Desktop\Project')
try:
    import backend.main
    print('backend.main OK')
except Exception:
    with open(r'C:\Users\lalit\OneDrive\Desktop\Project\debug_err.txt', 'w') as f:
        traceback.print_exc(file=f)
    traceback.print_exc()
