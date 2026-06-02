
import importlib
import os
import subprocess
import sys
import sysconfig

def header(s):
    print("\n" + "═" * 70)
    print(s)
    print("═" * 70)

header("python")
print(f"sys.executable     : {sys.executable}")
print(f"sys.version        : {sys.version.split()[0]}")
print(f"site-packages root : {sysconfig.get_paths()['purelib']}")

header("env vars (CUDA-relevant)")
for k in ("CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "BNB_CUDA_VERSION",
         "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES"):
    print(f"  {k:30s} = {os.environ.get(k, '(unset)')}")

header("where is libcudart.so.* on disk?")
subprocess.run(["bash", "-lc",
                "find / -name 'libcudart.so*' 2>/dev/null | head -50"])

header("ldconfig -p | grep cudart")
subprocess.run(["bash", "-lc", "ldconfig -p | grep -i cudart || echo '(none)'"])

header("nvidia-* pip packages")
subprocess.run(["bash", "-lc", "pip list 2>/dev/null | grep -i ^nvidia || echo '(none)'"])

header("bitsandbytes package files")
subprocess.run(["bash", "-lc",
                "ls -la /usr/local/lib/python3.12/dist-packages/bitsandbytes/ 2>/dev/null | head -40 "
                "|| ls -la $(python -c 'import bitsandbytes, os; print(os.path.dirname(bitsandbytes.__file__))') 2>/dev/null "
                "|| echo '(bnb dir not found)'"])

header("torch CUDA build info")
try:
    import torch
    print(f"torch.__version__         : {torch.__version__}")
    print(f"torch.version.cuda        : {torch.version.cuda}")
    print(f"torch.backends.cudnn.ver  : {torch.backends.cudnn.version()}")
    print(f"torch.cuda.is_available() : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device 0               : {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"  capability             : {cap}  (SM{cap[0]*10+cap[1]})")
except Exception as e:
    print(f"torch import failed: {e}")

header("bitsandbytes import attempt")
try:
    import bitsandbytes as bnb
    print(f"✅ bnb {bnb.__version__} imported cleanly")
except Exception as e:
    print(f"❌ {type(e).__name__}: {e}")
    # On failure, dlopen each candidate so we see the real linker error.
    import ctypes
    for cand in ("libcudart.so.12", "libcudart.so.13", "libcudart.so"):
        try:
            ctypes.CDLL(cand)
            print(f"   ✅ dlopen({cand}) OK")
        except OSError as oe:
            print(f"   ❌ dlopen({cand}): {oe}")

header("nvidia-smi")
subprocess.run(["bash", "-lc", "nvidia-smi || echo '(nvidia-smi not available)'"])

header("other relevant package versions")
for pkg in ("transformers", "accelerate", "datasets", "trl", "peft"):
    try:
        m = importlib.import_module(pkg)
        print(f"  {pkg:13s}: {getattr(m, '__version__', '(no __version__)')}")
    except Exception as e:
        print(f"  {pkg:13s}: MISSING ({e})")
