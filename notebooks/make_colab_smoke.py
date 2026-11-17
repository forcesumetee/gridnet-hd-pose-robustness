"""Generate notebooks/colab_smoke_litept.ipynb (v2: isolated micromamba env, training as a script)."""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})


md("""# GridNet-HD × LitePT — Colab smoke test v2 (Phase 0, gate G1)

**ทำไมต้องมี v2**: Colab รุ่นปัจจุบันมาพร้อม Python 3.13 + torch 2.11 + CUDA 13 ซึ่ง spconv และ flash-attn ยังไม่มี wheel สำเร็จรูป (ติดตั้งแล้ว compile พัง) notebook นี้จึงสร้าง **environment แยก** ด้วย micromamba (Python 3.10 + torch 2.4.1/cu124 — ทุก dependency มี wheel) แล้วรันการเทรนเป็น script ผ่าน interpreter นั้น — เป็นสูตรเดียวกับที่จะใช้บน RunPod/LANTA

**ก่อนรัน**
1. Runtime → Change runtime type → **L4** (หรือ A100) — T4 ใช้ไม่ได้
2. บน Drive ต้องมี `GridNet/smoke/smoke_t5b5.npz` (จาก `scripts/export_smoke.py`)
3. งบโดยประมาณ: L4 ~45 นาที ≈ 4 CU (ติดตั้ง env ~10 นาที + train ~15 นาที)""")

code("""#@title 1. ตรวจ GPU (ต้องเป็น L4 / A100)
import subprocess, torch, platform
print("GPU:", subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip())
print("kernel python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.version.cuda, "(kernel only; the env below uses its own)")
cc = torch.cuda.get_device_capability(); print("compute capability", cc)
assert cc[0] >= 8, "flash-attn needs Ampere or newer (A100 / L4). Change runtime type."
""")

code("""#@title 2. Mount Drive, copy smoke file
from google.colab import drive
drive.mount('/content/drive')
import os, shutil
DRIVE = '/content/drive/MyDrive/GridNet'
SMOKE_FILE = f'{DRIVE}/smoke/smoke_t5b5.npz'   #@param {type:"string"}
CKPT_DIR = f'{DRIVE}/ckpt/smoke'
os.makedirs(CKPT_DIR, exist_ok=True)
assert os.path.exists(SMOKE_FILE), f"upload smoke_t5b5.npz to {os.path.dirname(SMOKE_FILE)} first"
if not os.path.exists('/content/smoke.npz'):
    shutil.copy(SMOKE_FILE, '/content/smoke.npz')
print(f"smoke file: {os.path.getsize('/content/smoke.npz') / 1e6:.0f} MB")
""")

code("""#@title 3. สร้าง environment แยก (Python 3.10 + torch 2.4.1 cu124 + spconv + flash-attn) ~10 นาที
import subprocess, textwrap, time, os
ENV = '/content/env'
script = r'''__COLAB_ENV_SH__'''
open('/content/colab_env.sh', 'w').write(script)
t0 = time.time()
p = subprocess.Popen(['bash', '/content/colab_env.sh', ENV], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end='')
p.wait()
print(f"exit code {p.returncode} after {(time.time() - t0) / 60:.1f} min")
assert p.returncode == 0, "environment build failed - see log above"
PY = f'{ENV}/bin/python'
""")

code("""#@title 4. Clone LitePT และตรวจ import ใน env
import subprocess, os
if not os.path.exists('/content/LitePT'):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/prs-eth/LitePT.git", "/content/LitePT"], check=True)
r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0, '/content/LitePT'); from litept.model import LitePT; print('LitePT import ok')"],
                   capture_output=True, text=True)
print(r.stdout, r.stderr[-2000:])
assert r.returncode == 0
""")

code("""#@title 5. เขียน smoke_train.py แล้วรัน (forward check → 300 its → eval)
ITERS = 300   #@param {type:"integer"}
import subprocess, time
open('/content/smoke_train.py', 'w').write(r'''__SMOKE_TRAIN_PY__''')
t0 = time.time()
p = subprocess.Popen([PY, '/content/smoke_train.py', '--smoke', '/content/smoke.npz', '--litept', '/content/LitePT',
                      '--ckpt-dir', CKPT_DIR, '--iters', str(ITERS)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end='')
p.wait()
print(f"exit code {p.returncode} after {(time.time() - t0) / 60:.1f} min")
""")

code("""#@title 6. (ทางเลือก) เก็บ env ลง Drive เพื่อ session หน้าไม่ต้องสร้างใหม่ (~3 GB, 3–5 นาที)
SAVE_ENV = False   #@param {type:"boolean"}
if SAVE_ENV:
    import subprocess, os
    os.makedirs(f'{DRIVE}/env', exist_ok=True)
    subprocess.run(['tar', '-C', '/content', '-czf', f'{DRIVE}/env/env_py310_torch241_cu124.tar.gz', 'env'], check=True)
    print("saved; next session: tar -C /content -xzf <that file> then skip cell 3")
""")

md("""## ผลที่ต้องได้เพื่อผ่าน gate G1
- เซลล์ 3: บรรทัดสุดท้ายขึ้น `env ok | torch 2.4.1+cu124 ... gpu True`
- เซลล์ 4: `LitePT import ok`
- เซลล์ 5: `forward ok`, loss ลดจาก ~2.4 ชัดเจนใน 300 its, จด **s/it** และ **peak VRAM** ลง `notes/journal.md`, per-class IoU ของ pylon/herbaceous > 0

## ถ้าพัง
- micromamba โหลดไม่ได้ → รันเซลล์ 3 ซ้ำ (เครือข่ายชั่วคราว)
- flash-attn ใน env ยังพยายาม compile → ติดตั้ง wheel ตรง: `/content/env/bin/pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
- `CUDA driver version is insufficient` → runtime นี้ driver เก่าเกิน (ไม่น่าเกิดบน L4) เปลี่ยน runtime ใหม่

## เสร็จแล้ว
**Runtime → Disconnect and delete runtime** เพื่อหยุดนับ CU""")

here = os.path.dirname(os.path.abspath(__file__))
env_sh = open(os.path.join(here, "..", "scripts", "colab_env.sh")).read()
train_py = open(os.path.join(here, "..", "scripts", "smoke_train.py")).read()
for c in cells:
    if c["cell_type"] == "code":
        c["source"] = c["source"].replace("__COLAB_ENV_SH__", env_sh).replace("__SMOKE_TRAIN_PY__", train_py)

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
                                  "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                  "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(here, "colab_smoke_litept.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
print("wrote", out, "with", len(cells), "cells")
