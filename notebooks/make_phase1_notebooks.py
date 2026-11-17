"""Generate the Phase-1 Colab notebooks:
  notebooks/colab_build_cache.ipynb  - cache all zones on Colab (CPU or L4 runtime), outputs to Drive
  notebooks/colab_train_litept.ipynb - resume-safe LitePT-S LiDAR-only training on Colab GPU
  notebooks/colab_train_upernet.ipynb - 2D branch: UPerNet training + per-image cache export
  notebooks/colab_fusion.ipynb        - fusion on the val zones (ImageVote first)
  notebooks/colab_train_fusion.ipynb  - fusion heads (late / concat / K×K window) on frozen backbones
  notebooks/colab_compact_cache2d.ipynb - one-time: pack the 2D cache into one compact tar per zone (CPU runtime)
  notebooks/colab_protocol.ipynb      - robustness protocol: 4 models under pose noise on the val zones
  notebooks/colab_heads_lab.ipynb     - many head variants trained at once on precomputed features (feature bank)
  notebooks/colab_leaderboard.ipynb   - test zones: 2D cache -> predictions -> leaderboard files (LiDAR-only + fused)
  notebooks/colab_label_transfer.ipynb - model-free label transfer under pose error, all images of the val zones (CPU)
  notebooks/colab_qualitative.ipynb   - predictions around pylons under pose error, for the qualitative figure (short GPU run)
  notebooks/colab_shared_error.ipynb  - the protocol with SHARED pose errors (one error for all images of a zone) on the heads of the paper
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_SH = open(os.path.join(HERE, "..", "scripts", "colab_env.sh")).read()


def nb(cells, gpu):
    meta = {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"},
            "colab": {"provenance": []}}
    if gpu:
        meta["accelerator"] = "GPU"; meta["colab"]["gpuType"] = gpu
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


# ----------------------------------------------------------------------------- cache notebook
cache_cells = [
md("""# GridNet-HD — build per-zone caches on Colab (Phase 1)

สร้าง `points.npz` (voxel 5 ซม., labels 11 class, geometric features, HAG) ของทุก zone แล้วเก็บลง Drive `GridNet/cache/<zone>/`
ไม่ต้องใช้ GPU: รันบน **CPU High-RAM runtime** (Runtime → Change runtime type → CPU, High-RAM; ~5–20 นาที/zone)
⚠️ Standard CPU runtime (12.7 GB) จะโดน OOM-kill กับ zone ที่มี raw points > ~70M (t4z1 t5a3 t6z2 t6z3 t6z4 t6z6) — ขึ้น `[fail]` โดยไม่มี traceback
Notebook ข้าม zone ที่มีอยู่แล้ว → รันซ้ำหลาย session ได้จนครบ 36 zones (train 21 + val 6 + test 9)

**Test 9 zones (สำหรับ leaderboard)** อยู่ท้ายรายการ: LAS ของ test ไม่มี field `ground_truth` → cache ได้ `labels` = 255 ทุกจุดและ `labeled = False`
(ต้องใช้ `gridnet_code.zip` ของ **update35 ขึ้นไป** — โค้ดก่อนหน้านั้นจะอ่าน field `classification` มาเป็น label; cell 2 ตรวจให้)
test zones ใหญ่กว่าโดยเฉลี่ย (รวม 767 ล้านจุดดิบ = เฉลี่ย 85 ล้านจุดต่อโซน เทียบกับ 63 ล้านของ train/val) → ต้องเป็น **High-RAM**;
เวลาที่วัดได้ (14 พ.ย. 2026, High-RAM 8 cores): 9 โซนใช้ 5.2 ชม. **เมื่อติ๊ก `WITH_ASSOC`** (10–72 นาทีต่อโซน); ไม่ติ๊ก (ค่าเริ่มต้น) ยังไม่ได้วัดกับ test zones · ไม่ใช้ GPU จึงแทบไม่เสีย compute units
**ไม่ต้องติ๊ก `WITH_ASSOC`**: ไม่มีขั้นตอนไหนใช้ `assoc.npz` ของ cache (notebook ที่ต้องใช้ association คำนวณเองบน GPU) — ติ๊กแล้วต้องโหลดภาพทั้งโซน คำนวณบน CPU และได้ไฟล์ 5–13 GB ต่อโซนบน Drive

**ก่อนรัน**: บน Mac `bash scripts/pack_code.sh` แล้วอัปโหลด `data/gridnet_code.zip` ไป Drive `GridNet/code/`"""),
code("""#@title 1. Mount Drive + parameters
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/GridNet'
CODE_ZIP = f'{DRIVE}/code/gridnet_code.zip'   #@param {type:"string"}
OUT = f'{DRIVE}/cache'
TEST_ZONES = ["t1z4","t1z5a","t1z7","t3z1","t3z2","t3z5","t5a2","t6z1","t6z5"]             # test: no labels (leaderboard)
ALL_ZONES = ["t1z5b","t1z8","t3z4","t4z1","t5b1","t5b5",                                   # val
             "t1z6a","t1z6b","t2z5","t3z3","t3z6","t3z7","t5a1","t5a3","t5a4","t5a5","t5b2",
             "t5b3","t5b4","t5b6","t5c1","t5c2","t5c3","t6z2","t6z3","t6z4","t6z6"] + TEST_ZONES   # train, then test
ZONES = ALL_ZONES   # or a subset, e.g. TEST_ZONES
WITH_ASSOC = False  #@param {type:"boolean"}
os.makedirs(OUT, exist_ok=True)
done = [z for z in ZONES if os.path.exists(f'{OUT}/{z}/points.npz')]
print(f"{len(done)}/{len(ZONES)} zones already cached:", done)
print("to do:", [z for z in ZONES if z not in done])
"""),
code("""#@title 2. Install laspy + unpack code
import subprocess, sys, zipfile, os
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laspy[lazrs]", "huggingface_hub"], check=True)
os.makedirs('/content/code', exist_ok=True)
zipfile.ZipFile(CODE_ZIP).extractall('/content/code')
print(os.listdir('/content/code'))
assert 'no ground_truth field' in open('/content/code/gridnet/io.py').read(), "gridnet_code.zip on Drive is older than update35: pack and upload it again (the test zones need it)"
"""),
code("""#@title 3. Download → cache → copy to Drive, one zone at a time (resumable)
import subprocess, sys, os, shutil, time
from huggingface_hub import snapshot_download
patterns = lambda z: [f"{z}/*"] if WITH_ASSOC else [f"{z}/lidar/*", f"{z}/pose/*"]
for z in ZONES:
    if os.path.exists(f'{OUT}/{z}/points.npz'):
        print(f"[skip] {z}"); continue
    t0 = time.time()
    snapshot_download("heig-vd-geo/GridNet-HD", repo_type="dataset", allow_patterns=patterns(z), local_dir="/content/data")
    print(f"[{z}] downloaded in {(time.time() - t0) / 60:.1f} min", flush=True)
    cmd = [sys.executable, "/content/code/scripts/build_cache.py", "--zone", f"/content/data/{z}", "--out", "/content/cache",
           "--voxel", "0.05", "--feat-grid", "0.10", "--k", "8"] + ([] if WITH_ASSOC else ["--no-assoc"])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[fail] {z}\\n", r.stdout[-1500:], r.stderr[-1500:]); continue
    os.makedirs(f'{OUT}/{z}', exist_ok=True)
    for f in os.listdir(f'/content/cache/{z}'):
        shutil.copy(f'/content/cache/{z}/{f}', f'{OUT}/{z}/{f}')
    shutil.rmtree(f'/content/data/{z}', ignore_errors=True); shutil.rmtree(f'/content/cache/{z}', ignore_errors=True)
    print(f"[{z}] done in {(time.time() - t0) / 60:.1f} min  ->  {OUT}/{z}", flush=True)
    print(open(f'{OUT}/{z}/report.txt').read().splitlines()[0])
print("finished", sum(os.path.exists(f'{OUT}/{z}/points.npz') for z in ZONES), "/", len(ZONES))
"""),
]

# ----------------------------------------------------------------------------- train notebook
train_cells = [
md("""# GridNet-HD — LitePT-S LiDAR-only training on Colab (Phase 1, resume-safe)

เป้าหมาย: เทรน baseline LiDAR-only ด้วย cache ของเราเอง (อ้างอิง: entry `Lite-S-test` บน leaderboard = 72.88 mIoU; entry นี้ไม่มี publication)
Runtime: **A100 80 GB** (เร็วสุด ~15 CU/ชม.) หรือ **L4 24 GB** (ถูกกว่า ช้ากว่า ~2.5 เท่า)

**VRAM**: LitePT-S ใช้ ~18 KB/จุด (รวม backward) → `BATCH × MAX_POINTS` ต้องไม่เกิน ~3.5M จุดบน A100 และ **~0.8M จุดบน L4**
| GPU | MAX_POINTS | EVAL_MAX_POINTS |
|---|---|---|
| A100 80 GB | 600000 (default) | 900000 (default) |
| L4 24 GB | **300000** | **500000** |

**LR**: ตาม recipe ของ LitePT (nuScenes) — blocks ใช้ 0.1× ของ base LR; base 1e-3 สำหรับ batch 2×600k
(รัน v1 ที่ 2e-3 ทุก parameter ลู่ออกเป็น NaN ที่ it ~2350) ถ้าเห็น `skipped N` ใน log บ่อย → ลด LR อีกครึ่ง

**Resume**: กด Run all ซ้ำได้ทุกครั้งที่ session หลุด — เทรนต่อจาก `last.pth` บน Drive อัตโนมัติ
ตั้ง `MAX_HOURS` ให้น้อยกว่าอายุ session เพื่อให้หยุดพร้อม checkpoint เอง · **เปลี่ยน LR/batch = เปลี่ยน RUN_NAME** (resume ข้าม config ไม่ได้)

**ก่อนรัน**: cache ครบ (notebook build_cache) และ `GridNet/code/gridnet_code.zip` บน Drive"""),
code("""#@title 1. GPU
import subprocess, torch
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
assert torch.cuda.get_device_capability()[0] >= 8, "need L4 / A100"
"""),
code("""#@title 2. Mount Drive + parameters
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/GridNet'
CODE_ZIP = f'{DRIVE}/code/gridnet_code.zip'      #@param {type:"string"}
CACHE_DRIVE = f'{DRIVE}/cache'
RUN_NAME = 'litept_s_lidar_v2'                    #@param {type:"string"}
CKPT_DIR = f'{DRIVE}/ckpt/{RUN_NAME}'
ITERS = 20000        #@param {type:"integer"}
MAX_HOURS = 6.5      #@param {type:"number"}
BATCH = 2            #@param {type:"integer"}
MAX_POINTS = 600000  #@param {type:"integer"}
EVAL_MAX_POINTS = 900000  #@param {type:"integer"}
RADIUS = 25.0        #@param {type:"number"}
LR = 0.001           #@param {type:"number"}
BLOCK_LR_MULT = 0.1  #@param {type:"number"}
EVAL_EVERY = 4000    #@param {type:"integer"}
ZONES_LIMIT = 0      #@param {type:"integer"}
os.makedirs(CKPT_DIR, exist_ok=True)
import torch
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
if BATCH * MAX_POINTS * 18e3 > 0.85 * vram_gb * 1e9:
    print(f"WARNING: {BATCH} x {MAX_POINTS:,} points needs ~{BATCH * MAX_POINTS * 18e-6:.0f} GB but this GPU has {vram_gb:.0f} GB -> lower MAX_POINTS")
print("checkpoints ->", CKPT_DIR)
"""),
code("""#@title 3. Environment (restore from Drive tarball if present, else build ~3 min; then save tarball)
import subprocess, os, time
ENV = '/content/env'; TAR = f'{DRIVE}/env/env_py310_torch241_cu124.tar.gz'
t0 = time.time()
if not os.path.exists(f'{ENV}/bin/python'):
    if os.path.exists(TAR):
        subprocess.run(['tar', '-C', '/content', '-xzf', TAR], check=True); print("env restored from Drive")
    else:
        open('/content/colab_env.sh', 'w').write(r'''__COLAB_ENV_SH__''')
        p = subprocess.Popen(['bash', '/content/colab_env.sh', ENV], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            if 'FutureWarning' not in line and 'def backward' not in line: print(line, end='')
        p.wait(); assert p.returncode == 0
        os.makedirs(f'{DRIVE}/env', exist_ok=True)
        subprocess.run(['tar', '-C', '/content', '-czf', TAR, 'env'], check=True); print("env saved to Drive")
PY = f'{ENV}/bin/python'
print(f"env ready in {(time.time() - t0) / 60:.1f} min")
"""),
code("""#@title 4. Code + LitePT
import subprocess, zipfile, os
os.makedirs('/content/code', exist_ok=True); zipfile.ZipFile(CODE_ZIP).extractall('/content/code')
if not os.path.exists('/content/LitePT'):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/prs-eth/LitePT.git", "/content/LitePT"], check=True)
r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0,'/content/LitePT'); sys.path.insert(0,'/content/code'); "
                    "from litept.model import LitePT; import gridnet.train.engine; print('imports ok')"], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-1500:])
"""),
code("""#@title 5. Copy caches Drive → local disk (skips existing; ~1.8 GB per zone). VAL_ONLY = only the 6 val zones (enough for cells 7-8; ~11 GB instead of ~46 GB)
VAL_ONLY = False   #@param {type:"boolean"}
import os, shutil, time
TRAIN = ["t1z6a","t1z6b","t2z5","t3z3","t3z6","t3z7","t5a1","t5a3","t5a4","t5a5","t5b2","t5b3","t5b4","t5b6","t5c1","t5c2","t5c3","t6z2","t6z3","t6z4","t6z6"]
VAL = ["t1z5b","t1z8","t3z4","t4z1","t5b1","t5b5"]
want = [z for z in TRAIN + VAL if os.path.exists(f'{CACHE_DRIVE}/{z}/points.npz')]
if VAL_ONLY: want = [z for z in want if z in VAL]
if ZONES_LIMIT: want = [z for z in want if z in VAL][:2] + [z for z in want if z in TRAIN][:ZONES_LIMIT]
missing = [z for z in TRAIN + VAL if z not in want]
print(f"using {len(want)} zones; missing from Drive: {missing}")
t0 = time.time()
for z in want:
    dst = f'/content/cache/{z}/points.npz'
    if not os.path.exists(dst):
        os.makedirs(f'/content/cache/{z}', exist_ok=True); shutil.copy(f'{CACHE_DRIVE}/{z}/points.npz', dst)
print(f"caches ready in {(time.time() - t0) / 60:.1f} min")
"""),
code("""#@title 6. Train (resumes automatically from last.pth)
import subprocess, time, os
cmd = [PY, '/content/code/scripts/train_litept.py', '--cache', '/content/cache', '--ckpt-dir', CKPT_DIR, '--litept', '/content/LitePT',
       '--iters', str(ITERS), '--max-hours', str(MAX_HOURS), '--batch', str(BATCH), '--max-points', str(MAX_POINTS),
       '--eval-max-points', str(EVAL_MAX_POINTS), '--radius', str(RADIUS), '--eval-every', str(EVAL_EVERY),
       '--lr', str(LR), '--block-lr-mult', str(BLOCK_LR_MULT)]
if ZONES_LIMIT:
    present = sorted(os.listdir('/content/cache'))
    cmd += ['--train-zones', ','.join(z for z in present if z not in VAL), '--val-zones', ','.join(z for z in present if z in VAL)]
env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
t0 = time.time()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
try:
    for line in p.stdout:
        if 'FutureWarning' not in line and 'def backward' not in line and '_TORCH_CUSTOM_FWD' not in line: print(line, end='')
    p.wait(); print(f"exit {p.returncode} after {(time.time() - t0) / 3600:.2f} h")
except KeyboardInterrupt:
    # the stop button only interrupts this cell: kill the trainer too, or a second run would share the GPU with it
    p.terminate(); p.wait()
    print(f"stopped by user after {(time.time() - t0) / 60:.1f} min; last.pth on Drive is kept -> re-run this cell to resume")
"""),
code("""#@title 7. Evaluate best checkpoint on the val zones
import subprocess, json, os
env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
cmd = [PY, '/content/code/scripts/train_litept.py', '--cache', '/content/cache', '--ckpt-dir', CKPT_DIR, '--litept', '/content/LitePT',
       '--radius', str(RADIUS), '--eval-max-points', str(EVAL_MAX_POINTS), '--eval-only']
if ZONES_LIMIT:
    present = sorted(os.listdir('/content/cache'))
    cmd += ['--train-zones', ','.join(z for z in present if z not in VAL), '--val-zones', ','.join(z for z in present if z in VAL)]
r = subprocess.run(cmd, capture_output=True, text=True, env=env)
print('\\n'.join(l for l in r.stdout.splitlines() if 'Warning' not in l)[-3000:], r.stderr[-800:])
if os.path.exists(f'{CKPT_DIR}/eval.json'):
    print(json.dumps(json.load(open(f'{CKPT_DIR}/eval.json')), indent=1))
else:
    print("no eval.json -> training did not produce a checkpoint; see cell 6")
"""),
code("""#@title 8. Figures from best.pth: mIoU per zone + confusion matrix + qualitative renders (~20-30 min on A100; needs cells 1-5)
import subprocess, os, sys, time
QUAL_ZONES = "t5b5,t4z1"   #@param {type:"string"}
PRED = f'{CKPT_DIR}/pred'; FIG = f'{CKPT_DIR}/figures'; os.makedirs(FIG, exist_ok=True)
env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
def run(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in p.stdout:
        if 'Warning' not in line and 'def backward' not in line and '_TORCH_CUSTOM_FWD' not in line: print(line, end='')
    p.wait(); return p.returncode
t0 = time.time()
cmd = [PY, '/content/code/scripts/train_litept.py', '--cache', '/content/cache', '--ckpt-dir', CKPT_DIR, '--litept', '/content/LitePT',
       '--radius', str(RADIUS), '--eval-max-points', str(EVAL_MAX_POINTS), '--eval-only', '--save-pred', PRED]
if ZONES_LIMIT:
    present = sorted(os.listdir('/content/cache'))
    cmd += ['--train-zones', ','.join(z for z in present if z not in VAL), '--val-zones', ','.join(z for z in present if z in VAL)]
assert run(cmd) == 0, "eval failed (see output above)"
# the figure scripts need only numpy + matplotlib -> run them with Colab's own python
run([sys.executable, '/content/code/scripts/plot_eval.py', '--eval', f'{CKPT_DIR}/eval.json', '--out', f'{FIG}/eval_val.png'])
for z in [s.strip() for s in QUAL_ZONES.split(',') if s.strip()]:
    if os.path.exists(f'{PRED}/{z}_pred.npy'):
        run([sys.executable, '/content/code/scripts/render_predictions.py', '--cache', '/content/cache', '--zone', z,
             '--pred', f'{PRED}/{z}_pred.npy', '--out', f'{FIG}/qual_{z}'])
    else:
        print(f"no prediction for {z} (not a val zone or not cached)")
print(f"done in {(time.time() - t0) / 60:.1f} min -> figures on Drive: {FIG}")
"""),
md("""## หมายเหตุ
- `ZONES_LIMIT > 0` = โหมดทดสอบ: ใช้ train zones แค่ N zone + val 2 zone (เร็ว ใช้ตรวจ pipeline ก่อนรันเต็ม)
- loss/val mIoU ทุกครั้งถูกเขียนที่ `<CKPT_DIR>/log.csv`; config ที่ `config.json`
- จบแล้ว **Disconnect and delete runtime**"""),
]

# ----------------------------------------------------------------------------- 2D branch notebook
upernet_cells = [
md("""# GridNet-HD — 2D branch: UPerNet training + per-image cache export (Phase 1, resume-safe)

เป้าหมาย: เทรน UPerNet (ConvNeXt-T encoder) บน image masks ของ train zones → วัด 2D mIoU บน val crops →
export cache ต่อภาพ (class probabilities @1/4 res + 32-ch features @1/8 res, fp16, ~50 MB/ภาพ) ลง Drive `GridNet/cache2d/<zone>/`

Runtime: **L4** (พอ) หรือ A100 (เร็วกว่า ~2 เท่า) · T4 ใช้ได้แต่ต้อง `CROP 768, BATCH 2`

| GPU | CROP | BATCH | ~s/it (โดยประมาณ) |
|---|---|---|---|
| A100 40/80 GB | 1024 | 4 | 0.4–0.6 (ถ้า WORKERS 8) |
| L4 24 GB | 1024 | 2 | 0.8 |
| T4 16 GB | 768 | 2 | 1.2 |

ถ้าเจอ CUDA out of memory ใน cell 6 → ลด BATCH หรือ CROP แล้ว **เปลี่ยน RUN_NAME** · v1 (8 zones) ImageVote ได้ 53.5 บน val เพราะไม่เห็นพื้นที่ t1/t4 ตอนเทรน → v2 ใช้ครบ 21 zones

ภาพ+mask ถูกโหลดจาก Hugging Face ครั้งแรกแล้วเก็บสำเนาไว้บน Drive `GridNet/raw2d/<zone>/` (session ถัดไป copy จาก Drive)
**Resume**: Run all ซ้ำได้เมื่อ session หลุด — เทรนต่อจาก `last.pth` · เปลี่ยน encoder/crop = เปลี่ยน RUN_NAME"""),
code("""#@title 1. GPU
import subprocess, torch
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
assert torch.cuda.is_available(), "no GPU runtime"
"""),
code("""#@title 2. Mount Drive + parameters
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/GridNet'
CODE_ZIP = f'{DRIVE}/code/gridnet_code.zip'      #@param {type:"string"}
RAW_DRIVE = f'{DRIVE}/raw2d'                     # images + masks copied here after the first download
RUN_NAME = 'upernet_convnext_t_v2'               #@param {type:"string"}
CKPT_DIR = f'{DRIVE}/ckpt/{RUN_NAME}'
CACHE2D_DRIVE = f'{DRIVE}/cache2d/{RUN_NAME}'    # one 2D cache per 2D model (v1 is at cache2d/<zone>, legacy)
TRAIN_ZONES = 't1z6a,t1z6b,t2z5,t3z3,t3z6,t3z7,t5a1,t5a3,t5a4,t5a5,t5b2,t5b3,t5b4,t5b6,t5c1,t5c2,t5c3,t6z2,t6z3,t6z4,t6z6'   #@param {type:"string"}
VAL_ZONES = 't5b5,t1z8,t4z1'                     #@param {type:"string"}
IMAGES_PER_ZONE = 200   #@param {type:"integer"}
ENCODER = 'convnext_tiny'   #@param {type:"string"}
CROP = 1024             #@param {type:"integer"}
BATCH = 4               #@param {type:"integer"}
WORKERS = 8             #@param {type:"integer"}
ITERS = 12000           #@param {type:"integer"}
MAX_HOURS = 4.0         #@param {type:"number"}
LR = 0.001              #@param {type:"number"}
EVAL_EVERY = 1000       #@param {type:"integer"}
EXPORT_ZONES = 't1z6a,t1z6b,t2z5,t3z3,t3z6,t3z7,t5a1,t5a3,t5a4,t5a5,t5b2,t5b3,t5b4,t5b6,t5c1,t5c2,t5c3,t6z2,t6z3,t6z4,t6z6'  #@param {type:"string"}
os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(RAW_DRIVE, exist_ok=True); os.makedirs(CACHE2D_DRIVE, exist_ok=True)
import torch
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
if vram_gb < 20 and (CROP > 768 or BATCH > 2):
    print(f"WARNING: {vram_gb:.0f} GB GPU -> use CROP 768, BATCH 2")
elif vram_gb < 30 and BATCH > 2:
    print(f"WARNING: {vram_gb:.0f} GB GPU -> BATCH 2 is the safe value at CROP 1024")
print("checkpoints ->", CKPT_DIR)
"""),
code("""#@title 3. Environment (restore tarball from Drive or build; + timm)
import subprocess, os, time
ENV = '/content/env'; TAR = f'{DRIVE}/env/env_py310_torch241_cu124.tar.gz'
t0 = time.time()
if not os.path.exists(f'{ENV}/bin/python'):
    if os.path.exists(TAR):
        subprocess.run(['tar', '-C', '/content', '-xzf', TAR], check=True); print("env restored from Drive")
    else:
        open('/content/colab_env.sh', 'w').write(r\'\'\'__COLAB_ENV_SH__\'\'\')
        p = subprocess.Popen(['bash', '/content/colab_env.sh', ENV], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            if 'FutureWarning' not in line: print(line, end='')
        p.wait(); assert p.returncode == 0
        os.makedirs(f'{DRIVE}/env', exist_ok=True)
        subprocess.run(['tar', '-C', '/content', '-czf', TAR, 'env'], check=True); print("env saved to Drive")
PY = f'{ENV}/bin/python'
subprocess.run([PY, '-m', 'pip', 'install', '-q', 'timm', 'huggingface_hub', 'pillow'], check=True)
print(f"env ready in {(time.time() - t0) / 60:.1f} min")
"""),
code("""#@title 4. Code
import subprocess, zipfile, os
os.makedirs('/content/code', exist_ok=True); zipfile.ZipFile(CODE_ZIP).extractall('/content/code')
r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0,'/content/code'); import gridnet.train.engine2d, timm; print('imports ok, timm', timm.__version__)"],
                   capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-1500:])
"""),
code("""#@title 5. Images + masks for the train/val zones (Drive copy if present, else Hugging Face -> Drive)
import os, shutil, subprocess, time
DATA = '/content/data'; os.makedirs(DATA, exist_ok=True)
def fetch(zones):
    t0 = time.time()
    for z in zones:
        dst = f'{DATA}/{z}'
        if os.path.isdir(f'{dst}/images') and os.path.isdir(f'{dst}/masks'):
            continue
        src = f'{RAW_DRIVE}/{z}'
        if os.path.isdir(f'{src}/images') and os.path.isdir(f'{src}/masks'):
            shutil.copytree(src, dst, dirs_exist_ok=True); print(f"{z}: copied from Drive")
        else:
            code = ("import os,sys; sys.path.insert(0,'/content/code'); from gridnet.train.data2d import fetch_zones; "
                    f"fetch_zones(['{z}'], '{DATA}')")
            env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER='0')
            try:
                from google.colab import userdata; env['HF_TOKEN'] = userdata.get('HF_TOKEN')   # optional (public dataset)
            except Exception: pass
            subprocess.run([PY, '-c', code], check=True, env=env); print(f"{z}: downloaded from Hugging Face")
            shutil.copytree(dst, src, dirs_exist_ok=True); print(f"{z}: copy saved to Drive")
        n = len(os.listdir(f'{dst}/images')); print(f"   {n} images  ({(time.time() - t0) / 60:.1f} min)")
zones = [s.strip() for s in (TRAIN_ZONES + ',' + VAL_ZONES).split(',') if s.strip()]
fetch(zones)
print(subprocess.run(['du', '-sh', DATA], capture_output=True, text=True).stdout.strip())
"""),
code("""#@title 6. Train (resumes automatically from last.pth)
import subprocess, time, os
cmd = [PY, '/content/code/scripts/train_upernet.py', '--data', DATA, '--ckpt-dir', CKPT_DIR,
       '--train-zones', TRAIN_ZONES, '--val-zones', VAL_ZONES, '--images-per-zone', str(IMAGES_PER_ZONE),
       '--encoder', ENCODER, '--crop', str(CROP), '--batch', str(BATCH), '--iters', str(ITERS), '--lr', str(LR),
       '--eval-every', str(EVAL_EVERY), '--max-hours', str(MAX_HOURS), '--workers', str(WORKERS)]
env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore')
t0 = time.time()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
try:
    for line in p.stdout:
        if 'Warning' not in line: print(line, end='')
    p.wait(); print(f"exit {p.returncode} after {(time.time() - t0) / 3600:.2f} h")
except KeyboardInterrupt:
    p.terminate(); p.wait()
    print(f"stopped by user after {(time.time() - t0) / 60:.1f} min; last.pth on Drive is kept -> re-run this cell to resume")
"""),
code("""#@title 7. Evaluate best checkpoint on the fixed val crops
import subprocess, json, os
env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore')
cmd = [PY, '/content/code/scripts/train_upernet.py', '--data', DATA, '--ckpt-dir', CKPT_DIR, '--val-zones', VAL_ZONES,
       '--encoder', ENCODER, '--crop', str(CROP), '--batch', str(BATCH), '--eval-only']
r = subprocess.run(cmd, capture_output=True, text=True, env=env)
print('\\n'.join(l for l in r.stdout.splitlines() if 'Warning' not in l)[-2000:], r.stderr[-800:])
if os.path.exists(f'{CKPT_DIR}/eval2d.json'):
    e = json.load(open(f'{CKPT_DIR}/eval2d.json')); print({k: e[k] for k in ('ckpt', 'it', 'mIoU', 'OA')}); print(e['per_class'])
"""),
code("""#@title 8. Export the per-image 2D cache for EXPORT_ZONES -> Drive cache2d/<RUN_NAME> (resumable; compact uint8/int8 ~24 MB/image; ~1.7 s/image on A100, ~3-4 s on L4)
import subprocess, time, os
zones = [s.strip() for s in EXPORT_ZONES.split(',') if s.strip()]
fetch(zones)                                   # images of the export zones (cell 5 must have run)
env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore')
cmd = [PY, '/content/code/scripts/train_upernet.py', '--data', DATA, '--ckpt-dir', CKPT_DIR, '--encoder', ENCODER,
       '--crop', str(CROP), '--batch', str(BATCH), '--export-cache', CACHE2D_DRIVE, '--export-zones', ','.join(zones)]
t0 = time.time()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
try:
    for line in p.stdout:
        if 'Warning' not in line: print(line, end='')
    p.wait(); print(f"exit {p.returncode} after {(time.time() - t0) / 60:.1f} min")
except KeyboardInterrupt:
    p.terminate(); p.wait(); print("stopped; re-run to continue (existing files are skipped)")
"""),
md("""## หมายเหตุ
- cache ต่อภาพ: `prob` (11, H/4, W/4) fp16 = softmax ที่ความละเอียด output จริงของ UPerNet · `feat` (32, H/8, W/8) fp16 = bottleneck feature เฉลี่ย 2×2
- val mIoU ในนี้วัดบน crop 1024² ที่สุ่มแบบตายตัวจาก val images (ครึ่งหนึ่งมีคลาสโครงสร้างอยู่กลาง) ไม่ใช่ทั้งภาพ — ใช้เลือก checkpoint เท่านั้น
- จบแล้ว **Disconnect and delete runtime**"""),
]

# ----------------------------------------------------------------------------- fusion notebook (ImageVote first)
fusion_cells = [
md("""# GridNet-HD — Fusion on the validation zones: ImageVote (Phase 1)

ต้องมีก่อน: `GridNet/cache/<zone>/points.npz` (LiDAR cache) และ `GridNet/cache2d/<zone>/*.npz` (2D cache จาก colab_train_upernet cell 8)

ทำอะไร: GPU association (projection → z-buffer visibility τ 0.5 m → 8 views ใกล้สุดต่อจุด) → เฉลี่ย class probabilities จาก 2D cache ต่อจุด → mIoU บน val zones
ผลลัพธ์ไป Drive `GridNet/fusion/imagevote/`: `eval.json`, ต่อโซน `imagevote_pred.npy`, `img_prob.npy` (fp16, ใช้ต่อใน late fusion), `assoc.npz` (association table ใช้ซ้ำได้)

Runtime: L4 พอ (association ~2–5 นาที/โซน) · A100 เร็วกว่า · RAM: ใช้ High-RAM ถ้ามีให้เลือก (t4z1 80 M จุด)"""),
code("""#@title 1. GPU
import subprocess, torch
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
assert torch.cuda.is_available(), "no GPU runtime"
"""),
code("""#@title 2. Mount Drive + parameters
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/GridNet'
CODE_ZIP = f'{DRIVE}/code/gridnet_code.zip'      #@param {type:"string"}
CACHE_DRIVE = f'{DRIVE}/cache'
RUN2D = 'upernet_convnext_t_v2'                  #@param {type:"string"}
CACHE2D_DRIVE = f'{DRIVE}/cache2d/{RUN2D}'       # 2D cache of that UPerNet run (v1 legacy path: cache2d/<zone>)
OUT = f'{DRIVE}/fusion/imagevote_{RUN2D}'
ZONES = 't1z5b,t1z8,t3z4,t4z1,t5b1,t5b5'         #@param {type:"string"}
K = 8                   #@param {type:"integer"}
TAU = 0.5               #@param {type:"number"}
CHUNK = 20000000        #@param {type:"integer"}
SAVE_ASSOC = True       #@param {type:"boolean"}
COPY_CACHE2D_LOCAL = False  #@param {type:"boolean"}
os.makedirs(OUT, exist_ok=True)
zones = [z.strip() for z in ZONES.split(',') if z.strip()]
print("zones:", zones, "-> results in", OUT)
"""),
code("""#@title 3. Environment (restore tarball from Drive)
import subprocess, os, time
ENV = '/content/env'; TAR = f'{DRIVE}/env/env_py310_torch241_cu124.tar.gz'
t0 = time.time()
if not os.path.exists(f'{ENV}/bin/python'):
    assert os.path.exists(TAR), "no env tarball on Drive: run colab_train_litept cell 3 once"
    subprocess.run(['tar', '-C', '/content', '-xzf', TAR], check=True); print("env restored from Drive")
PY = f'{ENV}/bin/python'
subprocess.run([PY, '-m', 'pip', 'install', '-q', 'huggingface_hub', 'pillow'], check=True)
print(f"env ready in {(time.time() - t0) / 60:.1f} min")
"""),
code("""#@title 4. Code
import subprocess, zipfile, os
os.makedirs('/content/code', exist_ok=True); zipfile.ZipFile(CODE_ZIP).extractall('/content/code')
r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0,'/content/code'); import gridnet.train.assoc_gpu, gridnet.train.imagevote; print('imports ok')"],
                   capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-1500:])
"""),
code("""#@title 5. Data: LiDAR caches -> local disk, camera pose files from Hugging Face, (optional) 2D cache -> local disk
import os, shutil, subprocess, time
DATA = '/content/data'; os.makedirs(DATA, exist_ok=True); os.makedirs('/content/cache', exist_ok=True)
t0 = time.time()
for z in zones:
    dst = f'/content/cache/{z}/points.npz'
    if not os.path.exists(dst):
        os.makedirs(f'/content/cache/{z}', exist_ok=True); shutil.copy(f'{CACHE_DRIVE}/{z}/points.npz', dst)
print(f"LiDAR caches ready in {(time.time() - t0) / 60:.1f} min")
code = ("import sys; sys.path.insert(0,'/content/code'); from gridnet.train.data2d import fetch_zones; "
        f"fetch_zones({zones!r}, '{DATA}', what=('pose',))")
subprocess.run([PY, '-c', code], check=True)
print("pose files:", {z: sorted(os.listdir(f'{DATA}/{z}/pose')) for z in zones})
CACHE2D = CACHE2D_DRIVE
if COPY_CACHE2D_LOCAL:
    CACHE2D = '/content/cache2d'; t0 = time.time()
    for z in zones:
        shutil.copytree(f'{CACHE2D_DRIVE}/{z}', f'{CACHE2D}/{z}', dirs_exist_ok=True)
    print(f"2D cache copied in {(time.time() - t0) / 60:.1f} min")
"""),
code("""#@title 6. ImageVote on the validation zones
import subprocess, time, os
cmd = [PY, '/content/code/scripts/eval_imagevote.py', '--cache', '/content/cache', '--data', DATA, '--cache2d', CACHE2D,
       '--zones', ','.join(zones), '--out', OUT, '--k', str(K), '--tau', str(TAU), '--chunk', str(CHUNK)]
if SAVE_ASSOC: cmd.append('--save-assoc')
env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore')
t0 = time.time()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
try:
    for line in p.stdout:
        if 'Warning' not in line: print(line, end='')
    p.wait(); print(f"exit {p.returncode} after {(time.time() - t0) / 60:.1f} min")
except KeyboardInterrupt:
    p.terminate(); p.wait(); print("stopped")
"""),
code("""#@title 7. Figure: confusion matrix + mIoU per zone (ImageVote)
import subprocess, sys, os
from IPython.display import Image, display
fig = f'{OUT}/imagevote_val.png'
r = subprocess.run([sys.executable, '/content/code/scripts/plot_eval.py', '--eval', f'{OUT}/eval.json', '--out', fig,
                    '--title', 'ImageVote (UPerNet ConvNeXt-T, 8 nearest views) — validation zones'],
                   capture_output=True, text=True, env=dict(os.environ, PYTHONPATH='/content/code'))
print(r.stdout.strip(), r.stderr[-500:])
display(Image(fig)) if os.path.exists(fig) else None
"""),
md("""## หมายเหตุ
- จุดที่ไม่มี view ที่มองเห็นเลย (ปกติ < 1–2 %) ถูกทายเป็น herbaceous; `eval.json` รายงานทั้ง mIoU รวม และ mIoU เฉพาะจุดที่มี view
- `assoc.npz` ต่อโซน ~1 GB ต่อ 8 ล้านจุด (t4z1 ≈ 10 GB) — ปิด SAVE_ASSOC ได้ถ้าไม่ต้องการ
- จบแล้ว **Disconnect and delete runtime**"""),
]

# ----------------------------------------------------------------------------- fusion-head training notebook
fusion_train_cells = [
md("""# GridNet-HD — Fusion heads on frozen backbones (Phase 1: late fusion, concat, K×K window attention)

ต้องมีก่อน: LiDAR cache 27 zones (`GridNet/cache`), checkpoint LiDAR-only (`GridNet/ckpt/litept_s_lidar_v2/best.pth`),
2D cache ของ UPerNet v2 สำหรับ **val 6 zones + train zones ที่จะใช้** (`GridNet/cache2d/upernet_convnext_t_v2/<zone>/`,
จาก colab_train_upernet cell 8) และ **tar ต่อโซน** `<zone>.tar` จาก `colab_compact_cache2d` (รันครั้งเดียวบน CPU runtime) —
ถ้าโซนไหนยังไม่มี tar, cell 5 จะ copy ทีละไฟล์แทน (ช้ากว่ามาก: Drive ให้ ~10–15 MB/s ต่อไฟล์เล็ก แต่ >100 MB/s กับไฟล์ใหญ่ไฟล์เดียว)

ทำอะไร: เทรน head ตัวเล็กบน feature ของ frozen backbones (LitePT-S 72-d + UPerNet prob/feat ผ่าน association 8 views) แบบ zone-major
(โหลด 2D maps ของโซนขึ้น GPU ทีละโซน) → eval บน val 6 zones ทุก pass → `eval/eval.json` + `fused_pred.npy` ต่อโซน
LiDAR-only ของ backbone ถูกวัดบน window ชุดเดียวกันไปด้วย (sanity: ต้อง ≈ 69.9)

| HEAD | input | ~เวลาต่อ head บน A100 (8 train zones, 2 passes, eval 2 ครั้ง) |
|---|---|---|
| late | softmax(LiDAR) ⊕ image probs | ~1.7 ชม. (late_v1: เทรน 0.6 s/it, eval ~20 นาที/ครั้ง) |
| concat | LiDAR feat ⊕ image feat ⊕ probs | ~1.8 ชม. |
| window (K=3/5) | เหมือน concat แต่ attention บน K×K cells | ~2 ชม. |

เตรียมข้อมูล (cell 5) ~45 นาทีต่อ session → **รันหลาย head ใน session เดียว** ด้วย `EXTRA_RUNS` เช่น `window:3:window3_v1`
(รูปแบบ `head:K:run_name[:pose_t:pose_r[:seed]]` คั่นด้วย `,`) · `SHUTDOWN_WHEN_DONE` = ปิด runtime เองเมื่อจบทุก run (ไม่เสีย units ตอนเครื่องว่าง)

**Pose-noise augmentation** (reference remedy): `POSE_NOISE_T = 0.25` (m), `POSE_NOISE_R = 0.25` (deg) → ต้องเปลี่ยน `RUN_NAME`
**Resume**: Run all ซ้ำได้ — แต่ละ run ต่อจาก `last.pth` ของตัวเอง (run ที่จบแล้วจะผ่านไปทันที)"""),
code("""#@title 1. GPU
import subprocess, torch
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
assert torch.cuda.get_device_capability()[0] >= 8, "need L4 / A100 (flash-attn)"
"""),
code("""#@title 2. Mount Drive + parameters
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/GridNet'
CODE_ZIP = f'{DRIVE}/code/gridnet_code.zip'      #@param {type:"string"}
HEAD = 'concat'           #@param ["late", "concat", "window"]
K = 3                     #@param {type:"integer"}
RUN_NAME = 'concat_v1'    #@param {type:"string"}
RUN2D = 'upernet_convnext_t_v2'                  #@param {type:"string"}
BACKBONE_CKPT = f'{DRIVE}/ckpt/litept_s_lidar_v2/best.pth'   #@param {type:"string"}
TRAIN_ZONES = 't1z6a,t2z5,t3z3,t5a1,t5b2,t5c1,t6z2,t6z4'      #@param {type:"string"}
VAL_ZONES = 't1z5b,t1z8,t3z4,t4z1,t5b1,t5b5'                  #@param {type:"string"}
PASSES = 2                #@param {type:"integer"}
ITS_PER_ZONE = 300        #@param {type:"integer"}
HEAD_POINTS = 120000      #@param {type:"integer"}
MAX_POINTS = 600000       #@param {type:"integer"}
EVAL_MAX_POINTS = 900000  #@param {type:"integer"}
LR = 0.001                #@param {type:"number"}
POSE_NOISE_T = 0.0        #@param {type:"number"}
POSE_NOISE_R = 0.0        #@param {type:"number"}
AUG_PROB = 0.5            #@param {type:"number"}
SEED = 0                  #@param {type:"integer"}
MAX_HOURS = 5.5           #@param {type:"number"}
EXTRA_RUNS = ''           #@param {type:"string"}
SHUTDOWN_WHEN_DONE = False #@param {type:"boolean"}
COPY_CACHE2D_LOCAL = True #@param {type:"boolean"}
CACHE_DRIVE = f'{DRIVE}/cache'
CACHE2D_DRIVE = f'{DRIVE}/cache2d/{RUN2D}'
ASSOC_DIR = f'{DRIVE}/fusion/assoc_{RUN2D}'      # <zone>/assoc.npz archive (GPU sessions rebuild the table: faster than reading it)
IMAGEVOTE_DIR = f'{DRIVE}/fusion/imagevote_{RUN2D}'   # val assoc.npz already saved there by colab_fusion
work_of = lambda name: f'{DRIVE}/fusion/{name}'
# the queue: the run of the form fields first, then EXTRA_RUNS = 'head:K:run_name[:pose_t:pose_r[:seed]], ...'
runs = [dict(head=HEAD, k=int(K), name=RUN_NAME, pt=float(POSE_NOISE_T), pr=float(POSE_NOISE_R), seed=int(SEED))]
for spec in [x.strip() for x in EXTRA_RUNS.split(',') if x.strip()]:
    f = [x.strip() for x in spec.split(':')]
    assert len(f) >= 3 and f[0] in ('late', 'concat', 'window'), f"bad EXTRA_RUNS entry {spec!r}: use head:K:run_name[:pose_t:pose_r[:seed]]"
    runs.append(dict(head=f[0], k=int(f[1]), name=f[2], pt=float(f[3]) if len(f) > 3 else 0.0,
                     pr=float(f[4]) if len(f) > 4 else 0.0, seed=int(f[5]) if len(f) > 5 else int(SEED)))
assert len({r['name'] for r in runs}) == len(runs), "every run needs its own run_name"
for d in [ASSOC_DIR] + [work_of(r['name']) for r in runs]: os.makedirs(d, exist_ok=True)
train_zones = [z.strip() for z in TRAIN_ZONES.split(',') if z.strip()]
val_zones = [z.strip() for z in VAL_ZONES.split(',') if z.strip()]
import torch
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
if any(r['head'] == 'window' and r['k'] >= 5 for r in runs) and vram_gb < 30:
    print(f"WARNING: window K>=5 on a {vram_gb:.0f} GB GPU -> set HEAD_POINTS 60000 or use A100")
for i, r in enumerate(runs):
    done = os.path.exists(f"{work_of(r['name'])}/eval/eval.json") and os.path.exists(f"{work_of(r['name'])}/best.pth")
    print(f"run {i + 1}/{len(runs)}: {r['name']}  head {r['head']}" + (f" K={r['k']}" if r['head'] == 'window' else '')
          + (f"  pose noise {r['pt']} m / {r['pr']} deg" if r['pt'] or r['pr'] else '') + f"  seed {r['seed']}"
          + ("   (has results on Drive: resumes / skips)" if done else ''))
"""),
code("""#@title 3. Environment (restore tarball from Drive)
import subprocess, os, time
ENV = '/content/env'; TAR = f'{DRIVE}/env/env_py310_torch241_cu124.tar.gz'
t0 = time.time()
if not os.path.exists(f'{ENV}/bin/python'):
    assert os.path.exists(TAR), "no env tarball on Drive: run colab_train_litept cell 3 once"
    subprocess.run(['tar', '-C', '/content', '-xzf', TAR], check=True); print("env restored from Drive")
PY = f'{ENV}/bin/python'
subprocess.run([PY, '-m', 'pip', 'install', '-q', 'huggingface_hub', 'pillow'], check=True)
print(f"env ready in {(time.time() - t0) / 60:.1f} min")
"""),
code("""#@title 4. Code + LitePT
import subprocess, zipfile, os
os.makedirs('/content/code', exist_ok=True); zipfile.ZipFile(CODE_ZIP).extractall('/content/code')
if not os.path.exists('/content/LitePT'):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/prs-eth/LitePT.git", "/content/LitePT"], check=True)
r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0,'/content/LitePT'); sys.path.insert(0,'/content/code'); "
                    "from litept.model import LitePT; import gridnet.train.engine_fusion; print('imports ok')"], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-1500:])
"""),
code("""#@title 5. Data -> local disk: LiDAR caches, 2D caches (zone tars), camera files, backbone checkpoint, existing val associations
import os, sys, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/content/code')
from gridnet.train.maps2d import cached_image_ids, zone_tar_manifest, extract_zone_tar, write_compact
DATA = '/content/data'; os.makedirs(DATA, exist_ok=True)
zones = train_zones + val_zones
t0 = time.time()
for z in zones:
    dst = f'/content/cache/{z}/points.npz'
    if not os.path.exists(dst):
        os.makedirs(f'/content/cache/{z}', exist_ok=True); shutil.copy(f'{CACHE_DRIVE}/{z}/points.npz', dst)
print(f"LiDAR caches ready ({(time.time() - t0) / 60:.1f} min)")
missing = [z for z in zones if zone_tar_manifest(CACHE2D_DRIVE, z) is None and not os.path.isdir(f'{CACHE2D_DRIVE}/{z}')]
assert not missing, f"no 2D cache for {missing} in {CACHE2D_DRIVE}: export them with colab_train_upernet cell 8"
CACHE2D = CACHE2D_DRIVE
if COPY_CACHE2D_LOCAL:
    CACHE2D = '/content/cache2d'; t0 = time.time(); slow = []
    for z in zones:
        if zone_tar_manifest(CACHE2D_DRIVE, z, wait=30) is not None:   # one big sequential read: fast
            n = extract_zone_tar(CACHE2D_DRIVE, z, CACHE2D); how = 'tar'
        else:                                                             # no tar yet: parallel per-file copy (slow)
            ids = cached_image_ids(CACHE2D_DRIVE, z)
            with ThreadPoolExecutor(16) as ex:
                list(ex.map(lambda i: write_compact(CACHE2D_DRIVE, z, i, CACHE2D), ids))
            n = len(cached_image_ids(CACHE2D, z)); how = 'files'; slow.append(z)
        print(f"  {z}: {n} images from {how}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(subprocess.run(['du', '-sh', CACHE2D], capture_output=True, text=True).stdout.strip())
    if slow: print("zones without a tar (slow path) - run colab_compact_cache2d once for:", ','.join(slow))
code = ("import sys; sys.path.insert(0,'/content/code'); from gridnet.train.data2d import fetch_zones; "
        f"fetch_zones({zones!r}, '{DATA}', what=('pose',))")
subprocess.run([PY, '-c', code], check=True); print("camera files ready")
CKPT_LOCAL = '/content/backbone_best.pth'
if not os.path.exists(CKPT_LOCAL): shutil.copy(BACKBONE_CKPT, CKPT_LOCAL)
for z in val_zones:                                   # reuse the clean val associations saved by colab_fusion
    src, dst = f'{IMAGEVOTE_DIR}/{z}/assoc.npz', f'{ASSOC_DIR}/{z}/assoc.npz'
    if os.path.exists(src) and not os.path.exists(dst):
        os.makedirs(f'{ASSOC_DIR}/{z}', exist_ok=True); shutil.copy(src, dst); print(f"  assoc {z}: copied from imagevote")
print("associations present:", [z for z in zones if os.path.exists(f'{ASSOC_DIR}/{z}/assoc.npz')],
      "(missing ones are computed on first use, ~1 min each)")
"""),
code("""#@title 6. Train the queue of heads (each resumes from its own last.pth; evaluates on val after every pass)
import subprocess, time, os
env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
proc, status, t_all = None, {}, time.time()
try:
    for i, r in enumerate(runs):
        print(f"\\n===== run {i + 1}/{len(runs)}: {r['name']} (head {r['head']}" + (f" K={r['k']}" if r['head'] == 'window' else '') + ") =====", flush=True)
        cmd = [PY, '/content/code/scripts/train_fusion.py', '--cache', '/content/cache', '--cache2d', CACHE2D, '--data', DATA,
               '--assoc-dir', ASSOC_DIR, '--litept', '/content/LitePT', '--backbone-ckpt', CKPT_LOCAL, '--work-dir', work_of(r['name']),
               '--head', r['head'], '--k', str(r['k']), '--train-zones', ','.join(train_zones), '--val-zones', ','.join(val_zones),
               '--passes', str(PASSES), '--its-per-zone', str(ITS_PER_ZONE), '--head-points', str(HEAD_POINTS),
               '--max-points', str(MAX_POINTS), '--eval-max-points', str(EVAL_MAX_POINTS), '--lr', str(LR),
               '--pose-noise-t', str(r['pt']), '--pose-noise-r', str(r['pr']), '--aug-prob', str(AUG_PROB),
               '--seed', str(r['seed']), '--max-hours', str(MAX_HOURS)]
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            if 'Warning' not in line and 'def backward' not in line and '_TORCH_CUSTOM_FWD' not in line: print(line, end='')
        proc.wait(); status[r['name']] = proc.returncode
        print(f"[{r['name']}] exit {proc.returncode} after {(time.time() - t0) / 3600:.2f} h", flush=True)
    print(f"queue finished in {(time.time() - t_all) / 3600:.2f} h: {status}")
except KeyboardInterrupt:
    if proc is not None: proc.terminate(); proc.wait()
    print("stopped by user; last.pth of every run is kept on Drive -> re-run this cell to resume"); raise
"""),
code("""#@title 7. Results of best.pth on the val zones + figure, for every run of the queue (re-evaluates only when needed)
FORCE_EVAL = False        #@param {type:"boolean"}
import subprocess, json, os, sys
from IPython.display import Image, display
env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
table = []
for r in runs:
    work = work_of(r['name']); ev = f'{work}/eval/eval.json'
    if not (os.path.exists(f'{work}/best.pth') or os.path.exists(f'{work}/last.pth')):
        print(f"[{r['name']}] no checkpoint yet"); continue
    have = os.path.exists(ev) and json.load(open(ev)).get('tag') == 'best'     # cell 6 leaves the best checkpoint's evaluation there
    if FORCE_EVAL or not have:
        cmd = [PY, '/content/code/scripts/train_fusion.py', '--cache', '/content/cache', '--cache2d', CACHE2D, '--data', DATA,
               '--assoc-dir', ASSOC_DIR, '--litept', '/content/LitePT', '--backbone-ckpt', CKPT_LOCAL, '--work-dir', work,
               '--head', r['head'], '--k', str(r['k']), '--val-zones', ','.join(val_zones), '--eval-max-points', str(EVAL_MAX_POINTS), '--eval-only']
        x = subprocess.run(cmd, capture_output=True, text=True, env=env)
        print('\\n'.join(l for l in x.stdout.splitlines() if 'Warning' not in l)[-3000:], x.stderr[-800:])
    if not os.path.exists(ev): continue
    e = json.load(open(ev))
    print(f"\\n[{r['name']}]", {k: e.get(k) for k in ('model', 'tag', 'it')}, f"fused mIoU {e['mIoU']:.2f}  OA {e['OA']:.2f}   LiDAR-only {e['lidar_only']['mIoU']:.2f}")
    print('  fused     ', {k: (None if v is None else round(v, 1)) for k, v in e['per_class'].items()})
    print('  LiDAR-only', {k: (None if v is None else round(v, 1)) for k, v in e['lidar_only']['per_class'].items()})
    print('  per zone (fused, LiDAR-only)', {z: (round(v['mIoU'], 1), round(v['lidar_only_mIoU'], 1)) for z, v in e['per_zone'].items()})
    table.append((r['name'], e['mIoU'], e['lidar_only']['mIoU']))
    fig = f"{work}/eval/{r['name']}_val.png"
    title = f"{r['name']} ({r['head']}" + (f" K={r['k']}" if r['head'] == 'window' else '') + ") — validation zones"
    subprocess.run([sys.executable, '/content/code/scripts/plot_eval.py', '--eval', ev, '--out', fig, '--title', title], env=dict(os.environ, PYTHONPATH='/content/code'))
    display(Image(fig)) if os.path.exists(fig) else None
print('\\nsummary:'); [print(f"  {n:16s} fused {m:.2f}   LiDAR-only {l:.2f}   gain {m - l:+.2f}") for n, m, l in table]
"""),
code("""#@title 8. (optional) Shut the runtime down once everything above has finished - saves compute units on unattended runs
if SHUTDOWN_WHEN_DONE:
    from google.colab import drive, runtime
    print("flushing pending writes to Drive ..."); drive.flush_and_unmount()
    print("results are on Drive; shutting the runtime down"); runtime.unassign()
else:
    print("SHUTDOWN_WHEN_DONE is off -> remember to Disconnect and delete runtime")
"""),
md("""## หมายเหตุ
- ผลไป Drive `GridNet/fusion/<RUN_NAME>/`: `last.pth`, `best.pth`, `log.csv`, `eval/eval.json`, `eval/<zone>/fused_pred.npy`
- association ต่อโซนมีสำเนาเก็บที่ `GridNet/fusion/assoc_<RUN2D>/<zone>/assoc.npz` แต่ session ที่มี GPU จะคำนวณใหม่ทุกครั้ง (~0.4 นาที/โซน เร็วกว่าอ่านไฟล์ 4–10 GB จาก Drive)
- cell 6 จบด้วยการทำให้ `eval/` เป็นผลของ `best.pth` เสมอ (ไม่ eval ซ้ำถ้ามีอยู่แล้ว) → cell 7 แค่แสดงผล
- `lidar_only` ใน eval.json = frozen backbone บน window ชุดเดียวกัน ต้องได้ ≈ 69.9 ถ้าไม่ใช่ แปลว่าโหลด checkpoint ผิด
- จบแล้ว **Disconnect and delete runtime**"""),
]

compact_cells = [
md("""# GridNet-HD — pack the 2D cache into one compact tar per zone (one-time, **CPU runtime**)

ทำไม: Colab อ่าน Drive ผ่าน FUSE ได้แค่ ~10–15 MB/s เมื่ออ่านไฟล์เล็กทีละไฟล์ แต่ >100 MB/s กับไฟล์ใหญ่ไฟล์เดียว
notebook นี้อ่าน `GridNet/cache2d/<RUN2D>/<zone>/*.npz` (fp16 ~50 MB/ภาพ หรือ compact) → เขียน **`<zone>.tar`** (compact uint8/int8
~24 MB/ภาพ) + `<zone>.tar.json` ไว้ข้าง ๆ โฟลเดอร์เดิม (ไฟล์เดิมไม่ถูกแตะ) → `colab_train_fusion` cell 5 จะดึง tar แทนการ copy ทีละไฟล์

Runtime: **CPU** (ธรรมดาก็ได้; High-RAM ถ้าเลือกได้) — ไม่ใช้ GPU, ใช้แค่ numpy · 14 zones ≈ 2,760 ภาพ ≈ 140 GB fp16 → ≈ 67 GB tar ·
Drive mount อ่านได้ ~20–25 MB/s รวม (วัดจากโซนแรก ๆ: 111 ภาพ 4.7 นาที) → ทั้งหมดราว **2–2.5 ชม.**
การอ่านผ่าน Drive mount, แต่ **การอัปโหลด tar ใช้ Drive API โดยตรง** (resumable, ทีละ 32 MB) เพราะการเขียนไฟล์หลาย GB ผ่าน mount ทำให้ Drive client
ของ VM โดนฆ่าตอน RAM ตึง (`exit -9` / `Transport endpoint is not connected`) · cell 3 จะขอสิทธิ์ Google account 1 ครั้ง (popup)
**Resume**: รันซ้ำได้ — โซนที่มี tar ครบบน Drive แล้วจะถูกข้าม, ไฟล์ที่แปลงแล้วบนดิสก์ local ของ session เดิมถูกใช้ต่อ ·
ถ้า Drive mount ตาย cell 3 จะ remount แล้วทำโซนนั้นต่อเอง"""),
code("""#@title 1. Mount Drive + parameters
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/GridNet'
CODE_ZIP = f'{DRIVE}/code/gridnet_code.zip'      #@param {type:"string"}
RUN2D = 'upernet_convnext_t_v2'                  #@param {type:"string"}
ZONES = 't1z6a,t2z5,t3z3,t5a1,t5b2,t5c1,t6z2,t6z4,t1z5b,t1z8,t3z4,t4z1,t5b1,t5b5'   #@param {type:"string"}
THREADS = 4               #@param {type:"integer"}
WORK = '/content/cache2d_work'
CACHE2D_DRIVE = f'{DRIVE}/cache2d/{RUN2D}'
assert os.path.isdir(CACHE2D_DRIVE), CACHE2D_DRIVE
zones = [z.strip() for z in ZONES.split(',') if z.strip()]
have = {z: len([f for f in os.listdir(f'{CACHE2D_DRIVE}/{z}') if f.endswith('.npz')]) if os.path.isdir(f'{CACHE2D_DRIVE}/{z}') else 0 for z in zones}
print({z: n for z, n in have.items()}); assert all(have.values()), f"zones without a 2D cache: {[z for z, n in have.items() if not n]}"
"""),
code("""#@title 2. Code (system Python is enough: numpy only)
import zipfile, os, subprocess, sys
os.makedirs('/content/code', exist_ok=True); zipfile.ZipFile(CODE_ZIP).extractall('/content/code')
r = subprocess.run([sys.executable, '-c', "import sys; sys.path.insert(0,'/content/code'); from gridnet.train.maps2d import write_zone_tar; print('imports ok')"],
                   capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-1500:])
print(subprocess.run(['df', '-h', '/content'], capture_output=True, text=True).stdout)
ram_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9
print(f"RAM {ram_gb:.0f} GB")
if ram_gb < 20: print("standard runtime (12.7 GB): fine - tars are uploaded through the Drive API, not the mount")
"""),
code("""#@title 3. Convert (reads through the mount) + upload each zone tar through the Drive API (resumable; finished zones skipped)
import subprocess, sys, time, os, json
from google.colab import drive, auth
sys.path.insert(0, '/content/code')
from gridnet.train import drive_api
auth.authenticate_user()                                   # once per session (popup) - Drive API scope
svc = drive_api.drive_service()
folder_id = drive_api.resolve_path(svc, ['GridNet', 'cache2d', RUN2D])
print("Drive folder", f"GridNet/cache2d/{RUN2D}", "=", folder_id)
LOCAL_TARS = f'{WORK}/tars'
env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore', MALLOC_ARENA_MAX='2')

def convert(zone):
    cmd = [sys.executable, '/content/code/scripts/compact_cache2d.py', '--cache2d', CACHE2D_DRIVE, '--zones', zone,
           '--work', WORK, '--threads', str(THREADS), '--no-publish', '--pause', '0']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        for line in p.stdout: print(line, end='')
        p.wait()
    except KeyboardInterrupt:
        p.terminate(); p.wait(); raise
    return p.returncode

def remount():
    try: drive.flush_and_unmount()
    except Exception as e: print("  unmount:", str(e)[:200])
    time.sleep(15); drive.mount('/content/drive', force_remount=True)

def upload(path, mimetype):
    last = [0]
    def prog(done, total):
        if done - last[0] >= 512 << 20 or done == total:
            last[0] = done; print(f"\\r  upload {os.path.basename(path)}: {done / 1e9:.1f}/{total / 1e9:.1f} GB", end='', flush=True)
    r = drive_api.upload_file(svc, path, folder_id, mimetype=mimetype, progress=prog)
    print(f"  ok ({int(r['size']) / 1e9:.2f} GB in {r['seconds'] / 60:.1f} min, {int(r['size']) / 1e6 / max(r['seconds'], 1e-3):.0f} MB/s)")
    return r

t0 = time.time(); done_zones = []
try:
    for z in zones:
        m = drive_api.zone_tar_on_drive(svc, folder_id, z)
        if m is not None:
            print(f"[{z}] already on Drive ({m['n']} images, {m['bytes'] / 1e9:.1f} GB) - skip"); done_zones.append(z); continue
        tar, man = f'{LOCAL_TARS}/{z}.tar', f'{LOCAL_TARS}/{z}.tar.json'
        for attempt in range(1, 8):                        # the mount may die while reading: remount and resume the zone
            if os.path.exists(tar) and os.path.exists(man) and os.path.getsize(tar) == json.load(open(man))['bytes']:
                rc = 0; break
            rc = convert(z)
            if rc == 0: break
            if rc == -9: print("converter killed (out of memory) - lower THREADS or use a High-RAM runtime"); break
            print(f"[{z}] converter exit {rc} (attempt {attempt}) -> remounting Drive", flush=True); remount()
        if rc != 0 or not os.path.exists(tar): print(f"[{z}] FAILED - re-run this cell"); break
        upload(tar, 'application/x-tar'); upload(man, 'application/json')
        os.remove(tar); os.remove(man); done_zones.append(z)
        print(f"[{z}] done  ({(time.time() - t0) / 60:.1f} min elapsed, {len(done_zones)}/{len(zones)} zones)", flush=True)
except KeyboardInterrupt:
    print("stopped; re-run this cell to continue")
print(f"{len(done_zones)}/{len(zones)} zones on Drive after {(time.time() - t0) / 60:.1f} min")
"""),
code("""#@title 4. Check what is on Drive now (through the API, works even if the mount is dead)
import sys
sys.path.insert(0, '/content/code')
from gridnet.train import drive_api
from google.colab import auth
auth.authenticate_user()
svc = drive_api.drive_service()
folder_id = drive_api.resolve_path(svc, ['GridNet', 'cache2d', RUN2D])
total = 0
for z in zones:
    m = drive_api.zone_tar_on_drive(svc, folder_id, z)
    if m is None: print(f"  {z}: NO complete tar"); continue
    total += m['bytes']; print(f"  {z}: {m['n']} images  {m['bytes'] / 1e9:.1f} GB")
print(f"total {total / 1e9:.1f} GB of zone tars in GridNet/cache2d/{RUN2D}")
"""),
md("""## หมายเหตุ
- โฟลเดอร์ `<zone>/` เดิม (fp16) ยังอยู่ ใช้เป็นสำรองได้; ถ้าต้องการพื้นที่ค่อยลบทีหลังเมื่อ tar ครบแล้ว (Drive มี 4.8 TB ไม่รีบ)
- export โซนใหม่ในอนาคต: รัน colab_train_upernet cell 8 ตามปกติ แล้วรัน notebook นี้เพิ่มเฉพาะโซนนั้น (`ZONES`)
- จบแล้ว **Disconnect and delete runtime**"""),
]

protocol_cells = [
md("""# GridNet-HD — Robustness protocol (paper Sec. 5): models under camera-pose error

ต้องมีก่อน: LiDAR cache + zone tar ของ **val 6 zones**, checkpoint LiDAR-only, และ `best.pth` ของ heads ที่จะวัด (`GridNet/fusion/<run>/best.pth`)

ทำอะไร: ต่อ val zone — (1) คำนวณ feature/logits ของ frozen backbone บน subset แบบ stratified ตามคลาส **ครั้งเดียว** (เก็บไว้ที่
`GridNet/protocol/subsets_<RUN2D>/<zone>/subset.npz` ใช้ซ้ำได้ทุก run) → (2) ทุก (sweep, level, draw): perturb กล้องทุกตัว → re-project/re-associate →
head forward ของทุกโมเดล (LiDAR-only, ImageVote, heads) → weighted confusion → `results.json` → (3) `summary.json` / `curves.csv` / `scores.csv` + รูป

| MODE | association ภายใต้ noise | เวลาต่อ zone-draw (A100) | ใช้เมื่อ |
|---|---|---|---|
| `fast` | re-project 8 views เดิม คง visibility ของ association เดิมไว้ (ตัดเฉพาะคู่ที่หลุดขอบภาพ) | ~1 วินาที (6 โมเดล) | ระหว่างพัฒนา head (run `fast_v1`: 6 zones × 133 evaluations ใน 18 นาที); `colab_heads_lab` เรียกโหมดนี้ให้เองท้าย lab |
| `fast_retest` | เหมือน `fast` แต่เช็ค visibility ของ pixel ใหม่กับ depth map เดิมด้วย (แบบแรกที่ออกแบบไว้) | ~1 วินาที | ใช้เทียบใน cell 8 เท่านั้น |
| `exact` | render depth map ใหม่ด้วยกล้องที่ถูก perturb + เลือก views ใหม่ | ~6 วินาที | **ตัวเลขใน paper** (ทั้ง 6 zones ประมาณ 1–1.3 ชม. ยังไม่เคยรันเต็ม) |

วัดแล้ว (run `fast_v1`, 13 พ.ย.): `fast` ต่างจาก `exact` ไม่เกิน 0.13 mIoU สำหรับ fusion heads แต่ดีเกินจริงได้ถึง 0.8 mIoU สำหรับ ImageVote
(structural 4.8 IoU ที่ 10 cm) → ตัวเลขสุดท้ายใช้ `exact`; `fast_retest` ต่างถึง 1.4 mIoU

sweep `trans` / `rot`: `DRAWS` draws ต่อ level · sweep `drift`: 2 evaluations ต่อ level (ทิศ + และ −; drift ไม่มีความสุ่มอย่างอื่น)

**Resume**: Run all ซ้ำได้ — zone ที่อยู่ใน results.json แล้วถูกข้าม แต่ต้องเป็นค่าเดิม (heads, sweeps, draws, mode) ถ้าเปลี่ยนให้ตั้ง `TAG` ใหม่ (script จะหยุดเองถ้าไม่ตรง)"""),
code("""#@title 1. GPU (A100 asked for; an L4 also works for this notebook - only the validation zones are loaded - but is slower)
import subprocess, torch
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
assert torch.cuda.get_device_capability()[0] >= 8, "need L4 / A100 (flash-attn)"
"""),
code("""#@title 2. Mount Drive + parameters
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/GridNet'
CODE_ZIP = f'{DRIVE}/code/gridnet_code.zip'      #@param {type:"string"}
TAG = 'exact_v1'                                 #@param {type:"string"}
MODE = 'exact'                                   #@param ["fast", "fast_retest", "exact"]
DRAWS = 10                                       #@param {type:"integer"}
SWEEPS = 'trans,rot,drift'                       #@param {type:"string"}
HEADS = 'late_v1,concat_v1,window3_v1,late_nat_b1'   #@param {type:"string"}
RUN2D = 'upernet_convnext_t_v2'                  #@param {type:"string"}
BACKBONE_CKPT = f'{DRIVE}/ckpt/litept_s_lidar_v2/best.pth'   #@param {type:"string"}
VAL_ZONES = 't1z5b,t1z8,t3z4,t4z1,t5b1,t5b5'     #@param {type:"string"}
CAP = 500000              #@param {type:"integer"}
CAP_FULL = 2000000        #@param {type:"integer"}
EVAL_MAX_POINTS = 900000  #@param {type:"integer"}
SEED = 0                  #@param {type:"integer"}
EXACT_CHECK = True        #@param {type:"boolean"}
EXACT_ZONES = 't5b5,t1z8' #@param {type:"string"}
EXACT_SWEEPS = 'trans,rot,drift' #@param {type:"string"}
EXACT_DRAWS = 2           #@param {type:"integer"}
SHUTDOWN_WHEN_DONE = False #@param {type:"boolean"}
CACHE_DRIVE = f'{DRIVE}/cache'
CACHE2D_DRIVE = f'{DRIVE}/cache2d/{RUN2D}'
ASSOC_DIR = f'{DRIVE}/fusion/assoc_{RUN2D}'
IMAGEVOTE_DIR = f'{DRIVE}/fusion/imagevote_{RUN2D}'
SUBSET_DIR = f'{DRIVE}/protocol/subsets_{RUN2D}'
OUT = f'{DRIVE}/protocol/{TAG}'
OUT_EXACT = f'{DRIVE}/protocol/{TAG}_exact'
OUT_RETEST = f'{DRIVE}/protocol/{TAG}_retest'
for d in (ASSOC_DIR, SUBSET_DIR, OUT): os.makedirs(d, exist_ok=True)
val_zones = [z.strip() for z in VAL_ZONES.split(',') if z.strip()]
exact_zones = [z.strip() for z in EXACT_ZONES.split(',') if z.strip()]
assert MODE == 'exact' or not EXACT_CHECK or set(exact_zones) <= set(val_zones), "EXACT_ZONES must be among VAL_ZONES"
heads = {}
for name in [h.strip() for h in HEADS.split(',') if h.strip()]:
    path = f'{DRIVE}/fusion/{name}/best.pth'
    assert os.path.exists(path), f"no checkpoint for head {name}: {path}"
    heads[name] = path
print("heads:", list(heads), "\\nout ->", OUT)
"""),
code("""#@title 3. Environment (restore tarball from Drive)
import subprocess, os, time
ENV = '/content/env'; TAR = f'{DRIVE}/env/env_py310_torch241_cu124.tar.gz'
t0 = time.time()
if not os.path.exists(f'{ENV}/bin/python'):
    assert os.path.exists(TAR), "no env tarball on Drive: run colab_train_litept cell 3 once"
    subprocess.run(['tar', '-C', '/content', '-xzf', TAR], check=True); print("env restored from Drive")
PY = f'{ENV}/bin/python'
subprocess.run([PY, '-m', 'pip', 'install', '-q', 'huggingface_hub', 'pillow', 'matplotlib'], check=True)
print(f"env ready in {(time.time() - t0) / 60:.1f} min")
"""),
code("""#@title 4. Code + LitePT
import subprocess, zipfile, os
os.makedirs('/content/code', exist_ok=True); zipfile.ZipFile(CODE_ZIP).extractall('/content/code')
assert 'PROTOCOL_VERSION' in open('/content/code/gridnet/train/protocol.py').read(), "gridnet_code.zip on Drive is older than update32: pack and upload it again"
if not os.path.exists('/content/LitePT'):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/prs-eth/LitePT.git", "/content/LitePT"], check=True)
r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0,'/content/LitePT'); sys.path.insert(0,'/content/code'); "
                    "from litept.model import LitePT; import gridnet.train.protocol; print('imports ok')"], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-1500:])
"""),
code("""#@title 5. Data -> local disk (val zones only): LiDAR caches, zone tars, camera files, backbone checkpoint, associations
import os, sys, shutil, subprocess, time
sys.path.insert(0, '/content/code')
from gridnet.train.maps2d import cached_image_ids, zone_tar_manifest, extract_zone_tar, write_compact
from concurrent.futures import ThreadPoolExecutor
DATA = '/content/data'; os.makedirs(DATA, exist_ok=True)
zones = val_zones
t0 = time.time()
for z in zones:
    dst = f'/content/cache/{z}/points.npz'
    if not os.path.exists(dst):
        os.makedirs(f'/content/cache/{z}', exist_ok=True); shutil.copy(f'{CACHE_DRIVE}/{z}/points.npz', dst)
print(f"LiDAR caches ready ({(time.time() - t0) / 60:.1f} min)")
CACHE2D = '/content/cache2d'; t0 = time.time()
for z in zones:
    if zone_tar_manifest(CACHE2D_DRIVE, z, wait=30) is not None:
        n = extract_zone_tar(CACHE2D_DRIVE, z, CACHE2D); how = 'tar'
    else:
        ids = cached_image_ids(CACHE2D_DRIVE, z)
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(lambda i: write_compact(CACHE2D_DRIVE, z, i, CACHE2D), ids))
        n = len(cached_image_ids(CACHE2D, z)); how = 'files (slow: no tar yet)'
    print(f"  {z}: {n} images from {how}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
code = ("import sys; sys.path.insert(0,'/content/code'); from gridnet.train.data2d import fetch_zones; "
        f"fetch_zones({zones!r}, '{DATA}', what=('pose',))")
subprocess.run([PY, '-c', code], check=True); print("camera files ready")
CKPT_LOCAL = '/content/backbone_best.pth'
if not os.path.exists(CKPT_LOCAL): shutil.copy(BACKBONE_CKPT, CKPT_LOCAL)
for z in zones:                                        # clean val associations saved by colab_fusion / colab_train_fusion
    src, dst = f'{IMAGEVOTE_DIR}/{z}/assoc.npz', f'{ASSOC_DIR}/{z}/assoc.npz'
    if os.path.exists(src) and not os.path.exists(dst):
        os.makedirs(f'{ASSOC_DIR}/{z}', exist_ok=True); shutil.copy(src, dst); print(f"  assoc {z}: copied from imagevote")
print("associations present:", [z for z in zones if os.path.exists(f'{ASSOC_DIR}/{z}/assoc.npz')], "(missing ones are computed on first use)")
print("subsets present:", [z for z in zones if os.path.exists(f'{SUBSET_DIR}/{z}/subset.npz')], "(missing ones are computed on first use, ~2 min each)")
"""),
code("""#@title 6. Run the protocol (resumes per zone; results -> OUT/results.json)
import subprocess, time, os
cmd = [PY, '/content/code/scripts/run_protocol.py', '--cache', '/content/cache', '--cache2d', CACHE2D, '--data', DATA,
       '--assoc-dir', ASSOC_DIR, '--out', OUT, '--subset-dir', SUBSET_DIR, '--litept', '/content/LitePT',
       '--backbone-ckpt', CKPT_LOCAL, '--zones', ','.join(val_zones), '--sweeps', SWEEPS, '--draws', str(DRAWS),
       '--mode', MODE, '--cap', str(CAP), '--cap-full', str(CAP_FULL), '--eval-max-points', str(EVAL_MAX_POINTS), '--seed', str(SEED),
       '--heads'] + [f'{n}={p}' for n, p in heads.items()]
env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
t0 = time.time()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
try:
    for line in p.stdout:
        if 'Warning' not in line and 'def backward' not in line and '_TORCH_CUSTOM_FWD' not in line: print(line, end='')
    p.wait(); print(f"exit {p.returncode} after {(time.time() - t0) / 60:.1f} min")
    if p.returncode != 0: print("-> if it STOPPED on 'other settings': this TAG already holds a run with other heads / sweeps / draws / mode - give TAG a new name in cell 2")
except KeyboardInterrupt:
    p.terminate(); p.wait(); print("stopped by user; finished zones are in results.json -> re-run this cell to continue")
"""),
code("""#@title 7. Summary table + figure
import subprocess, json, os, sys
from IPython.display import Image, display
fig = f'{OUT}/protocol_{TAG}.png'
r = subprocess.run([PY, '/content/code/scripts/plot_protocol.py', '--summary', f'{OUT}/summary.json', '--out', fig],
                   capture_output=True, text=True, env=dict(os.environ, PYTHONPATH='/content/code', MPLBACKEND='Agg'))
print(r.stdout[-12000:], r.stderr[-800:])
display(Image(fig)) if os.path.exists(fig) else None
"""),
code("""#@title 8. (optional) Check of the fast approximation on EXACT_ZONES with the same cameras: exact mode (~6 s per draw) and the fast_retest variant
import subprocess, time, os
if EXACT_CHECK and MODE == 'fast':
    env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for mode, out in (('fast_retest', OUT_RETEST), ('exact', OUT_EXACT)):
        cmd = [PY, '/content/code/scripts/run_protocol.py', '--cache', '/content/cache', '--cache2d', CACHE2D, '--data', DATA,
               '--assoc-dir', ASSOC_DIR, '--out', out, '--subset-dir', SUBSET_DIR, '--litept', '/content/LitePT',
               '--backbone-ckpt', CKPT_LOCAL, '--zones', ','.join(exact_zones), '--sweeps', EXACT_SWEEPS, '--draws', str(EXACT_DRAWS),
               '--mode', mode, '--cap', str(CAP), '--cap-full', str(CAP_FULL), '--eval-max-points', str(EVAL_MAX_POINTS), '--seed', str(SEED),
               '--heads'] + [f'{n}={p}' for n, p in heads.items()]
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in p.stdout:
            if line.startswith('[protocol]') and (' draw ' not in line or mode == 'exact'): print(line, end='')
            elif 'Error' in line or 'Traceback' in line or line.startswith('  File'): print(line, end='')
        p.wait(); print(f"{mode}: exit {p.returncode} after {(time.time() - t0) / 60:.1f} min")
    for a, b, names, f in ((OUT, OUT_EXACT, 'fast,exact', 'compare.json'), (OUT_RETEST, OUT_EXACT, 'retest,exact', 'compare_retest.json')):
        r = subprocess.run([PY, '/content/code/scripts/compare_protocol.py', '--a', a, '--b', b, '--names', names, '--out', f'{OUT_EXACT}/{f}'],
                           capture_output=True, text=True, env=env)
        print(r.stdout[-14000:], r.stderr[-800:])
else:
    print("check skipped (EXACT_CHECK off, or MODE is not 'fast')")
"""),
code("""#@title 9. (optional) Shut the runtime down once everything above has finished - saves compute units on unattended runs
if SHUTDOWN_WHEN_DONE:
    from google.colab import drive, runtime
    print("flushing pending writes to Drive ..."); drive.flush_and_unmount()
    print("results are on Drive; shutting the runtime down"); runtime.unassign()
else:
    print("SHUTDOWN_WHEN_DONE is off -> remember to Disconnect and delete runtime")
"""),
md("""## หมายเหตุ
- `scores.csv`: ต่อ (model, sweep, class) → `clean_iou`, `R` (robustness score, 1 = ไม่ตก), `drop_ref` (IoU ตกที่ reference level: 10 cm / 0.1° / 25 cm drift)
- Gate G3 (run `fast_v1`, 13 พ.ย.): ที่ `trans` 0.10 class `structural` / `conductor` ของทุก head เปลี่ยนไม่เกิน 0.2 IoU (เกณฑ์เรื่อง A คือตก ≥ 15) → **เรื่อง B**
- log ของ cell 6: หนึ่งบรรทัดต่อ (zone, sweep, level, draw) พร้อม mIoU ของโซนนั้น (เฉลี่ยเฉพาะ class ที่โซนนั้นมี) สำหรับทุกโมเดล — `lidar` ต้องคงที่ทุกบรรทัดของโซนเดียวกัน
- ตรวจผล: คอลัมน์ `clean` ของตารางสรุป (cell 7) ต้องเท่ากับ bank estimate ของ heads lab เพราะใช้ subset ชุดเดียวกัน:
  `lidar` 69.9, `imagevote` 65.5, `late_v1` 81.9, `concat_v1` 78.6, `window3_v1` 78.6, `late_nat_b1` 81.8
- cell 8 (`EXACT_CHECK`, เฉพาะเมื่อ `MODE` = fast): รันโหมด exact และ fast_retest บน `EXACT_ZONES` ด้วยกล้องชุดเดียวกับ draw 0..`EXACT_DRAWS`-1 ของโหมด fast
  แล้วพิมพ์ตารางเทียบ → `GridNet/protocol/<TAG>_exact/compare.json` และ `compare_retest.json`
- จบแล้ว **Disconnect and delete runtime**"""),
]

lab_cells = [
md("""# GridNet-HD — Heads lab: เทรน fusion head หลายแบบพร้อมกันบน feature ที่คำนวณไว้แล้ว (feature bank)

ทำไม: backbone ทั้งสองเป็น frozen → feature ของ LiDAR คำนวณ **ครั้งเดียว** ต่อโซนก็พอ เก็บเป็น "bank" (subset แบบ stratified ชุดเดียวกับที่ robustness protocol ใช้)
แล้ว 1 iteration ของ head เหลือแค่ gather จาก 2D maps + forward/backward ของ head ตัวเล็ก → เทรน head หลายแบบ **พร้อมกัน** บน batch ชุดเดียวกันได้
(วัดจริงบน A100, lab1: 12 heads ใน 3 batch groups = 0.55 s/it → 2 passes x 8 zones x 300 its ใน 54 นาที)

ขั้นตอน: (1) เตรียมข้อมูล (2) bank ของแต่ละโซน: มีบน Drive แล้วก็ copy มาใช้ ไม่มีก็สร้าง (lab1: 26 นาทีสำหรับ 14 โซน ตอนนี้อยู่บน Drive ครบแล้ว)
(3) เทรนทุก head ใน `SPECS` พร้อมกัน validate บน bank ของ val zones ทุก pass (4) diagnostics: น้ำหนักที่ดีที่สุดของทุก head ถูกวัดแบบ
**ไม่มี LiDAR** / **ไม่มีภาพ** และบน **train zones** (โซนที่ backbone ทั้งสองเคยเห็นตอนเทรน) — head เดิมใน `REFERENCE_HEADS`
(เช่น `late_v1` ที่เทรนแบบ crop) ถูกวัดบน bank ชุดเดียวกันไปด้วยโดยไม่เทรนใหม่ (5) `PROTOCOL`: robustness protocol (โหมด fast) ของทุก head ใน lab นี้
+ reference heads → `GridNet/protocol/<TAG>/` (6) `FULL_EVAL`: full-zone evaluation ของ head ที่ระบุ (~20 นาที/head)

**lab4 = รอบสุดท้ายของการทดลอง** (หลัง lab3: G5 ครบ 3 seeds, fix = pose-noise augmentation): เทรน 4 heads ที่ยังขาด
(seeds 1 และ 2 ของ window K=5 แบบไม่มีและมี augmentation — `win5_n` seed 0 เสีย thin classes น้อยกว่า concat / K=3 ราวครึ่งหนึ่ง จึงต้องมี error bar)
แล้วรัน **protocol โหมด exact กับโมเดลทุกตัวในตารางของ paper** (`PROTOCOL_HEADS`: 35 heads + LiDAR-only + ImageVote → `GridNet/protocol/exact_v1/`)
และ **ablation จำนวน views ตอน test** (cell 8b: โหมด fast กับ V = 1, 2, 4, 8 views ที่ใกล้ที่สุด → `GridNet/protocol/views<V>/`)
เวลาบน A100: วัดจริงจาก lab3 (12 heads ใน 8 groups, 13 พ.ย.) = เตรียมข้อมูล 26 นาที, เทรน 58 นาที (0.58 s/it) + diagnostics 8 นาที, protocol โหมด fast 37 นาทีกับ 18 โมเดล
(3.4 s ต่อ configuration), full-zone evaluation 18 นาที; lab4 ประมาณจากอัตราเหล่านี้ (ยังไม่ได้วัด): เทรน ~50 นาที, protocol โหมด exact ~2.5–3 ชม.
(654 configurations × ราว 11–16 s), views ~35 นาที → ทั้ง notebook ราว 4.5–5 ชม. · resume ได้ทุกขั้น (protocol ต่อจากโซนที่ค้าง)

`SPECS` (cell 2): หนึ่งบรรทัดต่อ head รูปแบบ `name:head[:key=value,...]` · head = `late` | `concat` | `window`
keys: `k`, `lidar_input` (`feat` | `prob`), `image_input` (`feat` | `prob`, concat เท่านั้น), `lidar_drop`, `natural_frac`,
`pose_t`, `pose_r`, `aug_prob`, `seed`, `lr`, `wd`
ผลของแต่ละ head: `GridNet/fusion/<name>/best.pth` (ใช้กับ `colab_protocol` ได้เลย) + `eval_bank/eval.json` + `eval_bank/diagnostics.json`; ตารางรวม: `GridNet/fusion/bank_<TAG>.json`

**Resume**: Run all ซ้ำได้ — head ที่เทรนจบแล้วถูกข้าม ถ้าเพิ่มบรรทัดใหม่ใน `SPECS` จะเทรนเฉพาะตัวใหม่ (ชื่อ head ห้ามซ้ำกับ run เดิมที่ตั้งค่าต่างกัน)"""),
code("""#@title 1. GPU (this notebook needs the 40 GB of an A100: the 2D maps of t6z2 / t6z4 alone take ~14 GB on the GPU)
import subprocess, torch
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
assert torch.cuda.get_device_capability()[0] >= 8, "need an A100 (flash-attn)"
mem = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
assert mem >= 30000, ("Colab connected a smaller GPU than the A100 this notebook asks for (the A100 was not free, or another of your sessions still holds it). "
                      "Runtime -> Manage sessions: terminate old sessions; Runtime -> Disconnect and delete runtime; then run this cell again")
"""),
code("""#@title 2. Mount Drive + parameters
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/GridNet'
CODE_ZIP = f'{DRIVE}/code/gridnet_code.zip'      #@param {type:"string"}
TAG = 'lab4'                                     #@param {type:"string"}
# one head per line: name:head[:key=value,...]   (heads with the same seed and sampling share their batches)
# _n = batches at the natural class frequencies (default); _a = pose-noise augmentation: on half of the iterations
# every camera gets i.i.d. noise with sigma_t ~ U(0, pose_t) m and sigma_r ~ U(0, pose_r) deg; _sK = seed K
# lab4: seeds 1 and 2 of the 5x5 window, without and with augmentation (seed 0 = win5_n of lab3, win5_a of lab2)
SPECS = '''
win5_n_s1:window:k=5,seed=1
win5_n_s2:window:k=5,seed=2
win5_a_s1:window:k=5,pose_t=0.5,pose_r=1.0,seed=1
win5_a_s2:window:k=5,pose_t=0.5,pose_r=1.0,seed=2
'''
REFERENCE_HEADS = 'win5_n,win5_a'                #@param {type:"string"}
PROTOCOL = True                                  #@param {type:"boolean"}
PROTOCOL_MODE = 'exact'                          #@param ["fast", "exact"]
PROTOCOL_DRAWS = 10                              #@param {type:"integer"}
PROTOCOL_TAG = 'exact_v1'                        #@param {type:"string"}
# the heads of the protocol run, in the order of the paper tables ('' = the reference heads + the heads of SPECS)
PROTOCOL_HEADS = '''
late_nat_b1 late_n_s1 late_n_s2
concat_n concat_n_s1 concat_n_s2
win3_n win3_n_s1 win3_n_s2
win5_n win5_n_s1 win5_n_s2
late_a late_a_s1 late_a_s2
concat_a concat_a_s1 concat_a_s2
win3_a win3_a_s1 win3_a_s2
win5_a win5_a_s1 win5_a_s2
late_a10 late_a25 late_a100
concat_a10 concat_a25 concat_a100
concat_at concat_ar
concat_ip_n concat_lp_n concat_lp_a
'''
PROTOCOL_PLOT = 'lidar,imagevote,late_nat_b1,late_a,concat_n,concat_a,win3_n,win5_n'  #@param {type:"string"}
# number of views at test time (cell 8b): fast protocol with the V nearest views, '' = off
PROTOCOL_VIEWS = '1,2,4,8'                       #@param {type:"string"}
VIEWS_HEADS = 'late_nat_b1,late_a,concat_n,concat_a'  #@param {type:"string"}
VIEWS_DRAWS = 5                                  #@param {type:"integer"}
FULL_EVAL = ''                                   #@param {type:"string"}
RUN2D = 'upernet_convnext_t_v2'                  #@param {type:"string"}
BACKBONE_CKPT = f'{DRIVE}/ckpt/litept_s_lidar_v2/best.pth'   #@param {type:"string"}
TRAIN_ZONES = 't1z6a,t2z5,t3z3,t5a1,t5b2,t5c1,t6z2,t6z4'      #@param {type:"string"}
VAL_ZONES = 't1z5b,t1z8,t3z4,t4z1,t5b1,t5b5'                  #@param {type:"string"}
PASSES = 2                #@param {type:"integer"}
ITS_PER_ZONE = 300        #@param {type:"integer"}
BATCH_POINTS = 120000     #@param {type:"integer"}
DIAGNOSTICS = True        #@param {type:"boolean"}
MAX_HOURS = 5.0           #@param {type:"number"}
SHUTDOWN_WHEN_DONE = False #@param {type:"boolean"}
CACHE_DRIVE = f'{DRIVE}/cache'
CACHE2D_DRIVE = f'{DRIVE}/cache2d/{RUN2D}'
ASSOC_DIR = f'{DRIVE}/fusion/assoc_{RUN2D}'
BANK_ARCHIVE = f'{DRIVE}/protocol/subsets_{RUN2D}'   # <zone>/subset.npz: one file per zone, shared with colab_protocol
OUT_ROOT = f'{DRIVE}/fusion'
for d in (ASSOC_DIR, BANK_ARCHIVE, OUT_ROOT): os.makedirs(d, exist_ok=True)
train_zones = [z.strip() for z in TRAIN_ZONES.split(',') if z.strip()]
val_zones = [z.strip() for z in VAL_ZONES.split(',') if z.strip()]
specs = SPECS.replace(';', ' ').split()
for sp in specs:
    f = sp.split(':')
    assert len(f) in (2, 3) and f[1] in ('late', 'concat', 'window'), f"bad spec {sp!r}: use name:head[:key=value,...]"
names = [sp.split(':')[0] for sp in specs]
assert len(set(names)) == len(names), "every head needs its own name"
references = [n.strip() for n in REFERENCE_HEADS.split(',') if n.strip()]
assert not set(references) & set(names), "a head cannot be trained and be a reference at the same time"
full_eval = [n.strip() for n in FULL_EVAL.split(',') if n.strip()]
protocol_heads = PROTOCOL_HEADS.replace(',', ' ').split() or references + names
assert len(set(protocol_heads)) == len(protocol_heads), "a head is listed twice in PROTOCOL_HEADS"
views = [int(v) for v in PROTOCOL_VIEWS.replace(',', ' ').split()]
views_heads = [n.strip() for n in VIEWS_HEADS.split(',') if n.strip()]
have_bank = {z: os.path.exists(f'{BANK_ARCHIVE}/{z}/subset.npz') for z in train_zones + val_zones}
pose_noise = any('pose_t=' in sp or 'pose_r=' in sp for sp in specs)
print(len(specs), "heads:", ' '.join(names))
print("reference heads (evaluated on the banks, not trained):", [n for n in references if os.path.exists(f'{OUT_ROOT}/{n}/best.pth')] or 'none',
      "| missing:", [n for n in references if not os.path.exists(f'{OUT_ROOT}/{n}/best.pth')] or 'none')
print("already trained (best.pth on Drive):", [n for n in names if os.path.exists(f'{OUT_ROOT}/{n}/best.pth')] or 'none')
print(f"banks on Drive: {sum(have_bank.values())}/{len(have_bank)}", "" if all(have_bank.values()) else "-> the missing ones are built in cell 6 (~3-8 min per zone)")
PTAG = PROTOCOL_TAG.strip() or TAG
print("protocol after training:", f"{PROTOCOL_MODE} mode, {len(protocol_heads)} heads -> {DRIVE}/protocol/{PTAG}" if PROTOCOL else "off")
_missing = [n for n in protocol_heads if n not in names and not os.path.exists(f'{OUT_ROOT}/{n}/best.pth')]
if PROTOCOL and _missing: print("   NOT on Drive and not trained here (they would be left out of the protocol):", _missing)
print("views at test time (cell 8b):", f"V = {views} with {views_heads}, fast mode, {VIEWS_DRAWS} draws" if views else "off")
"""),
code("""#@title 3. Environment (restore tarball from Drive)
import subprocess, os, time
ENV = '/content/env'; TAR = f'{DRIVE}/env/env_py310_torch241_cu124.tar.gz'
t0 = time.time()
if not os.path.exists(f'{ENV}/bin/python'):
    assert os.path.exists(TAR), "no env tarball on Drive: run colab_train_litept cell 3 once"
    subprocess.run(['tar', '-C', '/content', '-xzf', TAR], check=True); print("env restored from Drive")
PY = f'{ENV}/bin/python'
subprocess.run([PY, '-m', 'pip', 'install', '-q', 'huggingface_hub', 'pillow', 'matplotlib'], check=True)
print(f"env ready in {(time.time() - t0) / 60:.1f} min")
"""),
code("""#@title 4. Code + LitePT
import subprocess, zipfile, os
os.makedirs('/content/code', exist_ok=True); zipfile.ZipFile(CODE_ZIP).extractall('/content/code')
assert 'def nearest_views' in open('/content/code/gridnet/train/protocol.py').read(), "gridnet_code.zip on Drive is older than update34: pack and upload it again"
if not os.path.exists('/content/LitePT'):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/prs-eth/LitePT.git", "/content/LitePT"], check=True)
r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0,'/content/LitePT'); sys.path.insert(0,'/content/code'); "
                    "from litept.model import LitePT; import gridnet.train.bank; print('imports ok')"], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-1500:])
"""),
code("""#@title 5. Data -> local disk: 2D caches (zone tars), camera files; LiDAR caches where needed (bank to build, pose noise, protocol, FULL_EVAL)
import os, sys, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/content/code')
from gridnet.train.maps2d import cached_image_ids, zone_tar_manifest, extract_zone_tar, write_compact
DATA = '/content/data'; os.makedirs(DATA, exist_ok=True)
zones = train_zones + val_zones
need_points = [z for z in zones if not have_bank[z] or (pose_noise and z in train_zones) or ((full_eval or PROTOCOL) and z in val_zones)]
t0 = time.time()
for z in need_points:
    dst = f'/content/cache/{z}/points.npz'
    if not os.path.exists(dst):
        os.makedirs(f'/content/cache/{z}', exist_ok=True); shutil.copy(f'{CACHE_DRIVE}/{z}/points.npz', dst)
print(f"LiDAR caches: {len(need_points)} zones ({(time.time() - t0) / 60:.1f} min)" + ("" if need_points else " - none needed"))
CACHE2D = '/content/cache2d'; t0 = time.time(); slow = []
for z in zones:
    if zone_tar_manifest(CACHE2D_DRIVE, z, wait=30) is not None:   # one big sequential read: fast
        n = extract_zone_tar(CACHE2D_DRIVE, z, CACHE2D); how = 'tar'
    else:                                                             # no tar yet: parallel per-file copy (slow)
        ids = cached_image_ids(CACHE2D_DRIVE, z)
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(lambda i: write_compact(CACHE2D_DRIVE, z, i, CACHE2D), ids))
        n = len(cached_image_ids(CACHE2D, z)); how = 'files'; slow.append(z)
    print(f"  {z}: {n} images from {how}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
if slow: print("zones without a tar (slow path) - run colab_compact_cache2d once for:", ','.join(slow))
code = ("import sys; sys.path.insert(0,'/content/code'); from gridnet.train.data2d import fetch_zones; "
        f"fetch_zones({zones!r}, '{DATA}', what=('pose',))")
subprocess.run([PY, '-c', code], check=True); print("camera files ready")
CKPT_LOCAL = '/content/backbone_best.pth'
if (full_eval or not all(have_bank.values())) and not os.path.exists(CKPT_LOCAL): shutil.copy(BACKBONE_CKPT, CKPT_LOCAL)   # the protocol reads the banks: no backbone
print(subprocess.run(['df', '-h', '--output=avail', '/content'], capture_output=True, text=True).stdout.split()[-1], "free on the local disk")
"""),
code("""#@title 6. Banks (copy from Drive, or build + archive) -> train every head of SPECS -> diagnostics
import subprocess, time, os
cmd = [PY, '/content/code/scripts/train_heads_bank.py', '--cache', '/content/cache', '--cache2d', CACHE2D, '--data', DATA,
       '--assoc-dir', ASSOC_DIR, '--subset-dir', '/content/banks', '--subset-archive', BANK_ARCHIVE, '--litept', '/content/LitePT',
       '--out-root', OUT_ROOT, '--specs', ' '.join(specs), '--tag', TAG, '--train-zones', ','.join(train_zones),
       '--val-zones', ','.join(val_zones), '--passes', str(PASSES), '--its-per-zone', str(ITS_PER_ZONE),
       '--batch-points', str(BATCH_POINTS), '--max-hours', str(MAX_HOURS), '--references', ','.join(references)]
if os.path.exists(CKPT_LOCAL): cmd += ['--backbone-ckpt', CKPT_LOCAL]
if DIAGNOSTICS: cmd += ['--diagnostics']
env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
t0 = time.time()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
try:
    for line in p.stdout:
        if 'Warning' not in line and 'def backward' not in line and '_TORCH_CUSTOM_FWD' not in line: print(line, end='')
    p.wait(); print(f"exit {p.returncode} after {(time.time() - t0) / 60:.1f} min")
except KeyboardInterrupt:
    p.terminate(); p.wait(); print("stopped by user; finished passes and banks are kept -> re-run this cell to continue"); raise
"""),
code("""#@title 7. Tables of this group (GridNet/fusion/bank_<TAG>.json)
import json, os
path = f'{OUT_ROOT}/bank_{TAG}.json'
def show(title, metrics, order):
    rows = [n for n in order if n in metrics]
    if not rows: return
    classes = list(metrics[rows[0]]['per_class'])
    print(f"=== {title} ===")
    print(f"{'model':20s} {'mIoU':>6s} {'OA':>6s}  " + ' '.join(f"{c[:9]:>9s}" for c in classes))
    for n in rows:
        m = metrics[n]
        print(f"{n:20s} {m['mIoU']:6.2f} {m['OA']:6.2f}  " + ' '.join(f"{'--' if m['per_class'][c] is None else format(m['per_class'][c], '.1f'):>9s}" for c in classes))
    print()
if os.path.exists(path):
    s = json.load(open(path)); order = ['lidar', 'imagevote'] + s.get('references', []) + [sp['name'] for sp in s['specs']]
    titles = {'val': 'validation zones, best weights (bank estimate)', 'val_no_lidar': 'validation, heads WITHOUT LiDAR evidence',
              'val_no_image': 'validation, heads WITHOUT image evidence', 'train': 'training zones (both backbones were trained on them)'}
    for case, t in titles.items():
        if case in s['diagnostics']: show(t, s['diagnostics'][case], order)
    val = s['diagnostics']['val']; rows = [n for n in order if n in val]
    zs = list(val[rows[0]]['per_zone'])
    print("=== validation mIoU per zone (classes present in the zone) ===")
    print(f"{'model':20s} " + ' '.join(f"{z:>7s}" for z in zs))
    for n in rows: print(f"{n:20s} " + ' '.join(f"{val[n]['per_zone'][z]:7.2f}" for z in zs))
else:
    print("no summary yet:", path, "(cell 6 writes it once every head has finished)")
"""),
code("""#@title 8. Robustness protocol of the heads in PROTOCOL_HEADS -> GridNet/protocol/<PROTOCOL_TAG>/ (654 configurations: fast ~0.4 s + 0.17 s per model each; exact ~5-10 s more each; one progress line per noise level)
import subprocess, time, os
from IPython.display import Image, display
if PROTOCOL:
    POUT = f'{DRIVE}/protocol/{PTAG}'
    pheads = {n: f'{OUT_ROOT}/{n}/best.pth' for n in protocol_heads if os.path.exists(f'{OUT_ROOT}/{n}/best.pth')}
    left_out = [n for n in protocol_heads if n not in pheads]
    if left_out: print("WARNING - no best.pth, LEFT OUT of the protocol:", left_out)
    print(f"{len(pheads)} heads + LiDAR-only + ImageVote, {PROTOCOL_MODE} mode")
    local = all(os.path.exists(f'/content/banks/{z}/subset.npz') for z in val_zones)
    cmd = [PY, '/content/code/scripts/run_protocol.py', '--cache', '/content/cache', '--cache2d', CACHE2D, '--data', DATA,
           '--assoc-dir', ASSOC_DIR, '--out', POUT, '--subset-dir', '/content/banks' if local else BANK_ARCHIVE, '--litept', '/content/LitePT',
           '--zones', ','.join(val_zones), '--sweeps', 'trans,rot,drift', '--draws', str(PROTOCOL_DRAWS), '--mode', PROTOCOL_MODE,
           '--heads'] + [f'{n}={p}' for n, p in pheads.items()]
    env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', MPLBACKEND='Agg')
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    last = None
    try:
        for line in p.stdout:                               # one line per noise level and per zone (per-draw numbers: results.json)
            if line.startswith('[protocol]') and ' draw ' in line:
                key = ' '.join(line.split()[1:4])           # zone sweep level
                if key != last: print(f"   {key}  ({(time.time() - t0) / 60:.1f} min)", flush=True); last = key
            elif line.startswith('[protocol]'): print(line, end='')
            elif 'Error' in line or 'Traceback' in line or line.startswith('  File'): print(line, end='')
    except KeyboardInterrupt:
        p.terminate(); p.wait(); print("stopped by user; finished zones are in results.json -> run this cell again (Runtime -> Run after) to continue"); raise
    p.wait(); print(f"protocol: exit {p.returncode} after {(time.time() - t0) / 60:.1f} min")
    if p.returncode != 0: print("-> if it STOPPED on 'other settings' (heads were added since the first run): give PROTOCOL_TAG a new name in cell 2, re-run cell 2 and this cell")
    fig = f'{POUT}/protocol_{PTAG}.png'
    r = subprocess.run([PY, '/content/code/scripts/plot_protocol.py', '--summary', f'{POUT}/summary.json', '--out', fig.replace('.png', '_all.png'),
                        '--tables', 'compact'], capture_output=True, text=True, env=env)        # table: every model
    print(r.stdout[-9000:], r.stderr[-800:])
    r = subprocess.run([PY, '/content/code/scripts/plot_protocol.py', '--summary', f'{POUT}/summary.json', '--out', fig, '--tables', 'compact',
                        '--models', PROTOCOL_PLOT], capture_output=True, text=True, env=env)     # the figure: a readable selection
    display(Image(fig)) if os.path.exists(fig) else print(r.stderr[-800:])
else:
    print("PROTOCOL is off")
"""),
code("""#@title 8b. (optional) Number of views at test time: fast protocol with the V nearest views (PROTOCOL_VIEWS) -> GridNet/protocol/views<V>/ (ImageVote + VIEWS_HEADS; ~1.3 s per configuration, ~8 min per V with 5 draws)
import subprocess, time, os, json
if views:
    vheads = {n: f'{OUT_ROOT}/{n}/best.pth' for n in views_heads if os.path.exists(f'{OUT_ROOT}/{n}/best.pth')}
    local = all(os.path.exists(f'/content/banks/{z}/subset.npz') for z in val_zones)
    env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', MPLBACKEND='Agg')
    for V in views:
        cmd = [PY, '/content/code/scripts/run_protocol.py', '--cache', '/content/cache', '--cache2d', CACHE2D, '--data', DATA,
               '--assoc-dir', ASSOC_DIR, '--out', f'{DRIVE}/protocol/views{V}', '--subset-dir', '/content/banks' if local else BANK_ARCHIVE,
               '--litept', '/content/LitePT', '--zones', ','.join(val_zones), '--sweeps', 'trans,rot,drift', '--draws', str(VIEWS_DRAWS),
               '--mode', 'fast', '--views', str(V), '--no-lidar', '--heads'] + [f'{n}={p}' for n, p in vheads.items()]
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        try:
            for line in p.stdout:                               # one line per zone
                if line.startswith('[protocol]') and ' draw ' not in line and ' head ' not in line: print(f"   V={V}  {line}", end='', flush=True)
                elif 'Error' in line or 'Traceback' in line or line.startswith('  File'): print(line, end='')
        except KeyboardInterrupt:
            p.terminate(); p.wait(); print("stopped by user; finished zones are in results.json -> run this cell again to continue"); raise
        p.wait(); print(f"views {V}: exit {p.returncode} after {(time.time() - t0) / 60:.1f} min")
        if p.returncode != 0: print(f"-> if it STOPPED on 'other settings': GridNet/protocol/views{V} was started with other heads or draws; rename that folder on Drive and run this cell again")
    done = {}
    for V in views:
        f = f'{DRIVE}/protocol/views{V}/summary.json'
        if os.path.exists(f):
            sm = json.load(open(f))
            done[V] = (sm['meta']['models'], {(c['model'], c['sweep'], float(c['level']), c['class']): 100 * c['iou_mean'] for c in sm['curves'] if c['iou_mean'] is not None})
    cols = [('trans', 0.0, 'clean'), ('trans', 0.1, 'trans 0.1'), ('trans', 0.5, 'trans 0.5'), ('rot', 0.1, 'rot 0.1'), ('rot', 1.0, 'rot 1'), ('drift', 0.5, 'drift 0.5')]
    if done:
        print()
        print(f"mean IoU [%]: all classes / electrical classes, with the V nearest views (fast mode, {VIEWS_DRAWS} draws)")
        print(f"{'model':14s}{'V':>3s}" + ''.join(f"{c[2]:>15s}" for c in cols))
        for m in done[min(done)][0]:
            for V in sorted(done):
                val = done[V][1]
                print(f"{m:14s}{V:3d}" + ''.join(f"{val[(m, sw, lv, 'mean_all')]:.1f} / {val[(m, sw, lv, 'mean_electrical')]:.1f}".rjust(15) for sw, lv, _ in cols))
else:
    print("PROTOCOL_VIEWS is empty")
"""),
code("""#@title 9. (optional) Full-zone evaluation of the heads in FULL_EVAL: backbone on every window + window votes (~20 min per head, a few lines per zone while it runs) -> <head>/eval/eval.json
import subprocess, json, os, time
env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
for name in full_eval:
    work = f'{OUT_ROOT}/{name}'; best = f'{work}/best.pth'; ev = f'{work}/eval/eval.json'
    if not os.path.exists(best):
        print(f"[{name}] no best.pth - skipped"); continue
    if not (os.path.exists(ev) and os.path.getmtime(ev) >= os.path.getmtime(best)):      # missing, or older than the weights
        cmd = [PY, '/content/code/scripts/train_fusion.py', '--cache', '/content/cache', '--cache2d', CACHE2D, '--data', DATA,
               '--assoc-dir', ASSOC_DIR, '--litept', '/content/LitePT', '--backbone-ckpt', CKPT_LOCAL, '--work-dir', work,
               '--val-zones', ','.join(val_zones), '--eval-only']
        t0 = time.time(); tail = []
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        try:
            for line in p.stdout:                                        # the zone being loaded / evaluated, with the minutes so far
                if 'Warning' in line: continue
                tail = (tail + [line])[-40:]
                if line.startswith('[eval]') or line.startswith('[fusion]'): print(f"   {(time.time() - t0) / 60:5.1f} min  {line}", end='', flush=True)
        except KeyboardInterrupt:
            p.terminate(); p.wait(); print("stopped by user -> run this cell again to evaluate this head from the start"); raise
        p.wait()
        if p.returncode != 0: print(f"[{name}] full-zone evaluation FAILED (exit {p.returncode}); last lines:"); print(''.join(tail)[-2500:])
    if not os.path.exists(ev): continue
    e = json.load(open(ev)); b = f'{work}/eval_bank/eval.json'
    print(f"[{name}] full-zone evaluation: fused mIoU {e['mIoU']:.2f}  OA {e['OA']:.2f}   LiDAR-only {e['lidar_only']['mIoU']:.2f}"
          + (f"   | bank estimate {json.load(open(b))['mIoU']:.2f}" if os.path.exists(b) else ''))
    print('  fused     ', {k: (None if v is None else round(v, 1)) for k, v in e['per_class'].items()})
    print('  per zone  ', {z: round(v['mIoU'], 1) for z, v in e['per_zone'].items()})
"""),
code("""#@title 10. (optional) Shut the runtime down once everything above has finished - saves compute units on unattended runs
if SHUTDOWN_WHEN_DONE:
    from google.colab import drive, runtime
    print("flushing pending writes to Drive ..."); drive.flush_and_unmount()
    print("results are on Drive; shutting the runtime down"); runtime.unassign()
else:
    print("SHUTDOWN_WHEN_DONE is off -> remember to Disconnect and delete runtime")
"""),
md("""## หมายเหตุ
- ตัวเลขในตารางเป็น **bank estimate**: วัดบน subset แบบ stratified แล้วคูณ weight กลับเป็นทั้งโซน (= level 0 ของ robustness protocol)
  โดยใช้ LiDAR feature ที่เฉลี่ยข้าม window แล้ว ส่วน full-zone evaluation (cell 9, `colab_train_fusion`) ใช้ head กับทุก window แล้วเฉลี่ย vote
- ตรวจก่อนเชื่อตารางอื่น: แถว `lidar` ต้องได้ใกล้ 69.9 และ `imagevote` ใกล้ 65.5 (ต่างเกิน ~0.5 = มีอะไรผิด ให้หยุด);
  `late_v1` บน bank เทียบกับ 82.0 จาก full evaluation บอกว่าสองวิธีวัดต่างกันเท่าไร
- lab1 (13 พ.ย.): head ที่เทรนจาก bank ด้วย batch ตามความถี่จริงของคลาส (`late_nat_b1` 81.8) ได้เท่ากับ head ที่เทรนแบบ crop (`late_v1` 81.9)
  ส่วน `natural_frac=0.5` (head ชื่อ `*_b1` ที่เหลือ) เสียราว 1.4 mIoU → ตั้งแต่ update31 default คือ `natural_frac=1.0`
- ชื่อย่อใน lab1: `_lp` = head เห็น LiDAR เป็น class probabilities เท่านั้น (`lidar_input=prob`), `_ip` = เห็นภาพเป็น probabilities เท่านั้น
  (`image_input=prob`), `_ld50` = LiDAR dropout 50 % ตอนเทรน, `_s1` = seed 1; ใน lab2: `_n` = batch ตามความถี่จริง
- head ที่ `seed` และ `natural_frac` เดียวกันเห็น batch ชุดเดียวกัน (เทียบแบบ paired); `seed=1` ได้ batch + initialisation ชุดใหม่
- `natural_frac` = สัดส่วนของ batch ที่ sample ตามความถี่จริงของคลาส (ที่เหลือ sample แบบ uniform จาก bank = rare classes ถูกเน้น); default 1.0
- cell 6 พิมพ์ตารางจำนวนจุดต่อคลาสของทุกโซน (หน่วยพันจุด) ก่อนเทรน: ใช้ดูว่าคลาสหายากอย่าง water อยู่ในโซนไหนบ้าง
- pose-noise augmentation (`pose_t`, `pose_r` ใน spec): ในครึ่งหนึ่งของ iterations (`aug_prob`) กล้องทุกตัวได้ noise แบบ i.i.d. ที่ σ_t ~ U(0, `pose_t`) m และ
  σ_r ~ U(0, `pose_r`) ° แล้ว re-project แบบเดียวกับ protocol โหมด fast (cell 5 จะ copy LiDAR cache ของ train zones ให้เอง)
- cell 8 (`PROTOCOL`): ตารางหนึ่งแถวต่อโมเดล = mean IoU ทุก class / เฉลี่ย 4 electrical classes ที่ clean → reference level → level ใหญ่สุดของแต่ละ sweep;
  คอลัมน์ `clean` ต้องเท่ากับ mIoU ของตาราง validation ใน cell 7; รูปแสดงเฉพาะโมเดลใน `PROTOCOL_PLOT` (ทุกโมเดล: `protocol_<TAG>_all.png`);
  ถ้า Run all ซ้ำหลังเพิ่ม head ใน `SPECS` ให้ตั้ง `PROTOCOL_TAG` ใหม่ (results.json เดิมต่อได้เฉพาะชุดโมเดลเดิม)
- ผลของ gate G3 (run `fast_v1`): ที่ 10 cm / 0.1° ทุก head เสียไม่เกิน 0.3 mIoU; ที่ 50 cm / 1° head ระดับ feature (concat, window) เสีย thin classes
  7–11 จุด ส่วน late เสีย ≤ 1.4 → lab2 ถามว่า pose-noise augmentation ปิดช่องว่างนั้นได้ไหม และ window K=3 / K=5 ช่วยเพิ่มหรือไม่
- ผลของ lab2 (protocol `lab2`, 13 พ.ย., เฉลี่ยข้าม seeds; mIoU ทุก class / electrical classes ที่ clean → 50 cm → 1°):
  late 81.8/98.0 → 80.2/96.6 → 76.8/95.8 · concat 79.2/98.0 → 74.2/86.2 → 70.0/80.9 · window K=3 79.6/98.0 → 74.7/86.5 → 69.7/79.8 ·
  late+aug 82.1/98.0 → 81.3/97.8 → 78.8/97.8 · concat+aug 81.1/98.1 → 80.3/97.9 → 77.8/97.8 · window K=3+aug 80.7/98.1 → 79.9/97.9 → 77.5/97.9
  → augmentation คือ fix (clean ไม่ตก), window K=3 / K=5 ไม่ช่วยเพิ่ม
- cell 8 ตั้งแต่ update33: ตารางมีบรรทัด "median displacement of a projected point [px]" (จุดที่ project ลงภาพขยับกี่ pixel ที่แต่ละระดับ noise)
  และเขียน `confusion.json` + `results.json.gz` ไว้ข้าง `summary.json`
- ตรวจผลของ lab3: reference heads ใช้กล้องชุดเดียวกับ lab2 (draw ถูก seed ด้วยชื่อ sweep + ระดับ) จึงต้องได้ตัวเลขเดิม (clean | trans 0.5 | rot 1):
  `late_nat_b1` 81.8/98.0 | 80.7/97.5 | 77.6/97.1 · `late_a` 82.1/98.0 | 81.4/97.8 | 79.0/97.8 · `concat_n` 79.4/98.1 | 74.8/87.3 | 70.5/82.0 ·
  `concat_a` 81.7/98.1 | 80.8/97.9 | 78.2/97.8 (ต่างเกิน 0.1 = มีอะไรเปลี่ยน ให้บอกก่อนใช้ผล)
- ผลของ lab3 (13 พ.ย.): head ที่มี augmentation ครบ 3 seeds — late 82.05 ± 0.02, concat 80.65 ± 0.98, window K=3 80.57 ± 0.71 (clean) และ electrical 97.8–97.9 ที่ 50 cm / 1° ทุกตัว (G5: K×K ไม่ชนะ augmentation);
  `win5_n` (K=5 ไม่มี augmentation) เสีย thin classes น้อยกว่า concat / K=3 ราวครึ่งหนึ่ง (92.6 / 88.0 เทียบ 87.3 / 82.0) → lab4 เทรน seeds เพิ่ม; noise ≤10 cm ตอนเทรนก็ช่วยแล้ว (96.2 / 94.8),
  ชนิดของ noise ไม่สำคัญ, `concat_lp_a` = late + aug; full-zone evaluation ของ `late_a` = 82.06 (bank 82.07); ระยะขยับ median: 10 cm = 6.4 px, 50 cm = 31.9 px, 0.1° = 8.3 px, 1° = 82.8 px
- lab4: cell 8 รันโหมด **exact** (สร้าง association ใหม่ทุก draw) กับ 37 โมเดล — ช้ากว่าโหมด fast: บรรทัดความคืบหน้าขึ้นทุก ~2–3 นาที (หนึ่งระดับ noise = 10 draws);
  ถ้า session หลุด ให้ Run all ใหม่ได้เลย: heads ที่เทรนแล้วถูกข้าม และ protocol ต่อจากโซนที่ค้าง
- cell 8b: จำนวน views ตอน test (V views ที่ใกล้ที่สุดจาก 8 ที่เก็บไว้) กับ ImageVote + `VIEWS_HEADS` ในโหมด fast, 5 draws → ตารางเทียบ V ท้าย cell
- จบแล้ว **Disconnect and delete runtime**"""),
]

# ----------------------------------------------------------------------------- leaderboard notebook
leaderboard_cells = [
md("""# GridNet-HD — Leaderboard: ทำนาย test 9 zones แล้วสร้างไฟล์ส่ง (LiDAR-only + fused model)

**ต้องมีก่อน**
1. LiDAR cache ของ test 9 zones บน Drive (`GridNet/cache/<zone>/points.npz`) — สร้างด้วย `colab_build_cache` บน **CPU High-RAM** (ไม่ใช้ GPU; ต้องใช้โค้ดของ update35 ขึ้นไป)
2. checkpoint ทั้งสาม: LiDAR-only (`ckpt/litept_s_lidar_v2/best.pth`), UPerNet (`ckpt/upernet_convnext_t_v2/best.pth`), head (`fusion/late_a/best.pth`)
3. `GridNet/code/gridnet_code.zip` ของ update35 ขึ้นไป

**ทำอะไร** (ทีละโซน, resume ได้): โหลดภาพ + pose ของโซนจาก Hugging Face → export 2D cache ลง local disk → association บน GPU →
ทำนายด้วย frozen LitePT-S + head (forward ชุดเดียวกันให้ทั้ง **fused** และ **LiDAR-only**) → ส่ง class ของ voxel กลับไปที่จุดดิบทุกจุดตามลำดับใน LAS →
`GridNet/leaderboard/<TAG>/fused/<zone>.npz` และ `.../lidar/<zone>.npz` (array `data`, uint8, class 0–10) → ลบไฟล์ local ของโซนนั้น
2D cache ของ test zones ไม่ถูกเก็บลง Drive (ใช้ครั้งเดียว; ราว 24 MB ต่อภาพ)

**ตรวจก่อนเริ่ม** (cell 6, `CHECK_ZONE`): val zone หนึ่งโซนถูกรันผ่านเส้นทางเดียวกันทั้งหมด แล้วเทียบกับผล validation ที่มีอยู่
(ต้องได้ mIoU เท่าเดิมภายใน `CHECK_TOLERANCE`) + export 2D ใหม่ 6 ภาพเทียบกับ cache เดิม + วัด mIoU บน **จุดดิบ** จาก LAS
(leaderboard วัดบนจุดดิบ ส่วนตัวเลข validation ของเราวัดบน voxel 5 cm) — ถ้าไม่ผ่าน notebook หยุดก่อนถึง test zones

**Runtime**: A100 = ค่าที่ใช้กับตัวเลข validation ทุกตัว (window 900,000 จุด) · L4 + High-RAM ใช้ได้ด้วย window 500,000 จุด แต่ **ยังไม่เคยวัด**: ดูผลของ cell 6 ก่อน
เวลาที่วัดได้บน A100-40GB (14 พ.ย. 2026; test = 767 ล้านจุดดิบ = 524 ล้าน voxel, 2,447 ภาพ): ตรวจ 4 นาที, test 9 zones 107 นาที = 6–22 นาทีต่อโซน
(export 2D 1.4 s/ภาพ จึงขึ้นกับจำนวนภาพ: สองโซนของพื้นที่ t6 มีโซนละกว่า 600 ภาพ ใช้โซนละ 21 นาที; ทำนาย 2.5–6.7 นาทีต่อโซน) → ทั้ง notebook ราว 1.9 ชม.

**ส่งจริง**: ตรวจผลของ notebook ที่รันแล้วก่อน แล้วอัปโหลดไฟล์ `.npz` ทั้ง 9 ไฟล์ของโฟลเดอร์ `lidar/` (sanity check) และ `fused/` ที่
https://huggingface.co/spaces/heig-vd-geo/GridNet-HD-Leaderboard → Sign in → Submit Eval · **ไม่ติ๊ก** "Remap classes" (ไฟล์เป็น class 0–10 ที่ remap แล้ว)"""),
code("""#@title 1. GPU (A100 = the setting of every validation number; an L4 works with smaller windows - see CHECK_ZONE)
import subprocess, torch
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8, "need an A100 or an L4 (flash-attn)"
VRAM_MB = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
print("windows of 900,000 points (the validation setting)" if VRAM_MB >= 30000 else
      "smaller GPU: windows of 500,000 points - NOT the validation setting; cell 6 shows what that changes on a validation zone")
"""),
code("""#@title 2. Mount Drive + parameters
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/GridNet'
CODE_ZIP = f'{DRIVE}/code/gridnet_code.zip'      #@param {type:"string"}
TAG = 'v1'                                       #@param {type:"string"}
HEAD = 'late_a'                                  #@param {type:"string"}
RUN2D = 'upernet_convnext_t_v2'                  #@param {type:"string"}
BACKBONE_CKPT = f'{DRIVE}/ckpt/litept_s_lidar_v2/best.pth'   #@param {type:"string"}
ZONES = 't1z4,t1z5a,t1z7,t3z1,t3z2,t3z5,t5a2,t6z1,t6z5'      #@param {type:"string"}
# a validation zone that goes through the same path first ('' = skip); t1z8 is the smallest and has the largest fusion gain
CHECK_ZONE = 't1z8'                              #@param {type:"string"}
CHECK_TOLERANCE = 0.5                            #@param {type:"number"}
EVAL_MAX_POINTS = 0                              #@param {type:"integer"}
EXPORT_BATCH = 0                                 #@param {type:"integer"}
SHUTDOWN_WHEN_DONE = False                       #@param {type:"boolean"}
CACHE_DRIVE = f'{DRIVE}/cache'; CACHE2D_DRIVE = f'{DRIVE}/cache2d/{RUN2D}'; CKPT2D_DRIVE = f'{DRIVE}/ckpt/{RUN2D}'
HEAD_CKPT = f'{DRIVE}/fusion/{HEAD}/best.pth'; HEAD_EVAL = f'{DRIVE}/fusion/{HEAD}/eval'
OUT = f'{DRIVE}/leaderboard/{TAG}'
zones = [z.strip() for z in ZONES.split(',') if z.strip()]
if not EVAL_MAX_POINTS: EVAL_MAX_POINTS = 900000 if VRAM_MB >= 30000 else 500000     # 0 = by the GPU
if not EXPORT_BATCH: EXPORT_BATCH = 4 if VRAM_MB >= 30000 else 2
for p in (BACKBONE_CKPT, HEAD_CKPT, f'{CKPT2D_DRIVE}/best.pth'):
    assert os.path.exists(p), f"missing {p}"
missing = [z for z in zones + ([CHECK_ZONE] if CHECK_ZONE else []) if not os.path.exists(f'{CACHE_DRIVE}/{z}/points.npz')]
assert not missing, f"no LiDAR cache on Drive for {missing}: run colab_build_cache (CPU High-RAM runtime) first"
os.makedirs(OUT, exist_ok=True)
done = [z for z in zones if all(os.path.exists(f'{OUT}/{m}/{z}.npz') for m in ('fused', 'lidar'))]
print(f"head {HEAD}, 2D model {RUN2D}, windows of {EVAL_MAX_POINTS:,} points, 2D export batch {EXPORT_BATCH}")
print(f"output -> {OUT}/fused and {OUT}/lidar")
print(f"{len(done)}/{len(zones)} test zones have their files on Drive:", done)
print("to do:", [z for z in zones if z not in done])
"""),
code("""#@title 3. Environment (restore tarball from Drive; + timm, laspy)
import subprocess, os, time
ENV = '/content/env'; TAR = f'{DRIVE}/env/env_py310_torch241_cu124.tar.gz'
t0 = time.time()
if not os.path.exists(f'{ENV}/bin/python'):
    assert os.path.exists(TAR), "no env tarball on Drive: run colab_train_litept cell 3 once"
    subprocess.run(['tar', '-C', '/content', '-xzf', TAR], check=True); print("env restored from Drive")
PY = f'{ENV}/bin/python'
subprocess.run([PY, '-m', 'pip', 'install', '-q', 'timm', 'huggingface_hub', 'pillow', 'laspy[lazrs]'], check=True)
print(f"env ready in {(time.time() - t0) / 60:.1f} min")
"""),
code("""#@title 4. Code + LitePT
import subprocess, zipfile, os
os.makedirs('/content/code', exist_ok=True); zipfile.ZipFile(CODE_ZIP).extractall('/content/code')
assert os.path.exists('/content/code/scripts/predict_zones.py'), "gridnet_code.zip on Drive is older than update35: pack and upload it again"
if not os.path.exists('/content/LitePT'):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/prs-eth/LitePT.git", "/content/LitePT"], check=True)
r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0,'/content/LitePT'); sys.path.insert(0,'/content/code'); "
                    "from litept.model import LitePT; import gridnet.train.predict, gridnet.train.engine2d, gridnet.submission, timm, laspy; "
                    "print('imports ok')"], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-1500:])
assert 'imports ok' in r.stdout
"""),
code("""#@title 5. Checkpoints -> local disk; helpers
import os, sys, shutil, subprocess, time, json
sys.path.insert(0, '/content/code')
os.makedirs('/content/ckpt2d', exist_ok=True)
for src, dst in ((BACKBONE_CKPT, '/content/backbone_best.pth'), (f'{CKPT2D_DRIVE}/best.pth', '/content/ckpt2d/best.pth')):
    if not os.path.exists(dst): shutil.copy(src, dst)
ENVV = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            HF_HUB_ENABLE_HF_TRANSFER='0')
try:
    from google.colab import userdata
    tok = userdata.get('HF_TOKEN')                  # optional (public dataset)
    if tok: ENVV['HF_TOKEN'] = tok
except Exception: pass

def run(cmd, show=lambda l: True, tag=''):
    # run a script, print the lines `show` accepts; on failure print its last lines. Returns the exit code.
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=ENVV)
    tail = []
    try:
        for line in p.stdout:
            tail.append(line); del tail[:-25]
            if 'Warning' in line or 'def backward' in line or '_TORCH_CUSTOM_FWD' in line: continue
            if show(line): print(tag + line, end='', flush=True)
    except KeyboardInterrupt:
        p.terminate(); p.wait(); print("stopped by user; finished zones are on Drive -> run the cell again to continue"); raise
    p.wait()
    if p.returncode != 0: print(f"{tag}exit {p.returncode}; last lines:"); print(''.join(tail[-15:]))
    return p.returncode

def fetch(zone, what, dst='/content/data'):
    # <zone>/<what>/* from the Hugging Face dataset (files already there are skipped)
    code = ("import sys; sys.path.insert(0,'/content/code'); from gridnet.train.data2d import fetch_zones; "
            f"fetch_zones(['{zone}'], '{dst}', what={tuple(what)!r})")
    return run([PY, '-c', code], show=lambda l: False)

def export_line(l):
    # of the 2D export: the first line of a zone, every 60th image and the last one
    if not l.startswith('[export]'): return 'Error' in l or 'Traceback' in l
    p = l.split()
    if len(p) > 2 and '/' in p[2]:
        k, n = p[2].split('/'); return k == n or int(k) % 60 == 0
    return True

def export2d(zone, data='/content/data', out='/content/cache2d'):
    return run([PY, '/content/code/scripts/train_upernet.py', '--data', data, '--ckpt-dir', '/content/ckpt2d', '--batch', str(EXPORT_BATCH),
                '--export-cache', out, '--export-zones', zone], show=export_line, tag='   ')

def predict(zone, out):
    return run([PY, '/content/code/scripts/predict_zones.py', '--cache', '/content/cache', '--cache2d', '/content/cache2d', '--data', '/content/data',
                '--assoc-dir', '/content/assoc', '--litept', '/content/LitePT', '--backbone-ckpt', '/content/backbone_best.pth', '--head', HEAD_CKPT,
                '--out', out, '--zones', zone, '--eval-max-points', str(EVAL_MAX_POINTS)],
               show=lambda l: l.startswith(('[predict]', '[eval]', '[fusion]')) or 'Error' in l or 'Traceback' in l, tag='   ')

def export(zone, pred_root, name, out, las_root=None):
    cmd = [PY, '/content/code/scripts/export_submission.py', '--cache', '/content/cache', '--pred', pred_root + '/{zone}/' + name + '_pred.npy',
           '--out', out, '--zones', zone, '--name', f'{name}: ' + (f'{HEAD} on {RUN2D}' if name == 'fused' else 'LiDAR-only LitePT-S') + f', windows {EVAL_MAX_POINTS}']
    if las_root: cmd += ['--las-root', las_root]
    return run(cmd, show=lambda l: l.startswith('[export]') or l.startswith('   ') or 'Error' in l or 'Traceback' in l, tag='   ')

def local_cache(zone):
    os.makedirs(f'/content/cache/{zone}', exist_ok=True)
    if not os.path.exists(f'/content/cache/{zone}/points.npz'): shutil.copy(f'{CACHE_DRIVE}/{zone}/points.npz', f'/content/cache/{zone}/points.npz')

def clean(zone):
    for d in (f'/content/data/{zone}', f'/content/cache2d/{zone}', f'/content/cache/{zone}', f'/content/assoc/{zone}'):
        shutil.rmtree(d, ignore_errors=True)
print("ready;", subprocess.run(['df', '-h', '--output=avail', '/content'], capture_output=True, text=True).stdout.split()[-1], "free on the local disk")
"""),
code("""#@title 6. Check on a validation zone (CHECK_ZONE): the same path end to end, compared with the validation run of the head (~10 min; stops the notebook when it does not reproduce)
import numpy as np
if CHECK_ZONE:
    from gridnet.train.maps2d import cached_image_ids, extract_zone_tar, zone_tar_manifest
    z, t0, check = CHECK_ZONE, time.time(), {'zone': CHECK_ZONE, 'eval_max_points': EVAL_MAX_POINTS, 'vram_mb': VRAM_MB}
    shutil.rmtree('/content/check', ignore_errors=True)
    local_cache(z)
    assert zone_tar_manifest(CACHE2D_DRIVE, z, wait=30) is not None, f"no 2D cache tar for {z} in {CACHE2D_DRIVE}"
    n = extract_zone_tar(CACHE2D_DRIVE, z, '/content/cache2d')
    assert fetch(z, ('pose', 'lidar')) == 0, "download from Hugging Face failed"
    print(f"[check {z}] LiDAR cache, 2D cache of {n} images, camera files and LAS ready ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # (a) the 2D export of today against the cache the validation numbers were made with
    ids = cached_image_ids('/content/cache2d', z)[:6]
    pats = [f'{z}/images/{i}.JPG' for i in ids]
    code = ("from huggingface_hub import snapshot_download; "
            f"snapshot_download('heig-vd-geo/GridNet-HD', repo_type='dataset', allow_patterns={pats!r}, local_dir='/content/check/data')")
    assert run([PY, '-c', code], show=lambda l: False) == 0, "image download failed"
    assert export2d(z, data='/content/check/data', out='/content/check/cache2d') == 0, "2D export failed"
    agree, mad = [], []
    for i in ids:
        a = np.load(f'/content/check/cache2d/{z}/{i}.npz')['prob']; b = np.load(f'/content/cache2d/{z}/{i}.npz')['prob']
        if b.dtype != np.uint8: b = np.clip(np.round(b.astype(np.float32) * 255), 0, 255).astype(np.uint8)
        agree.append(float((a.argmax(0) == b.argmax(0)).mean())); mad.append(float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) / 255)
    check['export2d'] = {'images': len(ids), 'same_class': float(np.mean(agree)), 'mean_abs_prob_diff': float(np.mean(mad))}
    print(f"[check {z}] 2D export of {len(ids)} images vs the stored cache: same class in {100 * np.mean(agree):.3f} % of the pixels, "
          f"mean |difference| of the probabilities {np.mean(mad):.5f}", flush=True)
    # (b) prediction: must reproduce the full-zone evaluation of the head on this zone
    assert predict(z, '/content/check/pred') == 0, "prediction failed"
    rec = json.load(open(f'/content/check/pred/{z}/predict.json'))
    got = {m: rec[m]['metrics']['mIoU_present'] for m in ('fused', 'lidar')}
    ev = json.load(open(f'{HEAD_EVAL}/eval.json')) if os.path.exists(f'{HEAD_EVAL}/eval.json') else None
    want = None
    if ev and z in ev.get('per_zone', {}):
        want = {'fused': ev['per_zone'][z]['mIoU'], 'lidar': ev['per_zone'][z]['lidar_only_mIoU']}
    old = f'{HEAD_EVAL}/{z}/fused_pred.npy'
    same = float((np.load(old) == np.load(f'/content/check/pred/{z}/fused_pred.npy')).mean()) if os.path.exists(old) else None
    check.update(predicted=got, validation_run=want, same_voxels=same, minutes_predict=rec['minutes'])
    for m in ('fused', 'lidar'):
        print(f"[check {z}] {m:5s} mIoU per voxel {got[m]:.2f}" + (f"   validation run {want[m]:.2f}   difference {got[m] - want[m]:+.2f}" if want else "   (no validation run to compare with)"))
    if same is not None: print(f"[check {z}] fused prediction identical to the validation run on {100 * same:.3f} % of the voxels")
    # (c) leaderboard files + score on the raw points of the LAS (what the leaderboard measures)
    for m in ('fused', 'lidar'):
        assert export(z, '/content/check/pred', m, f'/content/check/{m}', las_root='/content/data') == 0, "export failed"
        r = json.load(open(f'/content/check/{m}/manifest.json'))['zones'][z]
        check[m] = {'points': r['points'], 'voxels': r['voxels'], 'voxel_mIoU': r['voxel_metrics']['mIoU_present'], 'raw_mIoU': r['raw_metrics']['mIoU_present'],
                    'voxel_OA': r['voxel_metrics']['OA'], 'raw_OA': r['raw_metrics']['OA'], 'raw_per_class': r['raw_metrics']['per_class'],
                    'voxel_per_class': r['voxel_metrics']['per_class']}
        print(f"[check {z}] {m:5s} per voxel {check[m]['voxel_mIoU']:.2f} mIoU / {check[m]['voxel_OA']:.2f} OA   per raw point {check[m]['raw_mIoU']:.2f} mIoU / "
              f"{check[m]['raw_OA']:.2f} OA   ({r['points']:,} points, {r['voxels']:,} voxels)")
    check['minutes'] = (time.time() - t0) / 60
    json.dump(check, open(f'{OUT}/check_{z}.json', 'w'), indent=1)
    clean(z); shutil.rmtree('/content/check', ignore_errors=True)
    bad = [m for m in ('fused', 'lidar') if want and abs(got[m] - want[m]) > CHECK_TOLERANCE]
    if bad:
        raise RuntimeError(f"the check zone does not reproduce its validation numbers ({bad}): the test zones are NOT predicted. "
                           "Send this output; to go on anyway raise CHECK_TOLERANCE or empty CHECK_ZONE")
    print(f"[check {z}] passed in {check['minutes']:.1f} min -> {OUT}/check_{z}.json")
else:
    print("CHECK_ZONE is empty: no check")
"""),
code("""#@title 7. Test zones, one at a time (resumable): images + poses from Hugging Face -> 2D cache -> prediction -> leaderboard files on Drive
failed, timing = {}, {}
for z in zones:
    if all(os.path.exists(f'{OUT}/{m}/{z}.npz') for m in ('fused', 'lidar')):
        print(f"[{z}] files exist on Drive (skipped)"); continue
    t0 = time.time(); last = [t0]; step = {}
    def lap(name):
        now = time.time(); step[name] = (now - last[0]) / 60; last[0] = now
    try:
        local_cache(z); lap('cache')
        assert fetch(z, ('images', 'pose')) == 0, "download from Hugging Face failed"
        n_img = len([f for f in os.listdir(f'/content/data/{z}/images') if f.upper().endswith('.JPG')]); lap('download')
        print(f"[{z}] {n_img} images downloaded in {step['download']:.1f} min -> 2D cache", flush=True)
        assert export2d(z) == 0, "2D export failed"; lap('export2d')
        assert predict(z, f'{OUT}/pred') == 0, "prediction failed"; lap('predict')
        for m in ('fused', 'lidar'):
            assert export(z, f'{OUT}/pred', m, f'{OUT}/{m}') == 0, f"export of the {m} file failed"
        lap('files')
    except AssertionError as e:
        failed[z] = str(e); print(f"[{z}] FAILED: {e}  (its local files are kept: run the cell again to retry)", flush=True)
        continue
    clean(z); timing[z] = step
    print(f"[{z}] done in {(time.time() - t0) / 60:.1f} min  (" + ', '.join(f"{k} {v:.1f}" for k, v in step.items()) + ")", flush=True)
if timing: json.dump(timing, open(f'{OUT}/timing_{int(time.time())}.json', 'w'), indent=1)
print("FAILED zones:", failed if failed else "none")
"""),
code("""#@title 8. Summary: what is on Drive and what to upload
# LAS points of every test zone, read when the caches were built (report.txt in GridNet/cache/<zone>/, 2026-11-14).
# Their sum, 766,947,526, is 1.0 % more than the 759,222,245 points the dataset card gives for the test split (reason
# not known). A submission needs one value per LAS point, so the LAS counts are what is checked, zone by zone.
EXPECTED_POINTS = {'t1z4': 49_527_865, 't1z5a': 71_040_345, 't1z7': 122_763_604, 't3z1': 59_879_633, 't3z2': 67_365_114,
                   't3z5': 89_594_873, 't5a2': 95_358_184, 't6z1': 107_067_772, 't6z5': 104_350_136}
ALL_TEST = list(EXPECTED_POINTS)
complete, missing, differs = {}, {}, {}
for m in ('lidar', 'fused'):
    path = f'{OUT}/{m}/manifest.json'
    if not os.path.exists(path):
        print(f"{m}: no file yet"); continue
    man = json.load(open(path)); zs = man['zones']
    print(f"=== {m}  ({man.get('name', '')})  ->  {OUT}/{m}/ ===")
    print(f"{'zone':7s}{'points':>14s}{'voxels':>14s}{'MB':>7s}   predicted classes [%]  " + ' '.join(f"{c[:6]:>6s}" for c in man['class_names']))
    tot, missing[m], differs[m] = 0, [], []
    for z in ALL_TEST:
        if z not in zs or not os.path.exists(f'{OUT}/{m}/{z}.npz'):
            missing[m].append(z); print(f"{z:7s}   MISSING"); continue
        r = zs[z]; tot += r['points']
        if r['points'] != EXPECTED_POINTS[z]: differs[m].append(z)
        print(f"{z:7s}{r['points']:14,d}{r['voxels']:14,d}{r['bytes'] / 1e6:7.1f}" + ' ' * 26 + ' '.join(f"{100 * c / r['points']:6.2f}" for c in r['counts'])
              + ("" if r['points'] == EXPECTED_POINTS[z] else f"   <- the LAS of this zone has {EXPECTED_POINTS[z]:,} points"))
    complete[m] = not missing[m] and not differs[m]
    print(f"{len(ALL_TEST) - len(missing[m])}/{len(ALL_TEST)} zones, {tot:,} points" + (" = one prediction for every LAS point of the nine test zones" if complete[m]
          else f"   (the nine LAS files hold {sum(EXPECTED_POINTS.values()):,} points)"))
    print()
if any(differs.values()):
    print("STOP: the number of predictions differs from the number of LAS points in", {m: z for m, z in differs.items() if z})
    print("Do not submit these files. Find the cause first.")
elif all(complete.get(m) for m in ('lidar', 'fused')):
    print("Both sets are complete. Check the output above first; then upload the 9 files of ONE folder per submission")
    print("(lidar/ first: it checks the whole export path. Published LiDAR-only baselines: SPT 66.91, PTv3 69.32; an unpublished")
    print(" leaderboard entry named Lite-S-test has 72.88. Then fused/.)")
    print("at https://huggingface.co/spaces/heig-vd-geo/GridNet-HD-Leaderboard -> Sign in -> Submit Eval; leave 'Remap classes' UNCHECKED.")
else:
    print("NOT complete yet -> run cell 7 again (finished zones are skipped)")
"""),
code("""#@title 9. (optional) Shut the runtime down once everything above has finished - saves compute units on unattended runs
if SHUTDOWN_WHEN_DONE:
    from google.colab import drive, runtime
    print("flushing pending writes to Drive ..."); drive.flush_and_unmount()
    print("results are on Drive; shutting the runtime down"); runtime.unassign()
else:
    print("SHUTDOWN_WHEN_DONE is off -> remember to Disconnect and delete runtime")
"""),
md("""## หมายเหตุ
- รูปแบบไฟล์ของ leaderboard: หนึ่งไฟล์ต่อ test zone ชื่อ `<zone>.npz` มี array ชื่อ `data` (uint8) = class ของ **ทุกจุดใน LAS ตามลำดับในไฟล์**
  (เหมือนที่ `create_npz` ในหน้า leaderboard เขียน: `np.savez_compressed(path, data=...)`) · เราส่ง class 0–10 ที่ remap แล้ว จึงไม่ติ๊ก "Remap classes"
- จาก voxel ไปจุดดิบ: โมเดลทำนายหนึ่ง class ต่อ voxel 5 cm; `points.npz` เก็บ `inverse` (จุดดิบ → voxel) ไว้ตั้งแต่ตอนสร้าง cache → จุดดิบทุกจุดได้ class ของ voxel ของมัน
  ไฟล์ถูกอ่านกลับมาตรวจทุกครั้ง (จำนวนจุด, dtype, ช่วงของ class) และ cell 8 เทียบจำนวนจุดของแต่ละโซนกับจำนวนจุดใน LAS ของโซนนั้น (รวม 766,947,526 จุด)
- LiDAR-only กับ fused มาจาก forward pass ชุดเดียวกัน (window, seed, vote เหมือน full-zone evaluation ของ `late_a` ที่ได้ 82.06 บน val)
- `GridNet/leaderboard/<TAG>/pred/<zone>/` เก็บ prediction ระดับ voxel (`fused_pred.npy`, `lidar_pred.npy`) + `predict.json` ไว้ใช้ทำรูป qualitative ภายหลัง
- cell 6 เขียน `check_<zone>.json`: ตัวเลขต่อ voxel เทียบกับต่อจุดดิบของ val zone = ช่องว่างที่ควรคาดไว้ระหว่างตัวเลข validation ของเรากับตัวเลขบน leaderboard (นอกเหนือจากการที่เป็นคนละโซน)
- ถ้า session หลุด: Run all ใหม่ได้เลย โซนที่มีไฟล์บน Drive แล้วถูกข้าม (cell 6 จะรันซ้ำ ~10 นาที; ตั้ง `CHECK_ZONE` ว่างเพื่อข้าม)
- ส่ง leaderboard แค่ 2 ครั้งตามแผน: LiDAR-only (sanity) และ fused model ตัวสุดท้าย
- จบแล้ว **Disconnect and delete runtime**"""),
]

# ----------------------------------------------------------------------------- label transfer notebook (CPU)
label_cells = [
md("""# GridNet-HD — label transfer ภายใต้ pose error ทุกภาพของ val zones (CPU, ไม่ใช้ GPU)

การทดลองของรูป motivating ใน paper ทำใหม่ให้ครบทั้งโซน: จุดที่กล้องแต่ละภาพมองเห็น รับ label จาก pixel ของ mask (ของผู้สร้าง dataset) ที่มัน project ไปตก
ครั้งแรกด้วย pose ที่ให้มากับ dataset แล้วด้วย pose ที่ถูกรบกวน (ตำแหน่ง 2–50 ซม., มุม 0.05–1°) — ไม่มี network เกี่ยวข้อง
รูปเดิมใน paper มาจาก **ภาพเดียว** ของ t5b5; notebook นี้ทำ **ทุกภาพที่มี mask ของ 6 val zones** (ประมาณ 850 ภาพ) ด้วย error model เดียวกับ protocol (σ ต่อแกน)

**Runtime: CPU + High-RAM** (Runtime → Change runtime type → CPU, High-RAM) — 8 cores ทำงานขนานกัน
เวลาที่วัดได้ (14 พ.ย. 2026, High-RAM 8 cores): 1.75 ชม. สำหรับ 6 โซน 852 ภาพ (12.5–25.8 นาทีต่อโซน); CPU ธรรมดา (2 cores) ช้ากว่า ~4 เท่า → ตั้ง `STRIDE = 3`
ใช้ compute units น้อยมาก (ไม่มี GPU) · หยุดกลางทางได้: โซนที่เสร็จแล้วอยู่บน Drive และถูกข้ามเมื่อรันใหม่

**ก่อนรัน**: `GridNet/code/gridnet_code.zip` บน Drive ต้องเป็นของ **update36 ขึ้นไป** (cell 2 ตรวจให้)
**ผลลัพธ์**: `GridNet/label_transfer/<TAG>/<zone>/label_transfer.csv` ต่อโซน และ `.../all/label_transfer.csv` + รูป รวมทุกโซน — เสร็จแล้วผลอยู่บน Drive"""),
code("""#@title 1. Mount Drive + parameters
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/GridNet'
CODE_ZIP = f'{DRIVE}/code/gridnet_code.zip'      #@param {type:"string"}
CACHE_DRIVE = f'{DRIVE}/cache'
TAG = 'v1'                                       #@param {type:"string"}
ZONES = 't1z5b,t1z8,t3z4,t4z1,t5b1,t5b5'         #@param {type:"string"}
NOISE = 'protocol'                               #@param ["protocol", "fixed"]
DRAWS = 1                                        #@param {type:"integer"}
STRIDE = 1                                       #@param {type:"integer"}
WORKERS = 0                                      #@param {type:"integer"}
SHUTDOWN_WHEN_DONE = False                       #@param {type:"boolean"}
OUT = f'{DRIVE}/label_transfer/{TAG}'
zones = [z.strip() for z in ZONES.split(',') if z.strip()]
missing = [z for z in zones if not os.path.exists(f'{CACHE_DRIVE}/{z}/points.npz')]
assert not missing, f"no LiDAR cache on Drive for {missing}"
os.makedirs(OUT, exist_ok=True)
cores = os.cpu_count(); ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30
print(f"{cores} CPU cores, {ram:.0f} GB RAM" + ("" if ram > 20 else "   <- standard runtime: slow (2 cores); High-RAM is recommended, or set STRIDE = 3"))
done = [z for z in zones if os.path.exists(f'{OUT}/{z}/label_transfer.json')]
print(f"error model {NOISE}, {DRAWS} draw(s) per image and level, every {STRIDE}. image  ->  {OUT}")
print(f"{len(done)}/{len(zones)} zones already on Drive:", done)
"""),
code("""#@title 2. Code
import subprocess, sys, zipfile, os
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "pillow", "matplotlib"], check=True)
os.makedirs('/content/code', exist_ok=True)
zipfile.ZipFile(CODE_ZIP).extractall('/content/code')
assert os.path.exists('/content/code/scripts/label_transfer_zone.py'), "gridnet_code.zip on Drive is older than update36: pack and upload it again"
r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0,'/content/code'); import gridnet.label_transfer, gridnet.io; print('imports ok')"],
                   capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-1500:])
assert 'imports ok' in r.stdout
"""),
code("""#@title 3. One zone at a time: points + masks + poses -> label transfer -> Drive (resumable)
import subprocess, sys, os, shutil, time, json
from huggingface_hub import snapshot_download
SCRIPT = '/content/code/scripts/label_transfer_zone.py'
LOCAL = f'/content/label_transfer/{TAG}_{NOISE}_{DRAWS}'      # one local folder per setting: records of another setting are never mixed in

def run(cmd):
    # run the script and show its progress lines; returns the exit code
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dict(os.environ, PYTHONPATH='/content/code'))
    tail = []
    try:
        for line in p.stdout:
            tail.append(line); del tail[:-25]
            print('   ' + line, end='', flush=True)
    except KeyboardInterrupt:
        p.terminate(); p.wait(); print("stopped by user; finished zones are on Drive -> run the cell again to continue"); raise
    p.wait()
    return p.returncode

def settings_match(zone):
    # a zone on Drive counts as done only when it was run with the settings of cell 1
    path = f'{OUT}/{zone}/label_transfer.json'
    if not os.path.exists(path): return False
    d = json.load(open(path)); s = d['settings']
    full = d['images'] == len(range(0, d['images_with_mask'], STRIDE))
    return s['noise'] == NOISE and s['draws'] == DRAWS and full

t_all = time.time()
for z in zones:
    if settings_match(z):
        print(f"[skip] {z}: on Drive"); continue
    t0 = time.time()
    os.makedirs(f'/content/cache/{z}', exist_ok=True)
    if not os.path.exists(f'/content/cache/{z}/points.npz'):
        shutil.copy(f'{CACHE_DRIVE}/{z}/points.npz', f'/content/cache/{z}/points.npz')
    snapshot_download("heig-vd-geo/GridNet-HD", repo_type="dataset", allow_patterns=[f"{z}/masks/*", f"{z}/pose/*"], local_dir="/content/data")
    n_masks = len(os.listdir(f'/content/data/{z}/masks'))
    print(f"[{z}] points + {n_masks} masks + poses on local disk in {(time.time() - t0) / 60:.1f} min", flush=True)
    code = run([sys.executable, SCRIPT, '--cache', '/content/cache', '--data', '/content/data', '--zone', z, '--out', LOCAL,
                '--noise', NOISE, '--draws', str(DRAWS), '--stride', str(STRIDE), '--workers', str(WORKERS)])
    if code != 0:
        print(f"[fail] {z} (exit {code}) - its finished images stay in {LOCAL}/{z}; run the cell again"); continue
    os.makedirs(f'{OUT}/{z}', exist_ok=True)
    for f in os.listdir(f'{LOCAL}/{z}'):
        shutil.copy(f'{LOCAL}/{z}/{f}', f'{OUT}/{z}/{f}')
    shutil.rmtree(f'/content/data/{z}', ignore_errors=True); os.remove(f'/content/cache/{z}/points.npz')
    print(f"[{z}] done in {(time.time() - t0) / 60:.1f} min  ->  {OUT}/{z}", flush=True)
ok = [z for z in zones if settings_match(z)]
print(f"finished {len(ok)}/{len(zones)} zones in {(time.time() - t_all) / 60:.1f} min" + ("" if len(ok) == len(zones) else "  -> run this cell again"))
"""),
code("""#@title 4. All zones pooled: table + figure -> Drive
import subprocess, sys, os, shutil, json
from IPython.display import Image, display
ok = [z for z in zones if settings_match(z)]
assert ok, "no finished zone yet"
for z in ok:                                    # the pooled result is computed from the copies on Drive
    if not os.path.exists(f'{LOCAL}/{z}/label_transfer.json') or not os.path.exists(f'{LOCAL}/{z}/records.jsonl'):
        shutil.rmtree(f'{LOCAL}/{z}', ignore_errors=True); shutil.copytree(f'{OUT}/{z}', f'{LOCAL}/{z}')
assert run([sys.executable, SCRIPT, '--out', LOCAL, '--combine', ','.join(ok)]) == 0
assert run([sys.executable, '/content/code/scripts/plot_label_transfer.py', '--csv', f'{LOCAL}/all/label_transfer.csv',
            '--out', f'{LOCAL}/all/label_transfer_val']) == 0
os.makedirs(f'{OUT}/all', exist_ok=True)
for f in os.listdir(f'{LOCAL}/all'):
    shutil.copy(f'{LOCAL}/all/{f}', f'{OUT}/all/{f}')
display(Image(f'{LOCAL}/all/label_transfer_val.png', width=900))
d = json.load(open(f'{LOCAL}/all/label_transfer.json'))
print(f"{len(ok)}/{len(zones)} zones, {d['images']} images  ->  {OUT}/all/  (label_transfer.csv, label_transfer.json, label_transfer_val.pdf/.png)")
if len(ok) < len(zones): print("NOT all zones yet:", [z for z in zones if z not in ok], "-> run cell 3 again, then this cell")
"""),
code("""#@title 5. (optional) Shut the runtime down once everything above has finished
if SHUTDOWN_WHEN_DONE:
    from google.colab import drive, runtime
    print("flushing pending writes to Drive ..."); drive.flush_and_unmount()
    print("results are on Drive; shutting the runtime down"); runtime.unassign()
else:
    print("SHUTDOWN_WHEN_DONE is off -> remember to Disconnect and delete runtime")
"""),
md("""## หมายเหตุ
- นิยาม (เหมือนการทดลองภาพเดียวเดิม): จุดที่ "มองเห็น" = ผ่าน z-buffer ของกล้องที่ pose ถูกรบกวน (depth map 1/4, tolerance 0.5 ม. เท่ากับขั้น association ของ pipeline);
  ต่อ class = สัดส่วนจุดของ class นั้นที่ pixel ที่ตกมี label เดียวกัน (ตกบน pixel ที่ไม่มี label เช่นสายไฟตกบนท้องฟ้า = พลาด); `all` = ทุกจุดที่ตกบน pixel ที่มี label
- จุดที่ใช้คือจุด voxel 5 ซม. ของ cache (label ข้างมาก) ไม่ใช่จุดดิบของ LAS — ตัวเลขของภาพเดิมจึงอาจต่างจาก 94.5 % เล็กน้อย
- `NOISE = protocol`: ตำแหน่งกล้อง + N(0, σ_t²) ต่อแกน, หมุนด้วย axis-angle ที่มุมต่อแกน ~ N(0, σ_r²) = error model ของ robustness protocol (แกน x ของรูปจึงเทียบกับรูป benchmark ได้ตรง ๆ);
  `fixed` = ระยะ/มุมคงที่ในทิศสุ่ม (แบบรูปเดิม)
- หนึ่ง draw ต่อภาพต่อระดับ: ระดับหนึ่ง ๆ จึงเป็นผลรวมของ draw อิสระเท่าจำนวนภาพ (~850); แถบในรูป = ควอร์ไทล์ระหว่างภาพ
- seed ของแต่ละ draw ขึ้นกับ (ชื่อภาพ, ชนิด error, ระดับ, ลำดับ draw) เท่านั้น → รันซ้ำ/รันต่อ/เลือกภาพบางส่วน ได้ตัวเลขเดิม
- จบแล้ว **Disconnect and delete runtime**"""),
]

# ----------------------------------------------------------------------------- qualitative figure notebook
qual_cells = [
md("""# GridNet-HD — ข้อมูลของรูป qualitative: prediction รอบเสาไฟฟ้า ภายใต้ pose error (GPU, session สั้น)

รูปใน paper ที่ยังขาด: ให้ผู้อ่าน **เห็น** ว่าเมื่อ pose ของกล้องผิด head ระดับ feature ที่เทรนด้วย pose ที่ให้มา ทำสายไฟ/สายยึดหายไปตรงไหน และ head ตัวเดียวกันที่เทรนด้วย
pose-noise augmentation ไม่เป็น — notebook นี้ไม่ได้เทรนอะไร และไม่ได้ให้ตัวเลขใหม่ของ paper: มันรัน robustness protocol **เฉพาะ draw 0** ของระดับ error ที่เลือก
(`LEVELS`) แล้วเก็บ prediction ของจุดรอบเสาไฟฟ้าที่ใหญ่ที่สุด `PYLONS` ต้นของแต่ละโซน (รัศมี `RADIUS` ม.) เป็นไฟล์เล็ก ๆ ไว้บน Drive

- กล้องของ draw 0 ถูก seed ด้วย (ชื่อ sweep, ระดับ, ลำดับ draw) → เป็น **กล้องชุดเดียวกับ draw 0 ของ run `exact_v1`** ที่ตัวเลขใน paper มาจาก
  cell 7 เทียบ confusion matrix ของสอง run ให้เลย (ต้องต่างกัน 0.00) = รูปแสดง prediction ที่อยู่ในตัวเลขของ benchmark จริง ไม่ใช่รันแยก
- ต้องมีบน Drive แล้ว (มีครบจาก lab ก่อน ๆ): LiDAR cache + zone tar ของ val zones, feature bank (`GridNet/protocol/subsets_<RUN2D>/<zone>/subset.npz`),
  `GridNet/fusion/<head>/best.pth` ของ heads ใน `HEADS`
- **Runtime: GPU** — A100 คือเครื่องที่โหมด exact เคยรันจริง (L4 ยังไม่เคยลองกับโหมด exact; ถ้าได้แค่ L4 แล้ว memory ไม่พอ ให้ตั้ง `MODE = 'fast'`
  ซึ่งสำหรับ fusion heads ต่างจาก exact ไม่เกิน 0.04 mIoU)
- เวลาที่วัดได้ (14 พ.ย. 2026, A100): 21 นาทีทั้ง notebook (protocol 7.5 นาที)
- **ก่อนรัน**: `GridNet/code/gridnet_code.zip` บน Drive ต้องเป็นของ **update38 ขึ้นไป** (cell 4 ตรวจให้)
- **ผลลัพธ์**: `GridNet/qualitative/<TAG>/<zone>_crop<i>.npz` (ไฟล์ละไม่กี่ MB) + `preview/*.png` + `crops.txt` — เสร็จแล้วผลอยู่บน Drive (ใช้เลือกเสา/มุมมองและทำรูปของ paper)

**Resume**: Run all ซ้ำได้ — โซนที่เสร็จแล้วถูกข้าม แต่ถ้าเปลี่ยน `HEADS` / `LEVELS` / `RADIUS` / `PYLONS` / `MODE` ให้ตั้ง `TAG` ใหม่ (script หยุดเองถ้าไม่ตรง)"""),
code("""#@title 1. GPU (the A100 is the configuration the exact mode has run on; nothing is trained here)
import subprocess, torch
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
assert torch.cuda.get_device_capability()[0] >= 8, "need an L4 or A100 runtime (the environment tarball is built for these GPUs)"
"""),
code("""#@title 2. Mount Drive + parameters
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/GridNet'
CODE_ZIP = f'{DRIVE}/code/gridnet_code.zip'      #@param {type:"string"}
TAG = 'qual_v1'                                  #@param {type:"string"}
HEADS = 'late_nat_b1,late_a,concat_n,concat_a,win5_n,win5_a'   #@param {type:"string"}
ZONES = 't1z5b,t1z8,t3z4,t4z1,t5b1,t5b5'         #@param {type:"string"}
LEVELS = 'trans:0,trans:0.1,trans:0.5,rot:1'     #@param {type:"string"}
PYLONS = 2                                       #@param {type:"integer"}
RADIUS = 35.0                                    #@param {type:"number"}
MODE = 'exact'                                   #@param ["exact", "fast"]
SEED = 0                                         #@param {type:"integer"}
REFERENCE_RUN = 'exact_v1'                       #@param {type:"string"}
PREVIEW_MODELS = 'concat_n,concat_a'             #@param {type:"string"}
PREVIEW_LEVELS = 'trans:0,trans:0.5,rot:1'       #@param {type:"string"}
RUN2D = 'upernet_convnext_t_v2'                  #@param {type:"string"}
SHUTDOWN_WHEN_DONE = False                       #@param {type:"boolean"}
CACHE_DRIVE = f'{DRIVE}/cache'
CACHE2D_DRIVE = f'{DRIVE}/cache2d/{RUN2D}'
ASSOC_DIR = f'{DRIVE}/fusion/assoc_{RUN2D}'
SUBSET_DIR = f'{DRIVE}/protocol/subsets_{RUN2D}'
OUT = f'{DRIVE}/qualitative/{TAG}'
os.makedirs(OUT, exist_ok=True)
zones = [z.strip() for z in ZONES.split(',') if z.strip()]
heads = {}
for name in [h.strip() for h in HEADS.split(',') if h.strip()]:
    path = f'{DRIVE}/fusion/{name}/best.pth'
    assert os.path.exists(path), f"no checkpoint for head {name}: {path}"
    heads[name] = path
sweeps = {}                                      # the levels of the run: the wanted ones, and the delivered poses (0) in every sweep
for item in [x.strip() for x in LEVELS.split(',') if x.strip()]:
    sw, lv = item.split(':'); assert sw in ('trans', 'rot', 'drift'), f"unknown sweep in LEVELS: {item}"
    sweeps.setdefault(sw, [0.0])
    if float(lv) not in sweeps[sw]: sweeps[sw].append(float(lv))
for sw in sweeps: sweeps[sw].sort()
for z in zones:
    assert os.path.exists(f'{CACHE_DRIVE}/{z}/points.npz'), f"no LiDAR cache on Drive for {z}"
    assert os.path.exists(f'{SUBSET_DIR}/{z}/subset.npz'), f"no feature bank on Drive for {z}: {SUBSET_DIR}/{z}/subset.npz (colab_heads_lab writes them)"
done = []
if os.path.exists(f'{OUT}/protocol/results.json'):
    import json
    done = sorted({r['zone'] for r in json.load(open(f'{OUT}/protocol/results.json'))['runs']})
print("heads:", list(heads)); print("levels of the run:", sweeps, "| kept for the figure:", LEVELS)
print(f"{len(zones)} zones, {PYLONS} pylons each, radius {RADIUS:g} m, {MODE} mode  ->  {OUT}")
print(f"{len(done)}/{len(zones)} zones already on Drive:", done)
"""),
protocol_cells[3],
code("""#@title 4. Code + LitePT
import subprocess, zipfile, os
os.makedirs('/content/code', exist_ok=True); zipfile.ZipFile(CODE_ZIP).extractall('/content/code')
assert os.path.exists('/content/code/gridnet/qualitative.py'), "gridnet_code.zip on Drive is older than update38: pack and upload it again"
if not os.path.exists('/content/LitePT'):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/prs-eth/LitePT.git", "/content/LitePT"], check=True)
r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0,'/content/LitePT'); sys.path.insert(0,'/content/code'); "
                    "from litept.model import LitePT; import gridnet.train.protocol, gridnet.qualitative; print('imports ok')"], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-1500:])
assert 'imports ok' in r.stdout
"""),
code("""#@title 5. Data -> local disk: LiDAR caches, feature banks, 2D caches (zone tars), camera files of ZONES
import os, sys, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/content/code')
from gridnet.train.maps2d import cached_image_ids, zone_tar_manifest, extract_zone_tar, write_compact
DATA = '/content/data'; os.makedirs(DATA, exist_ok=True)
t0 = time.time()
for z in zones:
    for src, dst in ((f'{CACHE_DRIVE}/{z}/points.npz', f'/content/cache/{z}/points.npz'), (f'{SUBSET_DIR}/{z}/subset.npz', f'/content/banks/{z}/subset.npz')):
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(src, dst)
print(f"LiDAR caches + feature banks of {len(zones)} zones ({(time.time() - t0) / 60:.1f} min)")
CACHE2D = '/content/cache2d'; t0 = time.time(); slow = []
for z in zones:
    if zone_tar_manifest(CACHE2D_DRIVE, z, wait=30) is not None:   # one big sequential read: fast
        n = extract_zone_tar(CACHE2D_DRIVE, z, CACHE2D); how = 'tar'
    else:                                                             # no tar yet: parallel per-file copy (slow)
        ids = cached_image_ids(CACHE2D_DRIVE, z)
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(lambda i: write_compact(CACHE2D_DRIVE, z, i, CACHE2D), ids))
        n = len(cached_image_ids(CACHE2D, z)); how = 'files'; slow.append(z)
    print(f"  {z}: {n} images from {how}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
if slow: print("zones without a tar (slow path) - run colab_compact_cache2d once for:", ','.join(slow))
code = ("import sys; sys.path.insert(0,'/content/code'); from gridnet.train.data2d import fetch_zones; "
        f"fetch_zones({zones!r}, '{DATA}', what=('pose',))")
subprocess.run([PY, '-c', code], check=True); print("camera files ready")
print("associations present:", [z for z in zones if os.path.exists(f'{ASSOC_DIR}/{z}/assoc.npz')], "(missing ones are computed on first use)")
print(subprocess.run(['df', '-h', '--output=avail', '/content'], capture_output=True, text=True).stdout.split()[-1], "free on the local disk")
"""),
code("""#@title 6. Protocol, draw 0 of LEVELS only, with the predictions around the pylons saved -> GridNet/qualitative/<TAG>/ (resumes per zone)
import subprocess, time, os
cmd = [PY, '/content/code/scripts/run_protocol.py', '--cache', '/content/cache', '--cache2d', CACHE2D, '--data', DATA,
       '--assoc-dir', ASSOC_DIR, '--out', f'{OUT}/protocol', '--subset-dir', '/content/banks', '--litept', '/content/LitePT',
       '--zones', ','.join(zones), '--sweeps', ','.join(sweeps), '--draws', '1', '--mode', MODE, '--seed', str(SEED),
       '--dump-pred', OUT, '--dump-levels', LEVELS, '--dump-radius', str(RADIUS), '--dump-pylons', str(PYLONS)]
for sw, lv in sweeps.items(): cmd += [f'--levels-{sw}', ','.join(f'{x:g}' for x in lv)]
cmd += ['--heads'] + [f'{n}={p}' for n, p in heads.items()]
env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', MPLBACKEND='Agg')
t0 = time.time(); tail = []
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
try:
    for line in p.stdout:
        tail = (tail + [line])[-40:]
        if line.startswith('[protocol]'): print(f"   {(time.time() - t0) / 60:5.1f} min  {line}", end='', flush=True)
        elif 'Error' in line or 'Traceback' in line or line.startswith('  File'): print(line, end='')
except KeyboardInterrupt:
    p.terminate(); p.wait(); print("stopped by user; finished zones are on Drive -> run this cell again to continue"); raise
p.wait(); print(f"exit {p.returncode} after {(time.time() - t0) / 60:.1f} min")
if p.returncode != 0:
    print(''.join(l for l in tail if not l.startswith('[protocol]'))[-3000:])
    print("-> if it STOPPED on 'other settings': this TAG holds a run with other heads / levels / radius / pylons / mode - give TAG a new name in cell 2, re-run cell 2 and this cell")
    print("-> if it ran out of GPU memory on a GPU smaller than the A100: set MODE = 'fast' and a new TAG in cell 2")
crops = sorted(f for f in os.listdir(OUT) if f.endswith('.npz') and '_crop' in f and '.tmp' not in f)
print(f"{len(crops)} crop files on Drive:", ', '.join(f"{f} ({os.path.getsize(f'{OUT}/{f}') / 2**20:.1f} MB)" for f in crops))
assert p.returncode == 0 and crops, "the run did not finish: see the lines above"
"""),
code("""#@title 7. Check against the benchmark run (same cameras -> same confusion matrices), table of the crops, previews
import subprocess, json, os
from IPython.display import Image, display
env = dict(os.environ, PYTHONPATH='/content/code', MPLBACKEND='Agg')
ref = f'{DRIVE}/protocol/{REFERENCE_RUN}'
if REFERENCE_RUN and (os.path.exists(f'{ref}/results.json') or os.path.exists(f'{ref}/results.json.gz')):
    r = subprocess.run([PY, '/content/code/scripts/compare_protocol.py', '--a', f'{OUT}/protocol', '--b', ref, '--names', f'figure,{REFERENCE_RUN}',
                        '--out', f'{OUT}/compare_{REFERENCE_RUN}.json'], capture_output=True, text=True, env=env)
    print(r.stdout[-9000:], r.stderr[-1500:])
    if r.returncode == 0:
        rows = json.load(open(f'{OUT}/compare_{REFERENCE_RUN}.json'))['rows']
        worst = max(max(abs(x['mean_all_b'] - x['mean_all_a']), abs(x['mean_electrical_b'] - x['mean_electrical_a'])) for x in rows)
        verdict = ("(fast mode: small differences to the exact run are expected)" if MODE != 'exact' else
                   "-> the crops show predictions of the benchmark" if 100 * worst < 0.05 else "-> NOT the same predictions: find the cause before the figure is used")
        print(f"largest difference to {REFERENCE_RUN} over {len(rows)} (model, level) cells: {100 * worst:.3f} points of mean IoU {verdict}")
else:
    print(f"no reference run at {ref}: comparison skipped")
r = subprocess.run([PY, '/content/code/scripts/plot_qualitative.py', '--list', OUT], capture_output=True, text=True, env=env)
print(r.stdout[-14000:], r.stderr[-1500:])
open(f'{OUT}/crops.txt', 'w').write(r.stdout)
os.makedirs(f'{OUT}/preview', exist_ok=True)
for f in sorted(x for x in os.listdir(OUT) if x.endswith('.npz') and '_crop' in x and '.tmp' not in x):
    base = f"{OUT}/preview/{f[:-4]}"
    r = subprocess.run([PY, '/content/code/scripts/plot_qualitative.py', '--crop', f'{OUT}/{f}', '--models', PREVIEW_MODELS, '--levels', PREVIEW_LEVELS,
                        '--out', base], capture_output=True, text=True, env=env)
    print(r.stdout.strip() or r.stderr[-1500:])
    if os.path.exists(base + '.png'): display(Image(base + '.png', width=1000))
"""),
code("""#@title 8. (optional) Shut the runtime down once everything above has finished - saves compute units on unattended runs
if SHUTDOWN_WHEN_DONE:
    from google.colab import drive, runtime
    print("flushing pending writes to Drive ..."); drive.flush_and_unmount()
    print("results are on Drive; shutting the runtime down"); runtime.unassign()
else:
    print("SHUTDOWN_WHEN_DONE is off -> remember to Disconnect and delete runtime")
"""),
md("""## หมายเหตุ
- ไฟล์ crop หนึ่งไฟล์ = เสาไฟฟ้าหนึ่งต้น: จุดของ subset (จุดของ 4 electrical classes ครบตาม cap ของ subset + ตัวอย่างของ class อื่น) ภายในรัศมี `RADIUS` ม.
  ถูกลดเหลือหนึ่งจุดต่อ voxel 5 ซม. สำหรับ **วาดรูป** ส่วน **สถิติ** ของ crop (confusion matrix ในไฟล์) คิดจากจุดของ subset ทุกจุดในรัศมีพร้อม weight
  จึงเป็นตัวเลขของ benchmark ที่จำกัดเฉพาะบริเวณนั้น — พื้นและต้นไม้รอบ ๆ เป็นแค่ตัวอย่างเบาบาง (ไม่ใช่ cloud เต็ม) รูปนี้จึงเป็นรูปของสายและเสา
- `LEVELS` = ระดับที่เก็บ: pose ที่ให้มา, 10 cm (reference level: fusion heads ไม่ควรเปลี่ยน แต่ ImageVote เริ่มเสียสาย), 50 cm และ 1° (ระดับใหญ่สุดของตาราง)
- รูป preview ของ cell 7: แถว = `PREVIEW_MODELS`, คอลัมน์ = `PREVIEW_LEVELS`; จุดของสาย/เสาที่ทำนาย class ถูก = เทา, ผิด = แดง, จุดรอบ ๆ ที่ถูกทำนายเป็นสาย/เสา = น้ำเงิน;
  ตัวเลขในแต่ละช่อง = สัดส่วนจุดของ electrical classes ใน crop ที่ทำนายผิด (ถ่วง weight) — จุดผิดถูกวาดทับและขยาย พื้นที่สีแดงในรูปจึงดูมากกว่าตัวเลข
- เสาถูกเลือกจากจำนวนจุดของ class pylon (ต้นที่ใหญ่ที่สุดของแต่ละโซน) ก่อนเห็น prediction ใด ๆ; การเลือกว่าจะใช้ต้นไหนใน paper ทำทีหลังจาก `crops.txt`
  และ caption ต้องบอกว่าเลือกอย่างไร
- `drift` ใน `LEVELS` ได้เช่นกัน (เช่น `drift:0.5`): draw 0 = ทิศ +
- จบแล้ว **Disconnect and delete runtime**"""),
]

# ----------------------------------------------------------------------------- shared pose errors notebook
shared_cells = [
md("""# GridNet-HD — Robustness protocol กับ pose error แบบ "ร่วม" (shared): ทุกภาพของโซนผิดเหมือนกัน (GPU)

**ทำไมต้องมี**: sweep หลักของ paper ให้ทุกกล้องมี error ของตัวเองแบบสุ่มอิสระ (i.i.d.) ซึ่งเป็นกรณีที่ง่ายสำหรับโมเดลที่เฉลี่ยหลาย view
error จริงมักเอียงไปทางเดียวกันทุกภาพ notebook นี้รัน protocol เดิมทุกอย่าง (โหมด exact, 10 draws, ระดับ error ชุดเดิม) แต่เปลี่ยนอย่างเดียวคือ
**ทุกกล้องของโซนใช้ error ตัวเดียวกัน** (สุ่มหนึ่งครั้งต่อโซนต่อ draw จาก distribution เดียวกับ sweep i.i.d.):

| sweep | ความหมาย | ระดับ |
|---|---|---|
| `trans_shared` | เลื่อนกล้องทุกตัวด้วยเวกเตอร์เดียวกัน = ชุดภาพทั้งชุดเหลื่อมกับ point cloud (georeference สองชุดข้อมูลแยกกัน) | σ_t 2, 5, 10, 25, 50 cm |
| `rot_shared` | หมุนกล้องทุกตัวด้วยมุมเดียวกันรอบแกนของกล้อง = boresight error | σ_r 0.05, 0.1, 0.25, 0.5, 1° |

ไม่มีการเทรนอะไร: ใช้ head 24 ตัวของตาราง (4 แบบ × เทรนด้วย pose ที่ให้มา / pose-noise augmentation × 3 seeds) + LiDAR-only + ImageVote
ผลจะถูกวางเทียบกับ run `exact_v1` (error แบบ i.i.d. ขนาดเดียวกัน) ใน cell 7

- ต้องมีบน Drive แล้ว (มีครบจาก lab ก่อน ๆ): LiDAR cache + zone tar ของ val zones, feature bank (`GridNet/protocol/subsets_<RUN2D>/<zone>/subset.npz`),
  `GridNet/fusion/<head>/best.pth` ของ heads ใน `HEADS`, และ `GridNet/protocol/exact_v1/summary.json`
- **Runtime: GPU A100** (เครื่องที่โหมด exact เคยรันจริง)
- **ก่อนรัน**: `GridNet/code/gridnet_code.zip` บน Drive ต้องเป็นของ **update41 ขึ้นไป** (cell 4 ตรวจให้)
- เวลาที่วัดได้ (run `shared_v1`, 14 พ.ย. 2026, A100-SXM4-40GB): cell 6 ใช้ **139.5 นาที** = 16–28 นาทีต่อโซน, 13.7 วินาทีต่อ evaluation
  (612 evaluations × 26 โมเดล); เตรียมข้อมูล 6.5 นาที; ทั้ง session ราว 2.5 ชม. (ที่ประมาณไว้ก่อนรันคือ 1.9 ชม. สำหรับ cell 6)
  cell 2 พิมพ์ค่าประมาณตามค่าที่ตั้ง โดย interpolate ระหว่าง run ที่วัดแล้ว (6 โมเดล ~6 วินาที, 26 โมเดล 13.7, 37 โมเดล 14.4 วินาทีต่อ evaluation)
- **ผลลัพธ์**: `GridNet/protocol/<TAG>/` (results.json(.gz), summary.json, `compare_exact_v1.json/.txt`, รูป) — เสร็จแล้วผลอยู่บน Drive

**Resume**: Run all ซ้ำได้ — โซนที่เสร็จแล้วถูกข้าม แต่ถ้าเปลี่ยน `HEADS` / `SWEEPS` / ระดับ / `DRAWS` / `MODE` ให้ตั้ง `TAG` ใหม่ (script หยุดเองถ้าไม่ตรง)"""),
code("""#@title 1. GPU (the A100 is the configuration the exact mode has run on; nothing is trained here)
import subprocess, torch
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
assert torch.cuda.get_device_capability()[0] >= 8, "need an L4 or A100 runtime (the environment tarball is built for these GPUs)"
"""),
code("""#@title 2. Mount Drive + parameters
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/GridNet'
CODE_ZIP = f'{DRIVE}/code/gridnet_code.zip'      #@param {type:"string"}
TAG = 'shared_v1'                                #@param {type:"string"}
# the 24 heads of the fix table: 4 designs x (delivered poses | pose-noise augmentation) x 3 seeds
HEADS = 'late_nat_b1,late_n_s1,late_n_s2,concat_n,concat_n_s1,concat_n_s2,win3_n,win3_n_s1,win3_n_s2,win5_n,win5_n_s1,win5_n_s2,late_a,late_a_s1,late_a_s2,concat_a,concat_a_s1,concat_a_s2,win3_a,win3_a_s1,win3_a_s2,win5_a,win5_a_s1,win5_a_s2'   #@param {type:"string"}
ZONES = 't1z5b,t1z8,t3z4,t4z1,t5b1,t5b5'         #@param {type:"string"}
SWEEPS = 'trans_shared,rot_shared'               #@param {type:"string"}
LEVELS_T = '0,0.02,0.05,0.1,0.25,0.5'            #@param {type:"string"}
LEVELS_R = '0,0.05,0.1,0.25,0.5,1'               #@param {type:"string"}
DRAWS = 10                                       #@param {type:"integer"}
MODE = 'exact'                                   #@param ["exact", "fast"]
SEED = 0                                         #@param {type:"integer"}
REFERENCE_RUN = 'exact_v1'                       #@param {type:"string"}
RUN2D = 'upernet_convnext_t_v2'                  #@param {type:"string"}
SHUTDOWN_WHEN_DONE = False                       #@param {type:"boolean"}
CACHE_DRIVE = f'{DRIVE}/cache'
CACHE2D_DRIVE = f'{DRIVE}/cache2d/{RUN2D}'
ASSOC_DIR = f'{DRIVE}/fusion/assoc_{RUN2D}'
SUBSET_DIR = f'{DRIVE}/protocol/subsets_{RUN2D}'
OUT = f'{DRIVE}/protocol/{TAG}'
os.makedirs(OUT, exist_ok=True)
zones = [z.strip() for z in ZONES.split(',') if z.strip()]
heads = {}
for name in [h.strip() for h in HEADS.split(',') if h.strip()]:
    path = f'{DRIVE}/fusion/{name}/best.pth'
    assert os.path.exists(path), f"no checkpoint for head {name}: {path}"
    heads[name] = path
sweeps = [x.strip() for x in SWEEPS.split(',') if x.strip()]
assert sweeps and all(sw in ('trans_shared', 'rot_shared') for sw in sweeps), f"SWEEPS of this notebook: trans_shared, rot_shared (got {SWEEPS})"
levels = {'trans_shared': [float(x) for x in LEVELS_T.split(',')], 'rot_shared': [float(x) for x in LEVELS_R.split(',')]}
assert all(levels[sw][0] == 0.0 for sw in sweeps), "every level list starts with 0 (the delivered poses)"
for z in zones:
    assert os.path.exists(f'{CACHE_DRIVE}/{z}/points.npz'), f"no LiDAR cache on Drive for {z}"
    assert os.path.exists(f'{SUBSET_DIR}/{z}/subset.npz'), f"no feature bank on Drive for {z}: {SUBSET_DIR}/{z}/subset.npz (colab_heads_lab writes them)"
if not os.path.exists(f'{DRIVE}/protocol/{REFERENCE_RUN}/summary.json'):
    print(f"WARNING: no {DRIVE}/protocol/{REFERENCE_RUN}/summary.json - cell 7 cannot compare with the i.i.d. run")
done = []
if os.path.exists(f'{OUT}/results.json'):
    import json
    done = sorted({r['zone'] for r in json.load(open(f'{OUT}/results.json'))['runs']})
EVALS_PER_ZONE = sum(1 + (len(levels[sw]) - 1) * DRAWS for sw in sweeps)
n_models = len(heads) + 2
import numpy as np
# seconds per evaluation, measured on an A100: exact mode 6 models ~6 s (two-zone check), 26 models 13.7 s (shared_v1), 37 models
# 14.4 s (exact_v1) - interpolated, flat beyond the measured range; fast mode from fast_v1 and the heads labs
sec = float(np.interp(n_models, [6, 26, 37], [6.0, 13.7, 14.4])) if MODE == 'exact' else (0.4 + 0.1 * n_models)
todo = [z for z in zones if z not in done]
print(f"{n_models} models: LiDAR-only, ImageVote and {len(heads)} heads")
print("levels:", {sw: levels[sw] for sw in sweeps}, f"| {DRAWS} draws, {MODE} mode, seed {SEED}  ->  {OUT}")
print(f"{len(done)}/{len(zones)} zones already on Drive: {done}")
print(f"to do: {len(todo)} zones x {EVALS_PER_ZONE} evaluations = {len(todo) * EVALS_PER_ZONE}; "
      f"estimate {len(todo) * EVALS_PER_ZONE * sec / 3600:.1f} h at about {sec:.1f} s per evaluation (interpolated between measured runs)")
"""),
protocol_cells[3],
code("""#@title 4. Code + LitePT
import subprocess, zipfile, os
os.makedirs('/content/code', exist_ok=True); zipfile.ZipFile(CODE_ZIP).extractall('/content/code')
assert 'trans_shared' in open('/content/code/gridnet/train/protocol.py').read() and os.path.exists('/content/code/scripts/compare_shared.py'), \\
    "gridnet_code.zip on Drive is older than update41: pack and upload it again"
if not os.path.exists('/content/LitePT'):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/prs-eth/LitePT.git", "/content/LitePT"], check=True)
r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0,'/content/LitePT'); sys.path.insert(0,'/content/code'); "
                    "from litept.model import LitePT; import gridnet.train.protocol as P; print('imports ok', sorted(P.SHARED_SWEEPS))"], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-1500:])
assert 'imports ok' in r.stdout
"""),
qual_cells[5],
code("""#@title 6. Protocol with shared pose errors -> GridNet/protocol/<TAG>/ (resumes per zone)
import subprocess, time, os
cmd = [PY, '/content/code/scripts/run_protocol.py', '--cache', '/content/cache', '--cache2d', CACHE2D, '--data', DATA,
       '--assoc-dir', ASSOC_DIR, '--out', OUT, '--subset-dir', '/content/banks', '--litept', '/content/LitePT',
       '--zones', ','.join(zones), '--sweeps', ','.join(sweeps), '--draws', str(DRAWS), '--mode', MODE, '--seed', str(SEED),
       '--levels-trans-shared', ','.join(f'{x:g}' for x in levels['trans_shared']), '--levels-rot-shared', ','.join(f'{x:g}' for x in levels['rot_shared'])]
cmd += ['--heads'] + [f'{n}={p}' for n, p in heads.items()]
env = dict(os.environ, PYTHONPATH='/content/code', PYTHONWARNINGS='ignore', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', MPLBACKEND='Agg')
n_total = max(1, len(todo) * EVALS_PER_ZONE)
t0 = time.time(); tail = []; k = 0
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
try:
    for line in p.stdout:
        tail = (tail + [line])[-40:]
        if line.startswith('[protocol]') and ' draw ' in line:          # one line per evaluation: every 25th is shown
            k += 1
            if k % 25 == 0 or k == 1:
                el = (time.time() - t0) / 60
                print(f"   {el:5.1f} min  {k}/{n_total} evaluations, about {el / k * (n_total - k):.0f} min left  |  {line[11:150].strip()} ...", flush=True)
        elif line.startswith('[protocol]'): print(f"   {(time.time() - t0) / 60:5.1f} min  {line}", end='', flush=True)
        elif 'Error' in line or 'Traceback' in line or line.startswith('  File'): print(line, end='')
except KeyboardInterrupt:
    p.terminate(); p.wait(); print("stopped by user; finished zones are on Drive -> run this cell again to continue"); raise
p.wait(); print(f"exit {p.returncode} after {(time.time() - t0) / 60:.1f} min")
if p.returncode != 0:
    print(''.join(l for l in tail if not l.startswith('[protocol]'))[-3000:])
    print("-> if it STOPPED on 'other settings': this TAG holds a run with other heads / sweeps / levels / draws / mode - give TAG a new name in cell 2, re-run cell 2 and this cell")
assert p.returncode == 0 and os.path.exists(f'{OUT}/summary.json'), "the run did not finish: see the lines above"
"""),
code("""#@title 7. Shared against independent errors (run REFERENCE_RUN), table + figure
import subprocess, os
from IPython.display import Image, display
env = dict(os.environ, PYTHONPATH='/content/code', MPLBACKEND='Agg')
ref = f'{DRIVE}/protocol/{REFERENCE_RUN}'
if REFERENCE_RUN and os.path.exists(f'{ref}/summary.json'):
    r = subprocess.run([PY, '/content/code/scripts/compare_shared.py', '--shared', OUT, '--iid', ref, '--names', f'{TAG},{REFERENCE_RUN}',
                        '--out', f'{OUT}/compare_{REFERENCE_RUN}.json'], capture_output=True, text=True, env=env)
    print(r.stdout[-30000:], r.stderr[-1500:])
    if r.returncode == 0: open(f'{OUT}/compare_{REFERENCE_RUN}.txt', 'w').write(r.stdout)
else:
    print(f"no reference run at {ref}: comparison skipped")
one_seed = [m for m in ('lidar', 'imagevote', 'late_nat_b1', 'concat_n', 'win3_n', 'win5_n', 'late_a', 'concat_a', 'win3_a', 'win5_a') if m in heads or m in ('lidar', 'imagevote')]
fig = f'{OUT}/protocol_{TAG}.png'
r = subprocess.run([PY, '/content/code/scripts/plot_protocol.py', '--summary', f'{OUT}/summary.json', '--out', fig, '--tables', 'compact',
                    '--models', ','.join(one_seed), '--title', f'{TAG}: shared pose errors, {MODE} mode (one seed per head type)'], capture_output=True, text=True, env=env)
print(r.stdout[-6000:], r.stderr[-1500:])
if os.path.exists(fig): display(Image(fig, width=1000))
print("files on Drive:", sorted(os.listdir(OUT)))
"""),
code("""#@title 8. (optional) Shut the runtime down once everything above has finished - saves compute units on unattended runs
if SHUTDOWN_WHEN_DONE:
    from google.colab import drive, runtime
    print("flushing pending writes to Drive ..."); drive.flush_and_unmount()
    print("results are on Drive; shutting the runtime down"); runtime.unassign()
else:
    print("SHUTDOWN_WHEN_DONE is off -> remember to Disconnect and delete runtime")
"""),
md("""## หมายเหตุ
- error ร่วมหนึ่งตัวต่อ (โซน, draw): seed ขึ้นกับชื่อโซนด้วย แต่ละโซนของ draw เดียวกันจึงได้ error คนละตัว (6 โซน × 10 draws = 60 ตัวต่อระดับ)
  ค่าเฉลี่ยของ draw จึงแกว่งกว่า sweep i.i.d. มาก (ที่นั่นหนึ่ง draw เฉลี่ยกล้องเป็นร้อยตัวอยู่แล้ว) — ตารางของ cell 7 จึงพิมพ์ ± ของ draws ไว้ด้วย
- ระดับ 0 (pose ที่ให้มา) ต้องได้ตัวเลขเท่ากับ run `exact_v1` ทุกโมเดล (head เดิม จุดเดิม) — บรรทัด check ของ cell 7 ต้องเป็น 0.000
- ค่า i.i.d. ที่ใช้เทียบมาจาก run `exact_v1` (ไม่รันซ้ำที่นี่): โมเดลเดียวกัน, subset เดียวกัน, โหมด exact เหมือนกัน
- `trans_shared` = เลื่อนกล้องทุกตัวเท่ากันในพิกัดโลก (เทียบเท่ากับเลื่อน point cloud ทั้งก้อนไปทางตรงข้าม); `rot_shared` = หมุนในพิกัดของกล้องแต่ละตัว
  ด้วยมุมเดียวกัน (ทุกภาพเลื่อนไปทางเดียวกันในภาพ)
- จบแล้ว **Disconnect and delete runtime**"""),
]

for c in train_cells + upernet_cells + fusion_cells + fusion_train_cells:
    if c["cell_type"] == "code":
        c["source"] = c["source"].replace("__COLAB_ENV_SH__", ENV_SH)

for name, cells, gpu in [("colab_build_cache.ipynb", cache_cells, None), ("colab_train_litept.ipynb", train_cells, "A100"),
                         ("colab_train_upernet.ipynb", upernet_cells, "L4"), ("colab_fusion.ipynb", fusion_cells, "L4"), ("colab_train_fusion.ipynb", fusion_train_cells, "A100"),
                         ("colab_compact_cache2d.ipynb", compact_cells, None), ("colab_protocol.ipynb", protocol_cells, "A100"),
                         ("colab_heads_lab.ipynb", lab_cells, "A100"), ("colab_leaderboard.ipynb", leaderboard_cells, "A100"),
                         ("colab_label_transfer.ipynb", label_cells, None), ("colab_qualitative.ipynb", qual_cells, "A100"),
                         ("colab_shared_error.ipynb", shared_cells, "A100")]:
    with open(os.path.join(HERE, name), "w") as fh:
        json.dump(nb(cells, gpu), fh, indent=1, ensure_ascii=False)
    print("wrote", name)
