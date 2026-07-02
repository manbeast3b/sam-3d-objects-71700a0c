#!/usr/bin/env bash
# Minimal reproduction of SAM 3D Objects' core claim: single image -> 3D.
# Builds the conda env, downloads (ungated mirror) HF checkpoints, runs demo.py
# on the sample kidsroom image+mask, and writes EVAL.md to .openresearch/artifacts/.
#
# Designed to run unattended on a single GPU instance (>=32GB VRAM). Every stage
# logs to both stdout and EVAL.md; a failure records a clear FAILED marker so the
# post-mortem is readable from `orx artifact ... EVAL.md`.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export HF_HUB_ENABLE_HF_TRANSFER=0
export DEBIAN_FRONTEND=noninteractive
# Build only for the GPU archs we actually launch on (H100=9.0, A100=8.0, L40/4090=8.9).
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.9;9.0}"
# Cap parallel nvcc jobs so flash-attn / pytorch3d compiles don't OOM the box.
export MAX_JOBS="${MAX_JOBS:-4}"

mkdir -p .openresearch/artifacts
EVAL=".openresearch/artifacts/EVAL.md"
: > "$EVAL"

log() { echo "[run] $*" | tee -a "$EVAL"; }
section() { echo | tee -a "$EVAL"; echo "### $*" | tee -a "$EVAL"; }

fail() {
  section "FAILED"
  log "$*"
  {
    echo
    echo "## Result"
    echo
    echo "- **Status:** FAILED — $*"
    echo "- See the run log for the full traceback."
  } >> "$EVAL"
  exit 1
}

{
  echo "# SAM 3D Objects — Minimal Reproduction"
  echo
  echo "Core claim: reconstruct a full 3D model (gaussian splat) from a single image + mask."
} >> "$EVAL"

# ---------------------------------------------------------------------------
section "Stage 1: install conda (Miniforge: conda + mamba)"
# The env spec (environments/default.yml) is a full conda spec that pins the
# CUDA 12.1 toolkit + gcc 12.4 needed to build pytorch3d/flash-attn/kaolin, so
# we use a real conda base (Miniforge) rather than micromamba to get reliable
# `conda activate` semantics.
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
  log "Installing Miniforge to $CONDA_ROOT"
  curl -fsSL -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    || fail "could not download Miniforge installer"
  bash /tmp/miniforge.sh -b -p "$CONDA_ROOT" || fail "Miniforge install failed"
fi
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
if [ -f "$CONDA_ROOT/etc/profile.d/mamba.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/mamba.sh"
fi
conda --version | tee -a "$EVAL"

# ---------------------------------------------------------------------------
section "Stage 2: create conda env 'sam3d-objects'"
if conda env list | grep -qE "^\s*sam3d-objects\s"; then
  log "env 'sam3d-objects' already exists — reusing"
else
  if command -v mamba >/dev/null 2>&1; then
    mamba env create -y -f environments/default.yml 2>&1 | tee -a "$EVAL"
  else
    conda env create -y -f environments/default.yml 2>&1 | tee -a "$EVAL"
  fi
  # PIPESTATUS[0] is the create command's exit status (tee is [1]).
  [ "${PIPESTATUS[0]:-0}" -eq 0 ] || fail "conda env create failed"
fi
conda activate sam3d-objects || fail "could not activate sam3d-objects env"
log "python: $(python --version 2>&1) @ $(which python)"

# ---------------------------------------------------------------------------
section "Stage 3: pip install sam3d-objects (dev -> p3d -> inference)"
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# Pin the torch stack FIRST so pytorch3d/flash-attn/kaolin build against a known
# ABI (matches torchaudio 2.5.1+cu121 / kaolin find-links / xformers 0.0.28.post3).
log "installing torch 2.5.1+cu121 stack"
pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -8 | tee -a "$EVAL"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" \
  2>&1 | tee -a "$EVAL" || fail "torch import failed after install"

# Build tooling needed for editable/no-build-isolation installs. pytorch3d and
# flash-attn `import torch` at BUILD time, so they must be built with isolation
# OFF (so the ambient torch is visible) — which in turn requires every build dep
# to be present in the env up front. Install them here.
log "installing build tooling"
pip install --no-cache-dir \
  "setuptools>=68" wheel ninja packaging cmake \
  hatchling hatch-requirements-txt 2>&1 | tail -5 | tee -a "$EVAL"

for extra in dev p3d inference; do
  log "pip install -e '.[$extra]' (no build isolation)"
  pip install --no-build-isolation -e ".[$extra]" 2>&1 | tail -20 | tee -a "$EVAL"
  [ "${PIPESTATUS[0]:-0}" -eq 0 ] || fail "pip install .[$extra] failed"
done

# Verify the extension modules that most often fail to build actually import.
log "verifying key extension imports"
python - <<'PY' 2>&1 | tee -a "$EVAL"
import importlib
ok = True
for m in ["torch", "torchvision", "pytorch3d", "kaolin", "gsplat"]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        ok = False
        print(f"  FAIL {m}: {type(e).__name__}: {e}")
# flash_attn is optional at import-time for this demo path; report but don't fail.
try:
    import flash_attn
    print(f"  OK   flash_attn {flash_attn.__version__}")
except Exception as e:
    print(f"  warn flash_attn import: {e}")
import sys
sys.exit(0 if ok else 3)
PY
[ "${PIPESTATUS[0]:-0}" -eq 0 ] || fail "a required extension module failed to import (see above)"

# ---------------------------------------------------------------------------
section "Stage 4: patch hydra"
./patching/hydra 2>&1 | tail -5 | tee -a "$EVAL" \
  || log "hydra patch returned non-zero (continuing; may already be patched)"

# ---------------------------------------------------------------------------
section "Stage 5: download checkpoints (public ungated mirror)"
pip install --no-cache-dir 'huggingface-hub[cli]<1.0' 2>&1 | tail -3 | tee -a "$EVAL"
# Public, ungated mirror of facebook/sam-3d-objects checkpoints (no token needed).
HF_REPO="${HF_REPO:-jetjodh/sam-3d-objects}"
TAG=hf
if [ ! -f "checkpoints/${TAG}/pipeline.yaml" ]; then
  rm -rf "checkpoints/${TAG}-download"
  hf download --repo-type model --local-dir "checkpoints/${TAG}-download" \
    --max-workers 2 "$HF_REPO" 2>&1 | tail -12 | tee -a "$EVAL" \
    || fail "checkpoint download from $HF_REPO failed"
  if [ -d "checkpoints/${TAG}-download/checkpoints" ]; then
    mv "checkpoints/${TAG}-download/checkpoints" "checkpoints/${TAG}"
  else
    # Some mirrors put files at the repo root instead of under checkpoints/.
    mv "checkpoints/${TAG}-download" "checkpoints/${TAG}"
  fi
  rm -rf "checkpoints/${TAG}-download"
fi
[ -f "checkpoints/${TAG}/pipeline.yaml" ] || fail "pipeline.yaml missing after download"
[ -s "checkpoints/${TAG}/ss_generator.ckpt" ] || fail "ss_generator.ckpt missing/empty"
[ -s "checkpoints/${TAG}/slat_generator.ckpt" ] || fail "slat_generator.ckpt missing/empty"
log "checkpoint files:"
ls -la "checkpoints/${TAG}" | tee -a "$EVAL"

# ---------------------------------------------------------------------------
section "Stage 6: run demo.py (single image -> 3D gaussian splat)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | tee -a "$EVAL" || true
python demo.py 2>&1 | tee -a "$EVAL"
demo_rc="${PIPESTATUS[0]:-1}"

# ---------------------------------------------------------------------------
section "Stage 7: results"
{
  echo
  echo "## Result"
  echo
  if [ "$demo_rc" -eq 0 ] && [ -s splat.ply ]; then
    SIZE=$(stat -c %s splat.ply)
    N_LINES=$(grep -c . splat.ply 2>/dev/null || echo "?")
    echo "- **Status:** SUCCESS"
    echo "- **Output:** \`splat.ply\` (${SIZE} bytes)"
    echo "- Reconstructed a 3D gaussian splat from a single RGB image (kidsroom, object index 14) with the pretrained SAM 3D Objects pipeline — full geometry + texture from one view."
    # Save a copy of the splat as an artifact for later inspection.
    cp splat.ply .openresearch/artifacts/splat.ply 2>/dev/null || true
  else
    echo "- **Status:** FAILED (demo exit=$demo_rc, splat.ply present=$( [ -s splat.ply ] && echo yes || echo no ))"
  fi
  echo
  echo "## Setup"
  echo
  echo "- **Entrypoint:** \`python demo.py\` → loads \`checkpoints/hf/pipeline.yaml\`, runs the full inference pipeline (MoGe pointmap → sparse-structure sample → SLAT sample → gaussian/mesh decode) on one image+mask, saves \`splat.ply\`."
  echo "- **Env:** Miniforge conda env from \`environments/default.yml\` (CUDA 12.1 toolkit, gcc 12.4) + torch 2.5.1+cu121 + \`pip install -e '.[dev,p3d,inference]'\` (pytorch3d, flash-attn, kaolin, gsplat) + hydra 1.3.2 patch."
  echo "- **Checkpoints:** \`$HF_REPO\` (public ungated mirror of \`facebook/sam-3d-objects\`)."
  echo "- **Compute:** single GPU, >=32GB VRAM (launched on H100_SXM)."
  echo
  echo "This is the smallest configuration that demonstrates the paper's central mechanism end to end: single-image 3D reconstruction of an arbitrary masked object into full 3D geometry and texture."
} >> "$EVAL"

if [ "$demo_rc" -ne 0 ] || [ ! -s splat.ply ]; then
  log "demo failed or produced no splat"
  exit 1
fi
log "done — splat.ply produced successfully"
