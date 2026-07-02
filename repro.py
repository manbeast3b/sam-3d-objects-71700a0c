# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Parametrized reproduction harness for SAM 3D Objects.

Runs one of the model's distinct "products" (output modes) on a fixed sample
image+mask and emits quantitative metrics so variants are directly comparable:

  gs        Gaussian splat  (single object)          -> splat.ply
  mesh      Textured mesh   (single object)          -> mesh.glb
  scene     Multi-object posed scene (all masks)     -> scene.ply
  stage1    Sparse-structure only (coarse voxels)    -> voxels.npy
  layout    Layout-posed reconstruction (pose R,t,s) -> layout.ply + pose.json

All products share the same pipeline load; they differ only in which decode /
post-process path is exercised. Metrics + timings go to
.openresearch/artifacts/metrics.json and are appended to EVAL.md.

Usage: python repro.py --product {gs,mesh,scene,stage1,layout} [--seed 42]
"""
import os
import sys
import json
import time
import argparse
import traceback

sys.path.append("notebook")

ARTIFACTS = os.path.join(".openresearch", "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

SAMPLE_DIR = "notebook/images/shutterstock_stylish_kidsroom_1640806567"
SAMPLE_IMAGE = os.path.join(SAMPLE_DIR, "image.png")
PRIMARY_MASK_INDEX = 14  # the object used by the upstream demo


def _now():
    return time.time()


def _gpu_mem_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    except Exception:
        pass
    return None


def _env_info():
    info = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["torch_error"] = repr(e)
    return info


def _seed_everything(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


REQUIRED_CKPTS = [
    "pipeline.yaml", "ss_generator.ckpt", "slat_generator.ckpt",
    "ss_decoder.ckpt", "slat_decoder_gs.ckpt", "slat_decoder_mesh.ckpt",
]


def selfcheck(tag):
    """Fast preflight: validate inputs, checkpoints, and heavy imports BEFORE
    paying for the (slow) model load. Returns a report dict; exits non-zero on
    any hard failure so a broken run dies in seconds, not after a 10-min load."""
    report = {"mode": "selfcheck", "checks": {}, "env": _env_info()}
    hard_fail = []

    def check(name, ok, detail=""):
        report["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            hard_fail.append(f"{name}: {detail}")

    check("sample_image", os.path.exists(SAMPLE_IMAGE), SAMPLE_IMAGE)
    primary_mask = os.path.join(SAMPLE_DIR, f"{PRIMARY_MASK_INDEX}.png")
    check("primary_mask", os.path.exists(primary_mask), primary_mask)

    ckpt_dir = f"checkpoints/{tag}"
    for f in REQUIRED_CKPTS:
        p = os.path.join(ckpt_dir, f)
        sz = os.path.getsize(p) if os.path.exists(p) else 0
        check(f"ckpt:{f}", sz > 0, f"{p} ({sz} bytes)")

    # Heavy imports are the #1 real failure (extension builds). Import them here
    # so selfcheck alone surfaces a broken pytorch3d/kaolin/gsplat build.
    for mod in ["torch", "pytorch3d", "kaolin", "gsplat"]:
        try:
            __import__(mod)
            check(f"import:{mod}", True)
        except Exception as e:
            check(f"import:{mod}", False, f"{type(e).__name__}: {e}")

    try:
        import inference  # noqa: F401  (notebook/inference.py; validates its imports)
        check("import:inference_module", True)
    except Exception as e:
        check("import:inference_module", False, f"{type(e).__name__}: {e}")

    report["status"] = "success" if not hard_fail else "failed"
    report["failures"] = hard_fail
    with open(os.path.join(ARTIFACTS, "selfcheck.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("SELFCHECK", json.dumps(report))
    if hard_fail:
        sys.exit(2)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", choices=["gs", "mesh", "scene", "stage1", "layout"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="hf")
    ap.add_argument("--selfcheck", action="store_true",
                    help="Validate inputs/checkpoints/imports and exit (no inference).")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck(args.tag)
        return
    if not args.product:
        ap.error("--product is required unless --selfcheck is given")

    metrics = {
        "product": args.product,
        "seed": args.seed,
        "status": "started",
        "env": _env_info(),
    }
    t_all = _now()

    from inference import (
        Inference, load_image, load_single_mask, load_masks, make_scene,
    )

    _seed_everything(args.seed)

    config_path = f"checkpoints/{args.tag}/pipeline.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"{config_path} not found; run checkpoint download (Stage 5) first")
    t0 = _now()
    inference = Inference(config_path, compile=False)
    metrics["load_seconds"] = round(_now() - t0, 2)
    pipe = inference._pipeline

    image = load_image(SAMPLE_IMAGE)
    metrics["image_hw"] = list(image.shape[:2])

    out_files = {}

    if args.product == "gs":
        mask = load_single_mask(SAMPLE_DIR, index=PRIMARY_MASK_INDEX)
        t0 = _now()
        output = inference(image, mask, seed=args.seed)
        metrics["infer_seconds"] = round(_now() - t0, 2)
        gs = output["gs"]
        n = int(gs.get_xyz.shape[0])
        metrics["num_gaussians"] = n
        path = "splat.ply"
        gs.save_ply(path)
        out_files["splat_ply"] = path

    elif args.product == "mesh":
        mask = load_single_mask(SAMPLE_DIR, index=PRIMARY_MASK_INDEX)
        t0 = _now()
        # Full mesh path: mesh postprocess + texture baking produce a GLB.
        output = pipe.run(
            image, mask, args.seed,
            stage1_only=False,
            with_mesh_postprocess=True,
            with_texture_baking=True,
            with_layout_postprocess=False,
            use_vertex_color=False,
            decode_formats=["mesh", "gaussian"],
        )
        metrics["infer_seconds"] = round(_now() - t0, 2)
        glb = output["glb"]
        metrics["num_vertices"] = int(len(glb.vertices))
        metrics["num_faces"] = int(len(glb.faces))
        path = "mesh.glb"
        glb.export(path)
        out_files["mesh_glb"] = path
        # also dump the companion splat for parity
        output["gs"].save_ply("mesh_gs.ply")
        out_files["mesh_gs_ply"] = "mesh_gs.ply"

    elif args.product == "scene":
        masks = load_masks(SAMPLE_DIR, extension=".png")
        metrics["num_objects"] = len(masks)
        t0 = _now()
        # Reconstruct each object; skip (don't abort) any single failure so one
        # hard mask can't kill the whole 27-object scene run.
        outputs, failed = [], []
        for i, m in enumerate(masks):
            try:
                outputs.append(inference(image, m, seed=args.seed))
            except Exception as e:
                failed.append({"index": i, "error": f"{type(e).__name__}: {e}"})
        metrics["infer_seconds"] = round(_now() - t0, 2)
        metrics["num_objects_ok"] = len(outputs)
        metrics["num_objects_failed"] = len(failed)
        if failed:
            metrics["failed_objects"] = failed
        if not outputs:
            raise RuntimeError("scene: every object reconstruction failed")
        scene_gs = make_scene(*outputs)
        metrics["num_gaussians"] = int(scene_gs.get_xyz.shape[0])
        path = "scene.ply"
        scene_gs.save_ply(path)
        out_files["scene_ply"] = path

    elif args.product == "stage1":
        mask = load_single_mask(SAMPLE_DIR, index=PRIMARY_MASK_INDEX)
        t0 = _now()
        output = pipe.run(
            image, mask, args.seed,
            stage1_only=True,  # coarse sparse-structure only: fast, cheap
        )
        metrics["infer_seconds"] = round(_now() - t0, 2)
        import numpy as np
        # stage1_only returns "voxel" ([N,3] coords); fall back to raw "coords".
        voxel = output.get("voxel")
        if voxel is None and "coords" in output:
            c = output["coords"]
            voxel = c[:, 1:] if hasattr(c, "__getitem__") else c
        if voxel is None:
            raise KeyError("stage1: neither 'voxel' nor 'coords' in output")
        try:
            voxel_np = voxel.detach().cpu().numpy()
        except Exception:
            voxel_np = np.asarray(voxel)
        metrics["num_voxels"] = int(voxel_np.shape[0])
        path = "voxels.npy"
        np.save(path, voxel_np)
        out_files["voxels_npy"] = path

    elif args.product == "layout":
        mask = load_single_mask(SAMPLE_DIR, index=PRIMARY_MASK_INDEX)
        t0 = _now()
        output = pipe.run(
            image, mask, args.seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=True,  # optimize object pose (R,t,s) in scene
            use_vertex_color=True,
            decode_formats=["gaussian"],
        )
        metrics["infer_seconds"] = round(_now() - t0, 2)
        gs = output["gs"]
        metrics["num_gaussians"] = int(gs.get_xyz.shape[0])

        def _tolist(k):
            v = output.get(k, None)
            if v is None:
                return None
            try:
                return v.detach().cpu().numpy().reshape(-1).tolist()
            except Exception:
                return None

        pose = {k: _tolist(k) for k in ("rotation", "translation", "scale")}
        metrics["pose"] = pose
        with open("pose.json", "w") as f:
            json.dump(pose, f, indent=2)
        out_files["pose_json"] = "pose.json"
        gs.save_ply("layout.ply")
        out_files["layout_ply"] = "layout.ply"

    metrics["gpu_peak_mem_mb"] = _gpu_mem_mb()
    metrics["total_seconds"] = round(_now() - t_all, 2)
    metrics["out_files"] = {}
    for k, p in out_files.items():
        if os.path.exists(p):
            metrics["out_files"][k] = {"path": p, "bytes": os.path.getsize(p)}
            # copy small text/geometry artifacts for later inspection
            try:
                if os.path.getsize(p) < 200 * 1024 * 1024:
                    import shutil
                    shutil.copy(p, os.path.join(ARTIFACTS, os.path.basename(p)))
            except Exception:
                pass
    metrics["status"] = "success"

    with open(os.path.join(ARTIFACTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("METRICS", json.dumps(metrics))
    return metrics


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        print(err, file=sys.stderr)
        with open(os.path.join(ARTIFACTS, "metrics.json"), "w") as f:
            json.dump({"status": "failed", "traceback": err}, f, indent=2)
        sys.exit(1)
