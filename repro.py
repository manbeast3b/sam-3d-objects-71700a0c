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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True,
                    choices=["gs", "mesh", "scene", "stage1", "layout"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="hf")
    args = ap.parse_args()

    metrics = {
        "product": args.product,
        "seed": args.seed,
        "status": "started",
    }
    t_all = _now()

    from inference import (
        Inference, load_image, load_single_mask, load_masks, make_scene,
    )

    config_path = f"checkpoints/{args.tag}/pipeline.yaml"
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
        outputs = [inference(image, m, seed=args.seed) for m in masks]
        metrics["infer_seconds"] = round(_now() - t0, 2)
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
        voxel = output["voxel"]
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
