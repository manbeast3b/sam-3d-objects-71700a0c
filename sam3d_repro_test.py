# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
CPU-only regression tests for the reproduction harness (repro.py).

These do NOT need CUDA, the model, or checkpoints. They inject a fake `inference`
module and exercise every product's control flow, return-key access, metrics
assembly, and artifact writes, plus the --selfcheck preflight. Run with:

    python -m pytest sam3d_repro_test.py -q      # (or: python sam3d_repro_test.py)

The point is to protect the accuracy-critical contracts:
  * gs/scene go through Inference.__call__ (mask embedded as 0/255 RGBA).
  * mesh/stage1/layout build the SAME canonical RGBA via merge_mask_to_rgba and
    pass mask=None to pipe.run (NOT a raw 0/1 boolean alpha).
  * each product reads only keys the real pipeline actually returns.
"""
import os
import sys
import json
import types
import runpy
import shutil
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


class _T:
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype="float32")

    @property
    def shape(self):
        return self.arr.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr

    def reshape(self, *a):
        return _T(self.arr.reshape(*a))


class _GS:
    def __init__(self, n):
        self._n = n
        self._xyz = _T(np.zeros((n, 3)))

    @property
    def get_xyz(self):
        return self._xyz

    def save_ply(self, path):
        open(path, "wb").write(b"ply\n" + b"0" * 4096)


class _GLB:
    def __init__(self, v=5000, f=9000):
        self.vertices = np.zeros((v, 3))
        self.faces = np.zeros((f, 3), dtype="int64")

    def export(self, path):
        open(path, "wb").write(b"glTF" + b"0" * 8192)


def _fake_inference_module(seen):
    m = types.ModuleType("inference")

    class Pipe:
        def run(self, image, mask, seed, **kw):
            # record the alpha encoding actually handed to the pipeline
            arr = np.asarray(image)
            if arr.ndim == 3 and arr.shape[-1] == 4:
                seen["alpha_max"] = int(arr[..., 3].max())
            seen["mask_arg_is_none"] = mask is None
            if kw.get("stage1_only"):
                return {"voxel": _T(np.zeros((4096, 3))), "coords": _T(np.zeros((4096, 4)))}
            out = {"gaussian": [_GS(4321)], "gs": _GS(4321)}
            if "mesh" in (kw.get("decode_formats") or []):
                out["glb"] = _GLB()
            if kw.get("with_layout_postprocess"):
                out["rotation"] = _T([1.0, 0, 0, 0])
                out["translation"] = _T([0.1, 0.2, 0.3])
                out["scale"] = _T([1.337, 1.337, 1.337])
            return out

    class Inference:
        def __init__(self, config_path, compile=False):
            assert os.path.exists(config_path)
            self._pipeline = Pipe()

        def merge_mask_to_rgba(self, image, mask):
            a = (np.asarray(mask).astype("uint8") * 255)[..., None]
            return np.concatenate([np.asarray(image)[..., :3], a], axis=-1)

        def __call__(self, image, mask, seed=None, pointmap=None):
            rgba = self.merge_mask_to_rgba(image, mask)
            return self._pipeline.run(rgba, None, seed, stage1_only=False,
                                      with_mesh_postprocess=False,
                                      with_texture_baking=False,
                                      with_layout_postprocess=False,
                                      use_vertex_color=True)

    m.Inference = Inference
    m.load_image = lambda p: np.zeros((720, 960, 3), dtype="uint8")
    m.load_single_mask = lambda d, index=0, extension=".png": np.ones((720, 960), dtype=bool)
    m.load_masks = lambda d, indices_list=None, extension=".png": [np.ones((720, 960), dtype=bool) for _ in range(3)]
    m.make_scene = lambda *o, in_place=False: _GS(3 * 4321)
    return m


def _run(product, extra_argv=None):
    work = tempfile.mkdtemp(prefix="repro_test_")
    os.makedirs(f"{work}/checkpoints/hf")
    for f in ["pipeline.yaml", "ss_generator.ckpt", "slat_generator.ckpt",
              "ss_decoder.ckpt", "slat_decoder_gs.ckpt", "slat_decoder_mesh.ckpt"]:
        open(f"{work}/checkpoints/hf/{f}", "w").write("x" * 10)
    sd = f"{work}/notebook/images/shutterstock_stylish_kidsroom_1640806567"
    os.makedirs(sd)
    for name in ["image.png", "14.png", "0.png", "1.png", "2.png"]:
        open(f"{sd}/{name}", "w").write("x")
    shutil.copy(os.path.join(HERE, "repro.py"), f"{work}/repro.py")

    cwd = os.getcwd()
    seen = {}
    os.chdir(work)
    sys.modules["inference"] = _fake_inference_module(seen)
    sys.modules.pop("repro", None)
    sys.argv = ["repro.py"] + (extra_argv or ["--product", product, "--seed", "42"])
    code = 0
    try:
        runpy.run_path(f"{work}/repro.py", run_name="__main__")
    except SystemExit as e:
        code = e.code or 0
    finally:
        os.chdir(cwd)
    mj = f"{work}/.openresearch/artifacts/metrics.json"
    m = json.load(open(mj)) if os.path.exists(mj) else {"status": "NO_METRICS"}
    return code, m, seen, work


def test_products_succeed():
    for p in ["gs", "mesh", "scene", "stage1", "layout"]:
        code, m, seen, _ = _run(p)
        assert code == 0, (p, m)
        assert m.get("status") == "success", (p, m)


def test_mask_encoding_is_canonical_0_255():
    # Every product that hits pipe.run must send 0/255 alpha, mask=None (the
    # reference encoding), never a raw 0/1 boolean alpha.
    for p in ["gs", "mesh", "stage1", "layout", "scene"]:
        _, _, seen, _ = _run(p)
        if "alpha_max" in seen:
            assert seen["alpha_max"] == 255, (p, seen)
        assert seen.get("mask_arg_is_none") is True, (p, seen)


def test_product_specific_metrics():
    _, m, _, _ = _run("gs");      assert m["num_gaussians"] > 0
    _, m, _, _ = _run("mesh");    assert m["num_vertices"] > 0 and m["num_faces"] > 0
    _, m, _, _ = _run("stage1");  assert m["num_voxels"] > 0
    _, m, _, _ = _run("layout");  assert m["pose"]["rotation"] and m["pose"]["scale"]
    _, m, _, _ = _run("scene");   assert m["num_objects_ok"] >= 1


def test_selfcheck_file_checks_pass():
    code, _, _, work = _run("selfcheck", extra_argv=["--selfcheck"])
    rep = json.load(open(f"{work}/.openresearch/artifacts/selfcheck.json"))
    file_checks = {k: v["ok"] for k, v in rep["checks"].items()
                   if k.startswith(("sample", "primary", "ckpt"))}
    assert all(file_checks.values()), file_checks
    assert rep["checks"]["import:inference_module"]["ok"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    sys.exit(1 if failed else 0)
