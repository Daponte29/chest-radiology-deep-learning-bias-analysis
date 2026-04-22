"""
generate_stylized.py — Apply all style-transfer transforms to CheXpert frontal images.

Walks src/data/1/train/ and src/data/1/valid/ directly, finds all frontal
images (filename contains 'frontal'), and saves stylized variants alongside
the originals with a short suffix:

    view1_frontal.jpg       <- original (untouched)
    view1_frontal_gb.jpg    <- gaussian_blur  (texture bias)
    view1_frontal_ps.jpg    <- patch_shuffle  (texture bias)
    view1_frontal_ce.jpg    <- canny_edge     (shape bias)
    view1_frontal_pr.jpg    <- patch_rotation (shape bias)

Each image is opened once and all requested transforms are applied in the
same worker process, avoiding redundant I/O.

Usage:
    python -m src.data.style_transfer.generate_stylized
    python -m src.data.style_transfer.generate_stylized --transforms gaussian_blur --splits train
    python -m src.data.style_transfer.generate_stylized --workers 8
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running directly in VS Code as well as via `python -m`
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUFFIX_MAP = {
    "gaussian_blur":  "_gb",
    "patch_shuffle":  "_ps",
    "canny_edge":     "_ce",
    "patch_rotation": "_pr",
}

KNOWN_SUFFIXES = set(SUFFIX_MAP.values())
TARGET_SIZE    = (224, 224)
IMAGE_ROOT     = Path(_PROJECT_ROOT) / "src" / "data" / "1"


# ---------------------------------------------------------------------------
# Worker (runs in subprocess — must be top-level and import its own deps)
# ---------------------------------------------------------------------------

def _worker_init(project_root: str) -> None:
    """Add project root to sys.path in each worker process."""
    import sys, os
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def _process_image(job: tuple) -> list[str]:
    """
    Open one image and apply all requested transforms.
    Returns a list of error strings (empty on full success).
    """
    src_str, transform_names = job
    from pathlib import Path
    from PIL import Image

    from src.data.style_transfer.texture.gaussian_blur import apply as _gb
    from src.data.style_transfer.texture.patch_shuffle import apply as _ps
    from src.data.style_transfer.shape.canny_edge      import apply as _ce
    from src.data.style_transfer.shape.patch_rotation  import apply as _pr

    fn_map = {
        "gaussian_blur": (_gb, "_gb"),
        "patch_shuffle": (_ps, "_ps"),
        "canny_edge":    (_ce, "_ce"),
        "patch_rotation":(_pr, "_pr"),
    }

    src    = Path(src_str)
    errors = []

    try:
        # Open and resize once — transforms skip internal resize if already 224×224
        img = Image.open(src).convert("L").resize((224, 224), Image.BILINEAR)
    except Exception as e:
        return [f"{src.name} [open]: {e}"]

    for t_name in transform_names:
        t_fn, suffix = fn_map[t_name]
        dst = src.with_name(src.stem + suffix + src.suffix)
        if dst.exists():
            continue
        try:
            result = t_fn(img)
            result.save(dst)
        except Exception as e:
            errors.append(f"{src.name}[{suffix}]: {e}")

    return errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_frontal_images(root: Path) -> list[Path]:
    """
    Recursively find all original frontal JPEGs under root.
    Skips already-stylized files (those ending with a known suffix).
    """
    results = []
    for path in root.rglob("*.jpg"):
        stem = path.stem
        if stem.startswith("._"):          # macOS metadata files — not real images
            continue
        if "frontal" not in stem:
            continue
        if any(stem.endswith(s) for s in KNOWN_SUFFIXES):
            continue
        results.append(path)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate stylized CheXpert images alongside originals"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "valid"],
        choices=["train", "valid"],
    )
    parser.add_argument(
        "--transforms", nargs="+", default=list(SUFFIX_MAP.keys()),
        choices=list(SUFFIX_MAP.keys()),
    )
    parser.add_argument("--workers",    type=int, default=6)
    parser.add_argument("--image-root", default=str(IMAGE_ROOT))
    args = parser.parse_args()

    image_root = Path(args.image_root)

    # ------------------------------------------------------------------
    # Find all frontal images
    # ------------------------------------------------------------------
    all_images = []
    for split in args.splits:
        found = find_frontal_images(image_root / split)
        print(f"  {split}: {len(found):,} frontal images")
        all_images.extend(found)

    # One job per image — all transforms applied inside the worker
    jobs = [(str(src), args.transforms) for src in all_images]

    print(f"\nTotal images : {len(jobs):,}")
    print(f"Transforms   : {args.transforms}")
    print(f"Workers      : {args.workers}\n")

    # ------------------------------------------------------------------
    # Process with multiprocessing (bypasses GIL for CPU-bound transforms)
    # ------------------------------------------------------------------
    all_errors = []
    first_errors_printed = 0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(_PROJECT_ROOT,),
    ) as executor:
        futures = {executor.submit(_process_image, job): job[0] for job in jobs}

        with tqdm(total=len(futures), desc="Stylizing", dynamic_ncols=True, unit="img") as pbar:
            for future in as_completed(futures):
                errs = future.result()
                if errs:
                    all_errors.extend(errs)
                    # Print first 5 errors immediately so we can diagnose
                    for e in errs:
                        if first_errors_printed < 5:
                            tqdm.write(f"  ERROR: {e}")
                            first_errors_printed += 1
                pbar.update(1)
                if all_errors:
                    pbar.set_postfix(errors=len(all_errors))

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\nDone.  Images processed: {len(jobs):,}  |  Errors: {len(all_errors)}")

    if all_errors:
        log_path = image_root / f"stylize_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, "w") as f:
            json.dump(all_errors, f, indent=2)
        print(f"Full error log: {log_path}")

    print("\nSuffix reference:")
    for t_name in args.transforms:
        print(f"  {SUFFIX_MAP[t_name]}  ->  {t_name}")


if __name__ == "__main__":
    main()
