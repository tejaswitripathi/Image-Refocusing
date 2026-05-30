"""
Orchestrates dataset generation across every Blender scene.

Designed to run on a VM where the .blend files are NOT stored locally; they
live in S3 at s3://<S3_BUCKET>/<S3_PREFIX>/<scene>/<scene>.blend. Only the
data_collection_<scene>.py scripts ship with the repo.

For each scene under ./scenes/ this script:
    1. Downloads the scene's .blend file from
       s3://<S3_BUCKET>/<S3_PREFIX>/<scene>/ into scenes/<scene>/.
    2. Runs the scene's data_collection_<scene>.py inside Blender (headless).
       The script renders into scenes/<scene>/dataset/ (the "//dataset/" path
       resolves relative to the .blend file) and uploads each rendered sample to
       s3://<S3_BUCKET>/<S3_PREFIX>/<scene>/dataset/ as it is produced, deleting
       it locally afterwards.
    3. Deletes the downloaded .blend (and any leftover local dataset/) to free
       disk before the next scene.

Usage:
    python generate_dataset.py                 # run every scene
    python generate_dataset.py cafe house      # run only the named scenes
    python generate_dataset.py --keep-local    # don't delete the downloaded blend / leftovers
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

# ----------------------------
# CONFIG
# ----------------------------

SCENES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes")

# Where each data_collection script writes its renders ("//dataset/" in the
# blend file resolves to <scene_folder>/dataset/).
DATASET_SUBDIR = "dataset"

# S3 destination: s3://<bucket>/<prefix>/<scene>/...
S3_BUCKET = "tejas-blender-bucket"
S3_PREFIX = "defocus-dataset"

# Scenes to skip during discovery (e.g. broken or unwanted scenes).
IGNORED_SCENES = {"bottle"}

def resolve_blender():
    """Locate the Blender executable.

    Resolution order:
      1. The BLENDER env var (explicit override).
      2. blender on PATH.
      3. macOS app bundle.
      4. A Blender install under /workspace (the VM layout, e.g.
         /workspace/blender-4.2.0-linux-x64/blender).
    """
    env = os.environ.get("BLENDER")
    if env:
        return env

    on_path = shutil.which("blender")
    if on_path:
        return on_path

    macos = "/Applications/Blender.app/Contents/MacOS/Blender"
    if os.path.exists(macos):
        return macos

    workspace_candidates = (
        glob.glob("/workspace/blender")
        + glob.glob("/workspace/blender*/blender")
        + glob.glob("/workspace/**/blender", recursive=True)
    )
    for cand in workspace_candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand

    # Fall back to the bare name; will error clearly at run time if missing.
    return "blender"


# Blender executable. Override with the BLENDER env var if needed.
BLENDER = resolve_blender()


# ----------------------------
# HELPERS
# ----------------------------

def discover_scenes():
    """Return a sorted list of scene names that ship a data_collection script.

    The .blend files are not expected locally (they are pulled from S3), so a
    scene only needs its data_collection_<scene>.py to be runnable.
    """
    scenes = []

    for entry in sorted(os.listdir(SCENES_DIR)):
        if entry in IGNORED_SCENES:
            continue

        scene_dir = os.path.join(SCENES_DIR, entry)
        if not os.path.isdir(scene_dir):
            continue

        if find_collection_script(scene_dir):
            scenes.append(entry)

    return scenes


def find_blend_file(scene_dir):
    """Find a local .blend file in a scene folder (ignoring Blender's .blend1 backups)."""
    candidates = [
        f for f in glob.glob(os.path.join(scene_dir, "*.blend"))
        if not f.endswith(".blend1")
    ]
    return candidates[0] if candidates else None


def find_collection_script(scene_dir):
    """Find the data_collection_*.py script in a scene folder."""
    candidates = glob.glob(os.path.join(scene_dir, "data_collection_*.py"))
    return candidates[0] if candidates else None


def s3_find_blend_name(scene):
    """List the scene's S3 prefix and return the .blend file name (ignoring .blend1)."""
    s3_prefix = f"s3://{S3_BUCKET}/{S3_PREFIX}/{scene}/"

    result = subprocess.run(
        ["aws", "s3", "ls", s3_prefix],
        check=True,
        capture_output=True,
        text=True,
    )

    for line in result.stdout.splitlines():
        # File lines look like: "2026-05-30 14:59:00   37117339 cafe_scene.blend"
        # Directory lines look like: "                           PRE dataset/"
        name = line.split()[-1]
        if name.endswith(".blend"):
            return name

    return None


def download_blend(scene):
    """Download the scene's .blend file from S3 into scenes/<scene>/ and return its local path."""
    scene_dir = os.path.join(SCENES_DIR, scene)

    # Use a local copy if one already exists (e.g. running on the dev machine).
    local = find_blend_file(scene_dir)
    if local:
        print(f"[{scene}] Using local blend file: {local}")
        return local

    blend_name = s3_find_blend_name(scene)
    if not blend_name:
        raise FileNotFoundError(
            f"No .blend file found at s3://{S3_BUCKET}/{S3_PREFIX}/{scene}/"
        )

    os.makedirs(scene_dir, exist_ok=True)
    local_path = os.path.join(scene_dir, blend_name)
    s3_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/{scene}/{blend_name}"

    print(f"[{scene}] Downloading {s3_uri} -> {local_path}")
    subprocess.run(["aws", "s3", "cp", s3_uri, local_path], check=True)

    return local_path


def run_scene(scene, blend_file):
    """Run the scene's data_collection script inside Blender (headless)."""
    scene_dir = os.path.join(SCENES_DIR, scene)
    script = find_collection_script(scene_dir)

    cmd = [
        BLENDER,
        "--background",
        blend_file,
        "--python",
        script,
    ]

    print(f"\n[{scene}] Rendering with Blender")
    print(f"[{scene}] $ {' '.join(cmd)}")

    subprocess.run(cmd, check=True)


def delete_local_files(scene, blend_file=None):
    """Delete the local dataset/ folder and the downloaded .blend for a scene."""
    local_dir = os.path.join(SCENES_DIR, scene, DATASET_SUBDIR)

    if os.path.isdir(local_dir):
        print(f"[{scene}] Deleting local dataset: {local_dir}")
        shutil.rmtree(local_dir)

    if blend_file and os.path.isfile(blend_file):
        print(f"[{scene}] Deleting downloaded blend: {blend_file}")
        os.remove(blend_file)


# ----------------------------
# MAIN
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate the defocus dataset across all Blender scenes.")
    parser.add_argument("scenes", nargs="*", help="Specific scene names to run (default: all).")
    parser.add_argument("--keep-local", action="store_true", help="Do not delete the downloaded blend / leftover output.")
    args = parser.parse_args()

    available = discover_scenes()

    if args.scenes:
        unknown = [s for s in args.scenes if s not in available]
        if unknown:
            print(f"Unknown scene(s): {', '.join(unknown)}")
            print(f"Available scenes: {', '.join(available)}")
            sys.exit(1)
        scenes = args.scenes
    else:
        scenes = available

    print(f"Blender: {BLENDER}")
    print(f"Scenes to process ({len(scenes)}): {', '.join(scenes)}")

    failures = []

    for scene in scenes:
        try:
            blend_file = download_blend(scene)

            run_scene(scene, blend_file)

            if not args.keep_local:
                delete_local_files(scene, blend_file)

            print(f"[{scene}] Done.")

        except subprocess.CalledProcessError as e:
            print(f"[{scene}] FAILED (exit code {e.returncode}). Keeping local files.")
            failures.append(scene)
        except Exception as e:
            print(f"[{scene}] FAILED: {e}. Keeping local files.")
            failures.append(scene)

    print("\n========================================")
    if failures:
        print(f"Completed with failures in: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("All scenes generated successfully!")


if __name__ == "__main__":
    main()
