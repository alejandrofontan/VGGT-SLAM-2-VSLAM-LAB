"""
Module: VGGT-SLAM 2.0 - VSLAM-LAB entry point (mono)
- Author: Alejandro Fontan Villacampa
- Version: 2.0
- Created: 2026-01-03
- Updated: 2026-09-15
- License: BSD-2-Clause (VGGT-SLAM)

Runs VGGT-SLAM 2.0 on one VSLAM-LAB sequence: frames come from the experiment's rgb csv (in csv
order, timestamps kept alongside the paths), keyframes are selected by optical-flow disparity as
upstream does, and the optimized camera-to-world poses are written to
<exp_folder>/<exp_it>_KeyFrameTrajectory.csv. Mirrors upstream main.py minus the open-set object
detection (--run_os), which needs Perception Encoder / SAM 3.
"""

import argparse
import time
import webbrowser
from pathlib import Path

import cv2
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from vggt_slam.solver import Solver
from vggt.models.vggt import VGGT

VGGT_WEIGHTS_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VGGT-SLAM 2.0 (VSLAM-LAB entry point, mono)")

    # VSLAM-LAB fixed arguments (BaselineVSLAMLAB.build_execute_command)
    parser.add_argument("--sequence_path", type=Path, required=True)
    parser.add_argument("--calibration_yaml", type=Path, required=True, help="unused: VGGT predicts its own intrinsics")
    parser.add_argument("--rgb_csv", type=Path, required=True)
    parser.add_argument("--exp_folder", type=Path, required=True)
    parser.add_argument("--exp_it", type=str, default="0")
    parser.add_argument("--settings_yaml", type=Path, default=None)
    parser.add_argument("--verbose", type=str, default="0", help="1 starts the viser viewer and updates it per submap")
    parser.add_argument("--mode", type=str, default="mono", choices=["mono"], help="VSLAM-LAB mode (selects this entry point; only mono is supported)")

    # VGGT-SLAM parameters (defaults = upstream main.py / evals/eval_tum.sh)
    parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
    parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
    parser.add_argument("--max_loops", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW or 0 to disable loop closures.")
    parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
    parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
    parser.add_argument("--lc_thres", type=float, default=0.95, help="Threshold for image retrieval. Range: [0, 1.0]. Higher = more loop closures")

    # Viewer (only used when verbose is on)
    parser.add_argument("--vis_imgs", action="store_true", help="Show camera images in the viser frustums. By default only the frustums are shown (faster visualization)")
    parser.add_argument("--vis_voxel_size", type=float, default=None, help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Default: no downsampling")
    parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow for keyframe selection")
    return parser


def load_frames(sequence_path: Path, rgb_csv: Path, settings_yaml: Path | None) -> tuple[list[str], list[int]]:
    """Frame paths and timestamps (ns) of the mono camera named by the settings yaml, in csv order."""
    cam_name = "rgb_0"
    if settings_yaml is not None and settings_yaml.is_file():
        with open(settings_yaml, "r") as f:
            cam_name = (yaml.safe_load(f) or {}).get("cam_mono", cam_name)

    df = pd.read_csv(rgb_csv)
    image_names = [str(sequence_path / p) for p in df[f"path_{cam_name}"]]
    timestamps = df[f"ts_{cam_name} (ns)"].astype("int64").tolist()
    return image_names, timestamps


def wait_for_viewer_clients(server, first_client_timeout_s: float = 60.0) -> None:
    """Keep the final map viewable: return once every viser client has disconnected, or when
    nobody connected within first_client_timeout_s (headless machine), or on Ctrl+C."""
    t0 = time.time()
    had_clients = False
    try:
        while True:
            num_clients = len(server.get_clients())
            had_clients = had_clients or num_clients > 0
            if had_clients and num_clients == 0:
                print("(viser) All clients disconnected. Shutting down server.")
                return
            if not had_clients and time.time() - t0 > first_client_timeout_s:
                print(f"(viser) No client connected within {first_client_timeout_s:.0f} s. Shutting down server.")
                return
            time.sleep(0.5)
    except KeyboardInterrupt:
        return


def main() -> None:
    args = build_parser().parse_args()
    vis_map = bool(int(args.verbose))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        lc_thres=args.lc_thres,
        vis_voxel_size=args.vis_voxel_size,
        vis_imgs=args.vis_imgs,
        enable_viewer=vis_map,
    )
    if vis_map:
        # The framework redirects stdout to a log file, so the viser URL is opened rather than printed only
        viewer_url = f"http://localhost:{solver.viewer.server.get_port()}"
        print(f"Viser viewer at {viewer_url}")
        webbrowser.open(viewer_url)

    print("Initializing and loading VGGT model...")
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_WEIGHTS_URL))
    model.eval()
    model = model.to(torch.bfloat16)  # use half precision
    model = model.to(device)

    image_names, timestamps = load_frames(args.sequence_path, args.rgb_csv, args.settings_yaml)
    print(f"Loading images from {args.sequence_path}...")
    print(f"Found {len(image_names)} images")

    image_names_subset: list[str] = []
    timestamps_subset: list[int] = []
    num_submaps = 0
    num_keyframes = 0
    total_time_start = time.time()
    for image_name, timestamp in tqdm(zip(image_names, timestamps), total=len(image_names)):
        img = cv2.imread(image_name)
        if solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow):
            image_names_subset.append(image_name)
            timestamps_subset.append(timestamp)
            num_keyframes += 1

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            num_submaps += 1
            predictions = solver.run_predictions(image_names_subset, model, args.max_loops, timestamps=timestamps_subset)
            solver.add_points(predictions)
            solver.graph.optimize()

            if vis_map:
                if len(predictions["detected_loops"]) > 0:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()

            # Reset for next submap.
            image_names_subset = image_names_subset[-args.overlapping_window_size:]
            timestamps_subset = timestamps_subset[-args.overlapping_window_size:]

    total_time = time.time() - total_time_start
    print(f"{num_keyframes} keyframes from {len(image_names)} frames, {num_submaps} submaps processed")
    print("Total time:", total_time)
    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    keyframe_csv = args.exp_folder / f"{args.exp_it.zfill(5)}_KeyFrameTrajectory.csv"
    solver.map.write_poses_to_file_vslamlab(keyframe_csv, solver.graph)
    print(f"Trajectory written to {keyframe_csv}")

    if vis_map:
        solver.update_all_submap_vis()  # final optimized map
        wait_for_viewer_clients(solver.viewer.server)


if __name__ == "__main__":
    main()
