import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2


from sam3.model_builder import _load_checkpoint, build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor



def _dump_array_stats(arr: np.ndarray, base_name: str, dump_dir: Path, save_npy: bool = False) -> None:
    """Compute numeric stats per channel and save to a text file; optionally save .npy."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    stats["dtype"] = str(arr.dtype)
    stats["shape"] = arr.shape
    if arr.ndim == 2:
        a = arr.astype(np.float64)
        stats.update({"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std())})
    elif arr.ndim == 3:
        chans = arr.shape[2]
        for c in range(chans):
            a = arr[..., c].astype(np.float64)
            stats[f"ch{c}_min"] = float(a.min())
            stats[f"ch{c}_max"] = float(a.max())
            stats[f"ch{c}_mean"] = float(a.mean())
            stats[f"ch{c}_std"] = float(a.std())
    else:
        a = arr.astype(np.float64).ravel()
        stats.update({"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std())})

    stats_path = dump_dir / f"{base_name}_stats.txt"
    with open(stats_path, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

    if save_npy:
        npy_path = dump_dir / f"{base_name}.npy"
        np.save(npy_path, arr)


def _mask_overlay_rgb(frame_rgb: np.ndarray, masks: np.ndarray, alpha: float) -> np.ndarray:
    overlay = frame_rgb.copy()

    for _,mask in enumerate(masks):
        color = (255, 0, 255)
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            continue
        color_layer = np.zeros_like(overlay, dtype=np.uint8)
        color_layer[:] = color
        overlay = np.where(
            mask_bool[..., None],
            (alpha * color_layer + (1 - alpha) * overlay).astype(np.uint8),
            overlay,
        )
    return overlay



def _extract_tiff_frames(
    tiff_path: str,
    start_frame: int,
    end_frame: int | None,
):
    with Image.open(tiff_path) as img:
        total_frames = getattr(img, "n_frames", 1)
        if total_frames <= 0:
            raise ValueError("TIFF stack has no frames.")

        end_frame = total_frames if end_frame is None else min(end_frame, total_frames)
        if end_frame <= start_frame:
            raise ValueError("End frame must be greater than start frame.")

        total_window_frames = end_frame - start_frame
        for idx in range(start_frame, end_frame):
            img.seek(idx)
            img_arr = np.array(img)  # could be uint16/uint8/float
            if img_arr.dtype != np.uint8:
                mn, mx = img_arr.min(), img_arr.max()
                if mx > mn:
                    img8 = ((img_arr - mn) / (mx - mn) * 255).astype(np.uint8)
                else:
                    img8 = img_arr.astype(np.uint8)
            else:
                img8 = img_arr
            pil_rgb = Image.fromarray(img8).convert("RGB")
            frame_rgb = np.array(pil_rgb)  #yield RGB array (no color conversion here)
            yield frame_rgb, idx - start_frame, total_window_frames



def run_segmented_clip(
    input_tiff: str,    
    output_tiff: str,
    finetune_ckpt: str,
    prompt: str,
    start_frame: int,
    end_frame: int | None,
    confidence_threshold: float = 0.25,
    alpha: float = 0.5,
    gamma: float = 1.0,
    brightness: float = 1.0,
    save_first_frame: bool = False,
    first_frame_path: str | None = None,
    save_first_frame_raw: bool = False,
    first_frame_raw_path: str | None = None,
    dump_stats: bool = False,
    dump_npy: bool = False,
    dump_dir: str | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_sam3_image_model(
        checkpoint_path=None,
        load_from_HF=True,
        device=device,
        eval_mode=True,
        enable_segmentation=True,
    )
    _load_checkpoint(model, finetune_ckpt)
    processor = Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)

    tiff_frames: list[Image.Image] = []
    first_frame_saved = False
    resolved_first_frame_path = first_frame_path
    if save_first_frame and not resolved_first_frame_path:
        out_path = Path(output_tiff)
        resolved_first_frame_path = str(out_path.with_name(f"{out_path.stem}_first_frame.png"))

    # Prepare dump directory if requested
    resolved_dump_dir: Path | None = None
    if dump_stats or dump_npy:
        resolved_dump_dir = Path(dump_dir) if dump_dir else Path(output_tiff).parent
        resolved_dump_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        if save_first_frame_raw:
            raw_path = first_frame_raw_path
            if not raw_path:
                out_path = Path(output_tiff)
                raw_path = str(out_path.with_name(f"{out_path.stem}_first_frame_raw.tif"))
            with Image.open(input_tiff) as img:
                img.seek(start_frame)
                img.save(raw_path)
                if resolved_dump_dir is not None:
                    raw_arr = np.array(img)
                    _dump_array_stats(raw_arr, "first_frame_raw", resolved_dump_dir, save_npy=dump_npy)

        frame_iter = _extract_tiff_frames(
            input_tiff, start_frame=start_frame, end_frame=end_frame
        )

        for frame_rgb, window_idx, total_window_frames in frame_iter:
            # frame_rgb is an RGB numpy array
            if save_first_frame and not first_frame_saved:
                Image.fromarray(frame_rgb).save(resolved_first_frame_path)
                if resolved_dump_dir is not None and (dump_stats or dump_npy):
                    _dump_array_stats(frame_rgb, "first_frame_input", resolved_dump_dir, save_npy=dump_npy)
                first_frame_saved = True
           
            if first_frame_saved and resolved_dump_dir is not None and (dump_stats or dump_npy):
                _dump_array_stats(frame_rgb, "first_frame_gamma", resolved_dump_dir, save_npy=dump_npy)
            image = Image.fromarray(frame_rgb)
            state = processor.set_image(image)
            state = processor.set_text_prompt(prompt=prompt, state=state)

            masks = state.get("masks", None)
            scores = state.get("scores", None)

            masks_np = []
            if masks is not None:
                masks_np = masks.detach().cpu().numpy()
                # masks: (N, 1, H, W) or (N, H, W)
                if masks_np.ndim == 4:
                    masks_np = masks_np[:, 0, :, :]

            if scores is not None and len(masks_np) > 0:
                scores_np = scores.detach().cpu().numpy()
                keep = scores_np >= confidence_threshold
                masks_np = masks_np[keep]

            if len(masks_np) > 0:
                out_frame = _mask_overlay_rgb(frame_rgb, masks_np, alpha=alpha)
            else:
                out_frame = frame_rgb

            # Prepare RGB for saving via PIL
            out_rgb = out_frame
            tiff_frames.append(Image.fromarray(out_rgb))

            # Dump output stats for first frame after overlay
            if window_idx == 0 and resolved_dump_dir is not None and (dump_stats or dump_npy):
                _dump_array_stats(out_rgb, "first_frame_output", resolved_dump_dir, save_npy=dump_npy)

            if total_window_frames > 0:
                progress = (window_idx + 1) / total_window_frames * 100
                print(
                    f"Processed {window_idx + 1}/{total_window_frames} frames ({progress:.1f}%)",
                    end="\r",
                )

    if tiff_frames:
        first, *rest = tiff_frames
        first.save(output_tiff, save_all=True, append_images=rest)
    print("\nDone.")


def main():
    # Set defaults here if you prefer not to use CLI flags
    CONFIG = {
        "input": r"C:\Users\keleslab\sam3\Test_Data\MMStack_Default1.tif",
        "output": r"C:\Users\keleslab\sam3\Test_Data\tiffoutput.tif",
        "ckpt": r"C:\Users\keleslab\sam3\exp_runs\fly_ft\checkpoints\checkpoint.pt",
        "prompt": "leg",
        "start": 1100,  # start frame index for TIFF stack
        "end": 1200, # end frame index for TIFF stack
        "threshold": 0.25,
        "alpha": 0.3,
        "save_first_frame": True,
        "first_frame_path": None,
        "save_first_frame_raw": True,
        "first_frame_raw_path": None,
        "dump_stats": True,
        "dump_npy": False,
        "dump_dir": None,
    }

    parser = argparse.ArgumentParser(
        description="Extract a frame window from a TIFF stack, run SAM3 image segmentation per frame, and write a TIFF stack."
    )
    parser.add_argument("--input", default=CONFIG["input"], help="Path to input TIFF stack")
    parser.add_argument("--output", default=CONFIG["output"], help="Path to output TIFF stack")
    parser.add_argument("--ckpt", default=CONFIG["ckpt"], help="Path to finetuned checkpoint .pt")
    parser.add_argument("--prompt", default=CONFIG["prompt"], help="Text prompt for segmentation")
    parser.add_argument(
        "--start",
        type=int,
        default=CONFIG["start"],
        help="Start frame index (TIFF stack)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=CONFIG["end"],
        help="End frame index (TIFF stack)",
    )
    parser.add_argument("--threshold", type=float, default=CONFIG["threshold"], help="Score threshold")
    parser.add_argument("--alpha", type=float, default=CONFIG["alpha"], help="Mask overlay alpha")
    parser.add_argument(
        "--save-first-frame",
        action="store_true",
        default=CONFIG["save_first_frame"],
        help="Save the first input frame of the segment as a PNG.",
    )
    parser.add_argument(
        "--first-frame-path",
        type=str,
        default=CONFIG["first_frame_path"],
        help="Path for saving the first input frame (default: <output>_first_frame.png).",
    )
    parser.add_argument(
        "--save-first-frame-raw",
        action="store_true",
        default=CONFIG["save_first_frame_raw"],
        help="Save the first input frame as raw TIFF (preserves bit depth; TIFF inputs only).",
    )
    parser.add_argument(
        "--first-frame-raw-path",
        type=str,
        default=CONFIG["first_frame_raw_path"],
        help="Path for saving the raw first frame (default: <output>_first_frame_raw.tif).",
    )
    parser.add_argument(
        "--dump-stats",
        action="store_true",
        default=CONFIG["dump_stats"],
        help="Dump numeric stats (min/max/mean/std) for the first frame variants.",
    )
    parser.add_argument(
        "--dump-npy",
        action="store_true",
        default=CONFIG["dump_npy"],
        help="Also save the dumped arrays to .npy files alongside stats.",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=CONFIG["dump_dir"],
        help="Directory to place dumps (defaults to output directory).",
    )

    args = parser.parse_args()

    if not args.input or not args.output or not args.ckpt or not args.prompt:
        raise ValueError("Set CONFIG values or pass --input --output --ckpt --prompt")
    if args.end is not None and args.end <= args.start:
        raise ValueError("End frame must be greater than start frame.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input TIFF not found: {input_path}")

    run_segmented_clip(
        input_tiff=str(input_path),
        output_tiff=str(args.output),
        finetune_ckpt=str(args.ckpt),
        prompt=args.prompt,
        start_frame=int(args.start),
        end_frame=int(args.end) if args.end is not None else None,
        confidence_threshold=float(args.threshold),
        alpha=float(args.alpha),
        save_first_frame=bool(args.save_first_frame),
        first_frame_path=args.first_frame_path,
        save_first_frame_raw=bool(args.save_first_frame_raw),
        first_frame_raw_path=args.first_frame_raw_path,
        dump_stats=bool(args.dump_stats),
        dump_npy=bool(args.dump_npy),
        dump_dir=args.dump_dir,
    )
    

if __name__ == "__main__":
    main()
