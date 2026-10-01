import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2
import tifffile
import matplotlib.pyplot as plt


from sam3.model_builder import _load_checkpoint, build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def load_tiff(input_path):
    with tifffile.TiffFile(input_path) as tif:
        data = tif.asarray()

        intensities = data.sum(axis=(1, 2))
        intensities = intensities.astype(np.float32)
        
        return intensities


def find_quiescent_ranges(mask, min_length=100):
    ranges = []
    start = None

    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_length:
                ranges.append((start + 1, i))  # 1-based, end exclusive
            start = None

    if start is not None and len(mask) - start >= min_length:
        ranges.append((start + 1, len(mask)))

    return ranges


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


def plot_persistent_mask_brightness(
    frames_rgb: list[np.ndarray],
    frame_masks: list[np.ndarray],
    clip_label: str,
    output_dir: Path,
    persistence_threshold: float = 0.9,
) -> None:
    if not frames_rgb or not frame_masks:
        return

    mask_stack = np.stack(frame_masks, axis=0).astype(np.float32)  # (T, H, W)
    persistent_mask = mask_stack.mean(axis=0) >= persistence_threshold

    brightness = []
    for frame_rgb in frames_rgb:
        if persistent_mask.any():
            brightness.append(float(frame_rgb[persistent_mask].sum()))
        else:
            brightness.append(0.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(brightness)), brightness)
    plt.xlabel("Frame Index")
    plt.ylabel("Total Brightness")
    plt.title(f"Persistent Mask Brightness ({clip_label})")
    plt.tight_layout()
    plot_path = output_dir / f"{clip_label}_persistent_brightness.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()


    
def segment_quiescent_ranges(
    input_tiff: str,
    output_tiff: str,
    finetune_ckpt: str,
    prompt: str,
    ranges: list[tuple[int, int]],
    confidence_threshold: float = 0.25,
    alpha: float = 0.5,
    max_clips: int | None = None,
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

    out_path = Path(output_tiff)
    total_ranges = len(ranges)

    with torch.inference_mode():
        for range_idx, (start1, end1) in enumerate(ranges, start=1):
            if max_clips is not None and range_idx > max_clips:
                break
            # `find_quiescent_ranges` returns 1-based inclusive ranges
            start0 = start1 - 1
            end0 = end1  # end is inclusive in 1-based -> exclusive in 0-based

            tiff_frames: list[Image.Image] = []
            frames_rgb: list[np.ndarray] = []
            frame_masks: list[np.ndarray] = []

            for frame_rgb, window_idx, total_window_frames in _extract_tiff_frames(input_tiff, start_frame=start0, end_frame=end0):
                image = Image.fromarray(frame_rgb)
                state = processor.set_image(image)
                state = processor.set_text_prompt(prompt=prompt, state=state)

                masks = state.get("masks", None)
                scores = state.get("scores", None)

                masks_np = []
                if masks is not None:
                    masks_np = masks.detach().cpu().numpy()
                    if masks_np.ndim == 4:
                        masks_np = masks_np[:, 0, :, :]

                if scores is not None and len(masks_np) > 0:
                    scores_np = scores.detach().cpu().numpy()
                    keep = scores_np >= confidence_threshold
                    masks_np = masks_np[keep]

                if len(masks_np) > 0:
                    frame_mask = np.any(masks_np.astype(bool), axis=0)
                    out_frame = _mask_overlay_rgb(frame_rgb, masks_np, alpha=alpha)
                else:
                    frame_mask = np.zeros(frame_rgb.shape[:2], dtype=bool)
                    out_frame = frame_rgb

                tiff_frames.append(Image.fromarray(out_frame))
                frames_rgb.append(frame_rgb)
                frame_masks.append(frame_mask)

                if total_window_frames > 0:
                    progress = (window_idx + 1) / total_window_frames * 100
                    print(
                        f"Range {range_idx}/{total_ranges} ({start1}-{end1}) - "
                        f"Processed {window_idx + 1}/{total_window_frames} frames ({progress:.1f}%)",
                        end="\r",
                    )

            if tiff_frames:
                range_out = out_path.with_name(f"{out_path.stem}_{start1}_{end1}.tif")
                first, *rest = tiff_frames
                first.save(range_out, save_all=True, append_images=rest)
                plot_persistent_mask_brightness(
                    frames_rgb=frames_rgb,
                    frame_masks=frame_masks,
                    clip_label=f"{out_path.stem}_{start1}_{end1}",
                    output_dir=out_path.parent,
                    persistence_threshold=0.9,
                )
            print(f"\nSaved range {range_idx}/{total_ranges}: {start1}-{end1}")
    




def main():
    CONFIG = {
        "input": r"C:\Users\keleslab\sam3\Test_Data\MMStack_Default1.tif",
        "output": r"C:\Users\keleslab\sam3\FullProcessTesting\tiffoutput2",
        "ckpt": r"C:\Users\keleslab\sam3\exp_runs\fly_ft\checkpoints\checkpoint.pt",
        "prompt": "leg",
        "threshold": 0.25,
        "alpha": 0.5,
        "intensity_threshold": 2.2e7,
        "max_clips": 1,
    }
    
    parser = argparse.ArgumentParser(
        description="Find low intensity segments in a TIFF image and save the results as a new TIFF.")
    
    parser.add_argument("--input", default=CONFIG["input"], help="Path to input TIFF stack")
    parser.add_argument("--output", default=CONFIG["output"], help="Path to output TIFF stack")
    parser.add_argument("--ckpt", default=CONFIG["ckpt"], help="Path to finetuned checkpoint .pt")
    parser.add_argument("--prompt", default=CONFIG["prompt"], help="Text prompt for segmentation")
    parser.add_argument("--intensity_threshold", type=float, default=CONFIG["intensity_threshold"], help="Intensity threshold for quiescence detection")
    parser.add_argument("--threshold", type=float, default=CONFIG["threshold"], help="Segmentation threshold")
    parser.add_argument("--alpha", type=float, default=CONFIG["alpha"], help="Alpha blending for overlay")
    parser.add_argument(
        "--max-clips",
        type=int,
        default=CONFIG["max_clips"],
        help="Maximum number of quiescent clips to process (use 0 or omit for all).",
    )


    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input TIFF not found: {input_path}")
    print(input_path)
    
    intensities = load_tiff(input_path)
    print(len(intensities))
    
    frames_below_threshold = intensities < args.intensity_threshold
    print(len(frames_below_threshold))

    ranges = find_quiescent_ranges(frames_below_threshold, min_length=100)
    print(ranges)

    if ranges:
        max_clips = None if args.max_clips == 0 else int(args.max_clips)
        segment_quiescent_ranges(
            input_tiff=str(input_path),
            output_tiff=str(args.output),
            finetune_ckpt=str(args.ckpt),
            prompt=args.prompt,
            ranges=ranges,
            confidence_threshold=float(args.threshold),
            alpha=float(args.alpha),
            max_clips=max_clips,
        )


    





if __name__ == "__main__":
    main()