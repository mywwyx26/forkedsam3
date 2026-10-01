# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
import logging

import torch

try:
    from cc_torch import get_connected_components

    HAS_CC_TORCH = True
except ImportError:
    logging.debug(
        "cc_torch not found. Consider installing for better performance. Command line:"
        " pip install git+https://github.com/ronghanghu/cc_torch.git"
    )
    HAS_CC_TORCH = False


def connected_components_cpu_single(values: torch.Tensor):
    assert values.dim() == 2
    values_np = values.to(device="cpu").numpy()

    # Prefer scikit-image when available.
    try:
        from skimage.measure import label  # type: ignore

        labels, num = label(values_np, return_num=True)
    except Exception:
        # Fall back to OpenCV (commonly available in this project via video I/O).
        try:
            import cv2  # type: ignore

            # OpenCV expects 0/1 or 0/255 uint8 for binary masks.
            values_u8 = (values_np != 0).astype("uint8")
            num, labels = cv2.connectedComponents(values_u8, connectivity=8)
            # OpenCV counts background as label 0.
            num = int(num) - 1
        except Exception as e:
            raise ImportError(
                "No CPU connected-components backend found. Install one of: "
                "'scikit-image' (recommended) or ensure 'opencv-python' is installed."
            ) from e

    labels = torch.from_numpy(labels)
    counts = torch.zeros_like(labels)
    for i in range(1, num + 1):
        cur_mask = labels == i
        cur_count = cur_mask.sum()
        counts[cur_mask] = cur_count
    return labels, counts


def connected_components_cpu(input_tensor: torch.Tensor):
    out_shape = input_tensor.shape
    if input_tensor.dim() == 4 and input_tensor.shape[1] == 1:
        input_tensor = input_tensor.squeeze(1)
    else:
        assert (
            input_tensor.dim() == 3
        ), "Input tensor must be (B, H, W) or (B, 1, H, W)."

    batch_size = input_tensor.shape[0]
    labels_list = []
    counts_list = []
    for b in range(batch_size):
        labels, counts = connected_components_cpu_single(input_tensor[b])
        labels_list.append(labels)
        counts_list.append(counts)
    labels_tensor = torch.stack(labels_list, dim=0).to(input_tensor.device)
    counts_tensor = torch.stack(counts_list, dim=0).to(input_tensor.device)
    return labels_tensor.view(out_shape), counts_tensor.view(out_shape)


def connected_components(input_tensor: torch.Tensor):
    """
    Computes connected components labeling on a batch of 2D tensors, using the best available backend.

    Args:
        input_tensor (torch.Tensor): A BxHxW integer tensor or Bx1xHxW. Non-zero values are considered foreground. Bool tensor also accepted

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Both tensors have the same shape as input_tensor.
            - A tensor with dense labels. Background is 0.
            - A tensor with the size of the connected component for each pixel.
    """
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(1)

    assert (
        input_tensor.dim() == 4 and input_tensor.shape[1] == 1
    ), "Input tensor must be (B, H, W) or (B, 1, H, W)."

    if input_tensor.is_cuda:
        if HAS_CC_TORCH:
            return get_connected_components(input_tensor.to(torch.uint8))
        else:
            # Triton fallback (optional). On some platforms (e.g., Windows), Triton may
            # not be available; in that case, fall back to CPU.
            try:
                from sam3.perflib.triton.connected_components import (
                    connected_components_triton,
                )

                return connected_components_triton(input_tensor)
            except Exception as e:
                logging.warning(
                    "Triton connected-components backend is unavailable (%s). "
                    "Falling back to CPU connected components. For better performance, "
                    "install 'cc_torch' or use a platform that supports Triton.",
                    type(e).__name__,
                )
                labels_cpu, counts_cpu = connected_components_cpu(input_tensor.detach().cpu())
                return labels_cpu.to(input_tensor.device), counts_cpu.to(input_tensor.device)

    # CPU fallback
    return connected_components_cpu(input_tensor)
