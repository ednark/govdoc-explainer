"""Convert fp32 embedding.json artifacts to int8-quantized versions.

Experiment tool — does not modify the main build pipeline. Reads the
existing fp32 artifacts and writes int8-quantized siblings:

    assets/embedding.json        -> assets/embedding_q8.json
    sources/<doc>/embedding.json -> sources/<doc>/embedding_q8.json

Quantization is symmetric per-vector int8:
    scale = max(|v|) / 127
    q     = round(v / scale) clipped to [-127, 127]

The client reconstructs v ~= q * scale. Because cosine similarity is
scale-invariant per vector, ranking can be done entirely on the int8
values (scales cancel) — see semantic_search_v2.js.

Usage:
    uv run python experiments/search-v2/quantize_embeddings.py [--verify]
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

from govdoc_explainer.embeddings import write_q8_file


def find_embedding_files(root: Path):
    targets = [root / "assets" / "embedding.json"]
    targets.extend(sorted((root / "sources").glob("*/embedding.json")))
    return [p for p in targets if p.exists()]


def load_entries(path):
    with open(path, "r") as f:
        return json.load(f)


def quantize_file(src_path: Path, dst_path: Path):
    entries = load_entries(src_path)
    write_q8_file(entries, dst_path)
    return src_path.stat().st_size, dst_path.stat().st_size, len(entries)


def dequantize(b64, scale):
    q = np.frombuffer(base64.b64decode(b64), dtype=np.int8).astype(np.float64)
    return q * scale


def fp32_vector(entry):
    arr = np.asarray(entry["embedding"], dtype=np.float64)
    return arr.reshape(-1) if arr.ndim == 2 else arr


def top8_overlap(matrix_a, matrix_b):
    """Mean top-8 neighbour-list agreement between two n x n similarity matrices."""
    n = matrix_a.shape[0]
    if n <= 9:  # top-8 excluding self is degenerate below 10 items
        return 1.0
    np.fill_diagonal(matrix_a, -np.inf)
    np.fill_diagonal(matrix_b, -np.inf)
    top_a = np.argsort(-matrix_a, axis=1)[:, :8]
    top_b = np.argsort(-matrix_b, axis=1)[:, :8]
    overlaps = [len(set(row_a) & set(row_b)) / 8 for row_a, row_b in zip(top_a.tolist(), top_b.tolist())]
    return sum(overlaps) / len(overlaps)


def unit_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # zero vectors (docs with no embeddings) -> stay zero
    return matrix / norms


def verify_file(src_path: Path, dst_path: Path):
    """Report cosine deviation and top-8 ranking agreement fp32 vs q8 (vectorized)."""
    fp32_entries = load_entries(src_path)
    q8_entries = load_entries(dst_path)["entries"]
    fp32_mat = np.stack([fp32_vector(e) for e in fp32_entries])
    q8_mat = np.stack([dequantize(e["embedding"], e["scale"]) for e in q8_entries])

    # Per-vector self-consistency: fp32 vector vs its own dequantized copy.
    with np.errstate(invalid="ignore", divide="ignore"):
        dev = 1.0 - np.einsum("ij,ij->i", fp32_mat, q8_mat) / (
            np.linalg.norm(fp32_mat, axis=1) * np.linalg.norm(q8_mat, axis=1)
        )

    # Top-8 neighbour agreement: rank each fp32 vector (rows) against the
    # fp32 database vs the quantized database (columns).
    fp32_unit = unit_rows(fp32_mat)
    q8_unit = unit_rows(q8_mat)
    sim_fp32 = fp32_unit @ fp32_unit.T
    sim_q8 = fp32_unit @ q8_unit.T
    return {
        "max_cos_dev": float(np.nanmax(np.abs(dev))),
        "mean_cos_dev": float(np.nanmean(np.abs(dev))),
        "mean_top8_overlap": top8_overlap(sim_fp32, sim_q8),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repo root (default: cwd)")
    parser.add_argument("--verify", action="store_true", help="also report quantization quality stats")
    args = parser.parse_args()

    files = find_embedding_files(args.root)
    if not files:
        sys.exit("No embedding.json files found — run the build first.")

    total_src = total_dst = 0
    print(f"{'file':<70} {'fp32':>10} {'int8':>10} {'ratio':>7}")
    for src in files:
        dst = src.with_name("embedding_q8.json")
        src_size, dst_size, n = quantize_file(src, dst)
        total_src += src_size
        total_dst += dst_size
        rel = str(src.relative_to(args.root))
        print(f"{rel:<70} {src_size / 1e6:>7.2f}MB {dst_size / 1e6:>7.2f}MB {dst_size / src_size:>5.1f}x ({n}v)")

        if args.verify:
            stats = verify_file(src, dst)
            print(
                f"  verify: mean cos dev {stats['mean_cos_dev']:.2e}, "
                f"max {stats['max_cos_dev']:.2e}, top-8 neighbour overlap "
                f"{stats['mean_top8_overlap'] * 100:.1f}%"
            )

    print(f"\nTotal: {total_src / 1e6:.1f} MB -> {total_dst / 1e6:.1f} MB ({total_dst / total_src:.1f}x)")
    print("Quantized copies are experiment artifacts; delete them to revert.")


if __name__ == "__main__":
    main()
