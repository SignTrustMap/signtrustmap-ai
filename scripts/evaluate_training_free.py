import argparse
import time
from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from loguru import logger


def evaluate_predictions(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float, float]:
    preds = logits.argmax(dim=-1)
    correct_top1 = (preds == targets).float()
    top1_acc = correct_top1.mean().item() * 100.0

    _, top5_indices = logits.topk(min(5, logits.shape[-1]), dim=-1)
    targets_expanded = targets.unsqueeze(-1).expand_as(top5_indices)
    top5_acc = (top5_indices == targets_expanded).any(dim=-1).float().mean().item() * 100.0

    num_classes = logits.shape[-1]
    f1_list = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        f1_list.append(f1.item())
    macro_f1 = float(sum(f1_list) / max(1, len(f1_list)))

    mask_c20 = targets == 20
    c20_acc = (preds[mask_c20] == 20).float().mean().item() * 100.0 if mask_c20.any() else 0.0

    return top1_acc, top5_acc, macro_f1, c20_acc


def run_evaluation():
    parser = argparse.ArgumentParser(description="Unified Training-Free Benchmark for Traffic Sign Recognition")
    parser.add_argument("--split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--method", choices=["all", "zeroshot", "protofuse", "tip", "gda", "timo"], default="all")
    parser.add_argument("--alpha", type=float, default=250.0)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    base_dir = Path(__file__).resolve().parent.parent

    support_cache_file = base_dir / "outputs" / "evaluation" / "tip_cache.pt"
    text_emb_file = base_dir / "weights" / "embeddings" / "text_embeddings_mobileclip_s2.pt"
    val_file = base_dir / "outputs" / "evaluation" / "features_val.pt"
    test_file = base_dir / "outputs" / "evaluation" / "features_test.pt"

    if not support_cache_file.exists():
        raise FileNotFoundError(f"Missing support cache: {support_cache_file}")
    if not text_emb_file.exists():
        raise FileNotFoundError(f"Missing text embeddings: {text_emb_file}")

    logger.info(f"Loading support keys and text prototypes on {device}...")
    supp_data = torch.load(support_cache_file, map_location=device)
    keys_raw = supp_data["keys"].float()
    labels_v = supp_data["labels"].long()
    keys_norm = keys_raw / torch.clamp(keys_raw.norm(dim=-1, keepdim=True), min=1e-8)

    text_emb = torch.load(text_emb_file, map_location=device).float()
    text_norm = text_emb / torch.clamp(text_emb.norm(dim=-1, keepdim=True), min=1e-8)
    clip_weights = text_norm.t()
    num_classes = text_norm.shape[0]

    eval_splits = []
    if args.split in ["val", "both"] and val_file.exists():
        val_data = torch.load(val_file, map_location=device)
        vf = val_data["features"].float()
        vf = vf / torch.clamp(vf.norm(dim=-1, keepdim=True), min=1e-8)
        eval_splits.append(("Validation", vf, val_data["labels"].long()))
    if args.split in ["test", "both"] and test_file.exists():
        test_data = torch.load(test_file, map_location=device)
        tf = test_data["features"].float()
        tf = tf / torch.clamp(tf.norm(dim=-1, keepdim=True), min=1e-8)
        eval_splits.append(("Test", tf, test_data["labels"].long()))

    if not eval_splits:
        raise FileNotFoundError("No evaluation split files found.")

    methods_to_run = (
        ["zeroshot", "protofuse", "tip", "gda", "timo"]
        if args.method == "all"
        else [args.method]
    )

    for split_name, query_f, targets in eval_splits:
        print(f"\n==================== Evaluation on {split_name} Split ({len(targets):,} crops) ====================")
        results = []

        if "zeroshot" in methods_to_run:
            t0 = time.time()
            zs_logits = 100.0 * query_f @ clip_weights
            t_ms = (time.time() - t0) / len(targets) * 1000.0
            t1, t5, f1, c20 = evaluate_predictions(zs_logits, targets)
            results.append(("Zero-Shot CLIP", t1, t5, f1, c20, t_ms))

        if "protofuse" in methods_to_run:
            proto_list = []
            for c in range(num_classes):
                mask = labels_v == c
                if mask.any():
                    v_mean = keys_norm[mask].mean(dim=0)
                    v_norm = v_mean / torch.clamp(v_mean.norm(), min=1e-8)
                    fused = 0.4 * v_norm + 0.6 * text_norm[c]
                    proto_list.append(fused / torch.clamp(fused.norm(), min=1e-8))
                else:
                    proto_list.append(text_norm[c])
            proto_mat = torch.stack(proto_list, dim=0)

            t0 = time.time()
            pf_logits = 100.0 * query_f @ proto_mat.t()
            t_ms = (time.time() - t0) / len(targets) * 1000.0
            t1, t5, f1, c20 = evaluate_predictions(pf_logits, targets)
            results.append(("ProtoFuse (a=0.4)", t1, t5, f1, c20, t_ms))

        if "tip" in methods_to_run:
            one_hot_labels = F.one_hot(labels_v, num_classes=num_classes).float()
            t0 = time.time()
            sim_v = query_f @ keys_norm.t()
            affinity = torch.exp(-1.0 * (1.0 - sim_v))
            tip_logits = 100.0 * (query_f @ clip_weights) + 1.0 * (affinity @ one_hot_labels)
            t_ms = (time.time() - t0) / len(targets) * 1000.0
            t1, t5, f1, c20 = evaluate_predictions(tip_logits, targets)
            results.append(("Tip-Adapter (Free)", t1, t5, f1, c20, t_ms))

        if "gda" in methods_to_run:
            t0 = time.time()
            mus = torch.stack([
                keys_norm[labels_v == i].mean(dim=0) if (labels_v == i).any() else text_norm[i]
                for i in range(num_classes)
            ])
            mus = mus / torch.clamp(mus.norm(dim=-1, keepdim=True), min=1e-8)
            center_v = torch.cat([
                keys_norm[labels_v == i] - mus[i].unsqueeze(0)
                for i in range(num_classes) if (labels_v == i).any()
            ], dim=0)
            cov = center_v.T.cov()
            d = center_v.shape[1]
            reg = cov.trace() * torch.eye(d, device=device, dtype=cov.dtype)
            mat_reg = (center_v.shape[0] - 1) * cov + reg
            try:
                cov_inv = d * torch.linalg.pinv(mat_reg)
            except Exception:
                cov_inv = d * torch.linalg.pinv(mat_reg.cpu()).to(device=device, dtype=cov.dtype)

            ps = torch.ones(num_classes, device=device) / float(num_classes)
            w_gda = mus @ cov_inv
            b_gda = ps.log() - 0.5 * torch.einsum("nd, dc, nc -> n", mus, cov_inv, mus)
            gda_logits = 250.0 * (query_f @ clip_weights) + (query_f @ w_gda.t() + b_gda)
            t_ms = (time.time() - t0) / len(targets) * 1000.0
            t1, t5, f1, c20 = evaluate_predictions(gda_logits, targets)
            results.append(("GDA-CLIP", t1, t5, f1, c20, t_ms))

        if "timo" in methods_to_run:
            t0 = time.time()
            labels_t = torch.arange(num_classes, device=device)
            vecs_c = torch.cat([keys_norm, args.beta * text_norm], dim=0)
            labels_c = torch.cat([labels_v, labels_t], dim=0)

            mus = torch.stack([vecs_c[labels_c == i].mean(dim=0) for i in range(num_classes)])
            mus = mus / torch.clamp(mus.norm(dim=-1, keepdim=True), min=1e-8)

            center_v = torch.cat([vecs_c[labels_c == i] - mus[i].unsqueeze(0) for i in range(num_classes)], dim=0)
            cov = center_v.T.cov()
            d = center_v.shape[1]
            reg = cov.trace() * torch.eye(d, device=device, dtype=cov.dtype)
            mat_reg = (center_v.shape[0] - 1) * cov + reg
            try:
                cov_inv = d * torch.linalg.pinv(mat_reg)
            except Exception:
                cov_inv = d * torch.linalg.pinv(mat_reg.cpu()).to(device=device, dtype=cov.dtype)

            ps = torch.ones(num_classes, device=device) / float(num_classes)
            w_gda = mus @ cov_inv
            b_timo = ps.log() - 0.5 * torch.einsum("nd, dc, nc -> n", mus, cov_inv, mus)
            w_timo = args.alpha * clip_weights + w_gda.t()

            t_fit = time.time() - t0
            logger.info(f"TIMO Collapsed Matrix built in {t_fit * 1000.0:.1f} ms")

            t_inf0 = time.time()
            timo_logits = query_f @ w_timo + b_timo
            t_ms = (time.time() - t_inf0) / len(targets) * 1000.0
            t1, t5, f1, c20 = evaluate_predictions(timo_logits, targets)
            results.append(("TIMO (Collapsed)", t1, t5, f1, c20, t_ms))

        print(f"\n| {'Method':<20} | {'Top-1 (%)':<10} | {'Top-5 (%)':<10} | {'Macro F1':<10} | {'Class 020 (%)':<14} | {'Latency (ms)':<12} |")
        print(f"|{'-'*22}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*16}|{'-'*14}|")
        for name, t1, t5, f1, c20, lat in results:
            print(f"| {name:<20} | {t1:>9.2f}% | {t5:>9.2f}% | {f1:>10.4f} | {c20:>13.2f}% | {lat:>10.3f} ms |")
        print("="*85)


if __name__ == "__main__":
    run_evaluation()
