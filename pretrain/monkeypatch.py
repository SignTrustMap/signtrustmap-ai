import os
import torch
import ultralytics.optim.muon
from loguru import logger
from ultralytics.optim.muon import zeropower_via_newtonschulz5
from ultralytics.data.base import BaseDataset


def _patched_muon_update(
    grad: torch.Tensor | list[torch.Tensor],
    momentum: torch.Tensor | list[torch.Tensor],
    beta: float = 0.95,
    nesterov: bool = True,
) -> torch.Tensor | list[torch.Tensor]:
    single = isinstance(grad, torch.Tensor)
    grads, momentums = ([grad], [momentum]) if single else (grad, momentum)
    torch._foreach_mul_(momentums, beta)
    torch._foreach_add_(momentums, grads, alpha=1 - beta)
    if nesterov:
        updates = list(torch._foreach_mul(momentums, beta))
        torch._foreach_add_(updates, grads, alpha=1 - beta)
    else:
        updates = list(momentums)
    buckets = {}
    for i, u in enumerate(updates):
        m = u.reshape(len(u), -1) if u.ndim > 2 else u
        transpose = m.size(0) > m.size(1)
        if transpose:
            m = m.transpose(0, 1)
        scale = max(1, grads[i].size(-2) / grads[i].size(-1)) ** 0.5
        buckets.setdefault((m.size(0), scale, m.device, m.dtype), []).append((i, m, transpose))
    for (_, scale, _, _), items in buckets.items():
        n = max(m.size(1) for _, m, _ in items)
        X = torch.stack([torch.nn.functional.pad(m, (0, n - m.size(1))) for _, m, _ in items])
        X = zeropower_via_newtonschulz5(X).to(grads[items[0][0]].dtype).mul_(scale)
        for j, (i, m, transpose) in enumerate(items):
            x = X[j, :, : m.size(1)]
            updates[i] = (x.T if transpose else x).reshape(grads[i].shape)
    return updates[0] if single else updates


ultralytics.optim.muon.muon_update = _patched_muon_update
logger.info("Patched ultralytics.optim.muon.muon_update with reshape support for non-contiguous tensors.")


_orig_update_labels = BaseDataset.update_labels

def _get_dataset_root_and_name(p: str, data_root: str) -> tuple[str, str]:
    p_abs = os.path.abspath(p if os.path.isabs(p) else os.path.join(data_root, p))
    root_dir = p_abs
    while root_dir:
        base = os.path.basename(root_dir)
        if base in ("images", "labels"):
            root_dir = os.path.dirname(root_dir)
            break
        parent = os.path.dirname(root_dir)
        if parent == root_dir:
            break
        root_dir = parent
    return root_dir, os.path.basename(root_dir)

def _patched_update_labels(self, include_class: list[int] | None) -> None:
    yaml_file = self.data.get("yaml_file")
    if yaml_file and yaml_file.endswith("all.yaml"):
        train_path = self.data.get("train")
        if isinstance(train_path, list) and len(train_path) > 1:
            data_root = self.data.get("path", "")
            dataset_names = [_get_dataset_root_and_name(p, data_root)[1] for p in train_path]

            max_cls_per_dataset = {}
            for label in self.labels:
                _, dname = _get_dataset_root_and_name(label["im_file"], data_root)
                cls_array = label["cls"]
                if cls_array.size > 0:
                    max_cls = int(cls_array.max())
                    if dname not in max_cls_per_dataset or max_cls > max_cls_per_dataset[dname]:
                        max_cls_per_dataset[dname] = max_cls

            offsets = {}
            current_offset = 0
            for dname in dataset_names:
                offsets[dname] = current_offset
                nc = max_cls_per_dataset.get(dname, -1) + 1
                current_offset += nc

            sep = os.sep
            active_offsets = [
                (f"{sep}{dname}{sep}", f"{dname}{sep}", offset)
                for dname, offset in offsets.items()
                if offset > 0
            ]

            for label in self.labels:
                im_file = label["im_file"]
                for ind1, ind2, offset in active_offsets:
                    if ind1 in im_file or im_file.startswith(ind2):
                        label["cls"] += offset
                        break

    _orig_update_labels(self, include_class)

BaseDataset.update_labels = _patched_update_labels
logger.info("Patched BaseDataset.update_labels to dynamically offset dataset classes on RAM.")
