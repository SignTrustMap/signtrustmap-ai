import argparse
import os
import re
import cv2
import numpy as np
from loguru import logger
from scipy.cluster.vq import kmeans2, whiten


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>| ]', "_", name)


def parse_dataset_yaml(yaml_path: str) -> tuple[str, dict[int, str], set[int]]:
    data_root = "data"
    names = {}
    blacklisted = set()

    with open(yaml_path, "r", encoding="utf-8") as f:
        in_names = False
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("path:") and not in_names:
                data_root = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            elif stripped.startswith("names:"):
                in_names = True
                continue
            elif in_names:
                if ":" in stripped:
                    parts = stripped.split(":", 1)
                    cls_key = parts[0].strip()
                    cls_val = parts[1].split("#")[0].strip()
                    try:
                        cls_id = int(cls_key)
                        names[cls_id] = cls_val
                        if "# blacklist" in line:
                            blacklisted.add(cls_id)
                    except ValueError:
                        pass

    return data_root, names, blacklisted


def get_dfg_prefix(name: str) -> str:
    if "-" in name:
        return name.split("-")[0]
    return "Other"


def get_mtsd_category(name: str) -> str:
    if "--" in name:
        return name.split("--")[0].capitalize()
    if "-" in name:
        return name.split("-")[0].capitalize()
    return "Other"


def compute_shape_features(crop_path: str) -> tuple[float, float, int, np.ndarray] | None:
    if not os.path.exists(crop_path):
        return None

    img = cv2.imread(crop_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh2 = 255 - thresh1

    cnts1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_c1 = max(cnts1, key=cv2.contourArea) if cnts1 else None
    max_c2 = max(cnts2, key=cv2.contourArea) if cnts2 else None

    area1 = cv2.contourArea(max_c1) if max_c1 is not None else 0
    area2 = cv2.contourArea(max_c2) if max_c2 is not None else 0

    best_cnt = max_c1 if area1 >= area2 else max_c2
    if best_cnt is None:
        return None

    area = cv2.contourArea(best_cnt)
    perimeter = cv2.arcLength(best_cnt, True)
    if perimeter == 0 or area < 10:
        return None

    circularity = 4.0 * np.pi * area / (perimeter**2 + 1e-6)
    circularity = min(1.0, circularity)

    _, _, w, h = cv2.boundingRect(best_cnt)
    aspect_ratio = float(w) / float(h + 1e-6)

    approx = cv2.approxPolyDP(best_cnt, 0.04 * perimeter, True)
    vertices = len(approx)

    moments = cv2.moments(best_cnt)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    return circularity, aspect_ratio, vertices, hu_log


def cluster_mtsd_shapes(
    classes: dict[int, str],
    crops_dir: str,
    n_clusters: int = 8,
    rng_seed: int = 42,
) -> dict[int, dict]:
    np.random.seed(rng_seed)
    features_list = []
    class_order = []
    metric_records = {}

    for cid, name in classes.items():
        clean_name = sanitize_filename(name)
        crop_paths = [
            os.path.join(crops_dir, "MTSD", f"{cid}_{clean_name}_sample_{i}.jpg")
            for i in range(1, 4)
        ]

        circs, ars, vtxs, hus = [], [], [], []
        for p in crop_paths:
            res = compute_shape_features(p)
            if res is not None:
                c, a, v, h = res
                circs.append(c)
                ars.append(a)
                vtxs.append(v)
                hus.append(h)

        if circs:
            med_circ = float(np.median(circs))
            med_ar = float(np.median(ars))
            med_vtx = int(np.median(vtxs))
            med_hu = np.median(hus, axis=0)
            std_circ = float(np.std(circs))
        else:
            med_circ = 0.5
            med_ar = 1.0
            med_vtx = 4
            med_hu = np.zeros(7)
            std_circ = 1.0

        vec = [
            med_circ,
            med_ar,
            min(med_vtx, 8) / 8.0,
            med_hu[0],
            med_hu[1],
            med_hu[2],
            med_hu[3],
        ]

        features_list.append(vec)
        class_order.append(cid)
        metric_records[cid] = {
            "circularity": med_circ,
            "aspect_ratio": med_ar,
            "vertices": med_vtx,
            "std_circ": std_circ,
        }

    feats = np.array(features_list, dtype=np.float32)
    feats_whitened = whiten(feats)

    actual_k = min(n_clusters, len(feats))
    centroids, labels = kmeans2(feats_whitened, actual_k, minit="points", seed=rng_seed)

    cluster_info = {}
    for cl_id in range(actual_k):
        c_indices = [i for i, l in enumerate(labels) if l == cl_id]
        if not c_indices:
            continue
        c_circ = np.mean([metric_records[class_order[i]]["circularity"] for i in c_indices])
        c_ar = np.mean([metric_records[class_order[i]]["aspect_ratio"] for i in c_indices])
        c_vtx = np.mean([metric_records[class_order[i]]["vertices"] for i in c_indices])

        if c_circ > 0.72:
            shape_name = "Circle / Round"
        elif c_ar > 1.35:
            shape_name = "Horizontal Rect"
        elif c_ar < 0.75:
            shape_name = "Vertical Rect"
        elif c_vtx == 3 or (c_circ < 0.55 and c_vtx <= 4 and 0.8 <= c_ar <= 1.2):
            shape_name = "Triangle / Diamond"
        elif c_vtx >= 6:
            shape_name = "Octagon / Hexagon"
        else:
            shape_name = "Square / Quad"

        cluster_info[cl_id] = {
            "name": f"Cluster {cl_id + 1}: {shape_name}",
            "avg_circ": c_circ,
            "avg_ar": c_ar,
        }

    result = {}
    for i, cid in enumerate(class_order):
        lbl = int(labels[i])
        result[cid] = {
            "cluster_id": lbl + 1,
            "cluster_name": cluster_info[lbl]["name"],
            "metrics": metric_records[cid],
        }

    return result


def generate_html_report(
    dataset_name: str,
    classes: dict[int, str],
    blacklisted: set[int],
    crops_dir: str,
    output_html_path: str,
    cluster_results: dict[int, dict] | None = None,
) -> None:
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

    rows_html = []
    group_options = set()

    for cid, name in sorted(classes.items()):
        clean_name = sanitize_filename(name)
        is_pre_blacklisted = cid in blacklisted

        if dataset_name == "DFG":
            group = get_dfg_prefix(name)
            sub_label = f"Prefix: {group}"
            metrics_display = ""
        else:
            group = get_mtsd_category(name)
            cl_data = cluster_results.get(cid, {}) if cluster_results else {}
            cluster_name = cl_data.get("cluster_name", "Cluster 1")
            sub_label = f"{group} | {cluster_name}"
            m = cl_data.get("metrics", {})
            metrics_display = (
                f"<span class='metric-badge'>Circ: {m.get('circularity', 0):.2f}</span>"
                f"<span class='metric-badge'>AR: {m.get('aspect_ratio', 0):.2f}</span>"
                f"<span class='metric-badge'>Vtx: {m.get('vertices', 0)}</span>"
            )

        group_options.add(group)

        sample_imgs_html = []
        for i in range(1, 4):
            rel_img = f"../sign_crops/{dataset_name}/{cid}_{clean_name}_sample_{i}.jpg"
            sample_imgs_html.append(
                f"<img src='{rel_img}' class='thumb' loading='lazy' alt='{name}' "
                f"onclick='previewImage(\"{rel_img}\", \"{name} ({cid})\")' />"
            )
        thumbs_block = "".join(sample_imgs_html)

        checked_attr = "checked" if is_pre_blacklisted else ""
        row_class = "row-blacklisted" if is_pre_blacklisted else "row-active"

        rows_html.append(
            f"""
            <tr class='{row_class}' data-group='{group}' data-cid='{cid}' data-name='{name}'>
                <td class='col-check'>
                    <input type='checkbox' class='sign-check' data-cid='{cid}' data-name='{name}' {checked_attr} onchange='updateStats()' />
                </td>
                <td class='col-id'><strong>{cid}</strong></td>
                <td class='col-name'>
                    <div class='name-text'>{name}</div>
                    <div class='sub-label'>{sub_label}</div>
                </td>
                <td class='col-thumbs'>{thumbs_block}</td>
                <td class='col-metrics'>{metrics_display}</td>
                <td class='col-status'>
                    <span class='status-pill {"pill-bl" if is_pre_blacklisted else "pill-ok"}'>
                        {"Blacklisted" if is_pre_blacklisted else "Keep"}
                    </span>
                </td>
            </tr>
            """
        )

    filter_options_html = "<option value='ALL'>All Groups / Categories</option>"
    for g in sorted(group_options):
        filter_options_html += f"<option value='{g}'>{g}</option>"

    total_count = len(classes)
    bl_count = len(set(classes.keys()) & blacklisted)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{dataset_name} Traffic Sign Blacklist Review</title>
<style>
:root {{
    --bg: #0f172a;
    --card: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #38bdf8;
    --danger: #ef4444;
    --success: #22c55e;
    --warning: #f59e0b;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    padding: 24px;
}}
.container {{
    max-width: 1400px;
    margin: 0 auto;
}}
header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
}}
h1 {{ font-size: 24px; font-weight: 700; color: var(--accent); }}
.stats-bar {{
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
}}
.stat-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 20px;
    flex: 1;
}}
.stat-card .val {{ font-size: 22px; font-weight: 700; }}
.stat-card .lbl {{ font-size: 13px; color: var(--text-muted); }}

.toolbar {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 20px;
    display: flex;
    gap: 16px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 20px;
}}
.toolbar input, .toolbar select {{
    background: #0b1120;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 14px;
    outline: none;
}}
.toolbar input:focus, .toolbar select:focus {{
    border-color: var(--accent);
}}
.search-box {{ flex: 1; min-width: 200px; }}
.btn {{
    padding: 8px 16px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
    border: none;
    transition: 0.2s;
}}
.btn-primary {{ background: var(--accent); color: #000; }}
.btn-primary:hover {{ opacity: 0.9; transform: translateY(-1px); }}
.btn-outline {{ background: transparent; border: 1px solid var(--border); color: var(--text); }}
.btn-outline:hover {{ background: var(--border); }}
.btn-danger {{ background: var(--danger); color: #fff; }}

table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--card);
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
}}
th, td {{
    padding: 12px 16px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}}
th {{
    background: #0b1120;
    color: var(--text-muted);
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
tr:hover {{ background: #24344d; }}
.row-blacklisted {{ background: rgba(239, 68, 68, 0.06); }}

.thumb {{
    width: 64px;
    height: 64px;
    object-fit: contain;
    background: #000;
    border-radius: 6px;
    margin-right: 8px;
    border: 1px solid var(--border);
    cursor: pointer;
    transition: transform 0.15s;
}}
.thumb:hover {{ transform: scale(1.15); border-color: var(--accent); z-index: 10; }}
.name-text {{ font-weight: 600; font-size: 15px; }}
.sub-label {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}
.metric-badge {{
    background: #0b1120;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    margin-right: 6px;
    border: 1px solid var(--border);
    color: var(--text-muted);
}}
.status-pill {{
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
}}
.pill-bl {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); }}
.pill-ok {{ background: rgba(34, 197, 94, 0.2); color: var(--success); }}

.sign-check {{
    width: 18px;
    height: 18px;
    cursor: pointer;
    accent-color: var(--danger);
}}

#modal {{
    display: none;
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.85);
    justify-content: center;
    align-items: center;
    z-index: 999;
}}
#modal img {{
    max-width: 80%;
    max-height: 80%;
    border-radius: 8px;
    box-shadow: 0 0 30px rgba(0,0,0,0.8);
    border: 2px solid var(--border);
}}
#modal-caption {{
    position: absolute;
    bottom: 30px;
    color: #fff;
    font-size: 16px;
    background: rgba(0,0,0,0.7);
    padding: 8px 16px;
    border-radius: 6px;
}}
.toast {{
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: var(--accent);
    color: #000;
    padding: 12px 24px;
    border-radius: 6px;
    font-weight: 700;
    display: none;
    z-index: 1000;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}}
</style>
</head>
<body>

<div class="container">
    <header>
        <div>
            <h1>{dataset_name} Traffic Sign Blacklist Review</h1>
            <p style="color: var(--text-muted); font-size: 14px; margin-top: 4px;">
                Review signs and check boxes to mark them for blacklisting (omitting from Vietnamese pretraining).
            </p>
        </div>
        <div style="display: flex; gap: 10px;">
            <button class="btn btn-outline" onclick="copyBlacklistToClipboard()">📋 Copy Blacklist</button>
            <button class="btn btn-primary" onclick="exportBlacklistTxt()">📥 Export Blacklist .txt</button>
        </div>
    </header>

    <div class="stats-bar">
        <div class="stat-card">
            <div class="val" id="stat-total">{total_count}</div>
            <div class="lbl">Total Classes</div>
        </div>
        <div class="stat-card">
            <div class="val" id="stat-blacklisted" style="color: var(--danger);">{bl_count}</div>
            <div class="lbl">Selected for Blacklist</div>
        </div>
        <div class="stat-card">
            <div class="val" id="stat-kept" style="color: var(--success);">{total_count - bl_count}</div>
            <div class="lbl">Kept for Training</div>
        </div>
    </div>

    <div class="toolbar">
        <input type="text" id="search-input" class="search-box" placeholder="Search class name or ID..." oninput="applyFilters()" />
        <select id="group-select" onchange="applyFilters()">
            {filter_options_html}
        </select>
        <button class="btn btn-outline" onclick="checkAllVisible(true)">Select All Visible</button>
        <button class="btn btn-outline" onclick="checkAllVisible(false)">Deselect All Visible</button>
    </div>

    <table>
        <thead>
            <tr>
                <th style="width: 40px;">BL</th>
                <th style="width: 60px;">ID</th>
                <th style="width: 320px;">Name & Group</th>
                <th>Sample Crops</th>
                <th style="width: 220px;">Shape Metrics</th>
                <th style="width: 120px;">Status</th>
            </tr>
        </thead>
        <tbody id="sign-table-body">
            {"".join(rows_html)}
        </tbody>
    </table>
</div>

<div id="modal" onclick="closeModal()">
    <img id="modal-img" src="" alt="preview" />
    <div id="modal-caption"></div>
</div>

<div id="toast" class="toast">Copied to clipboard!</div>

<script>
function previewImage(src, caption) {{
    const modal = document.getElementById('modal');
    document.getElementById('modal-img').src = src;
    document.getElementById('modal-caption').innerText = caption;
    modal.style.display = 'flex';
}}

function closeModal() {{
    document.getElementById('modal').style.display = 'none';
}}

function applyFilters() {{
    const search = document.getElementById('search-input').value.toLowerCase();
    const group = document.getElementById('group-select').value;
    const rows = document.querySelectorAll('#sign-table-body tr');

    rows.forEach(r => {{
        const rGroup = r.getAttribute('data-group');
        const rCid = r.getAttribute('data-cid');
        const rName = r.getAttribute('data-name').toLowerCase();

        const matchSearch = !search || rCid.includes(search) || rName.includes(search);
        const matchGroup = (group === 'ALL') || (rGroup === group);

        if (matchSearch && matchGroup) {{
            r.style.display = '';
        }} else {{
            r.style.display = 'none';
        }}
    }});
}}

function saveToLocalStorage() {{
    const checkboxes = document.querySelectorAll('.sign-check');
    const checkedIds = [];
    checkboxes.forEach(c => {{
        if (c.checked) {{
            checkedIds.push(c.getAttribute('data-cid'));
        }}
    }});
    localStorage.setItem('blacklist_{dataset_name}', JSON.stringify(checkedIds));
}}

function loadFromLocalStorage() {{
    const saved = localStorage.getItem('blacklist_{dataset_name}');
    if (saved) {{
        try {{
            const checkedIds = new Set(JSON.parse(saved));
            const checkboxes = document.querySelectorAll('.sign-check');
            checkboxes.forEach(c => {{
                const cid = c.getAttribute('data-cid');
                c.checked = checkedIds.has(cid);
            }});
        }} catch (e) {{
            console.error('Error loading saved state:', e);
        }}
    }}
}}

function resetToDefault() {{
    if (confirm('Reset selection to default states from all.yaml?')) {{
        localStorage.removeItem('blacklist_{dataset_name}');
        location.reload();
    }}
}}

function checkAllVisible(state) {{
    const rows = document.querySelectorAll('#sign-table-body tr');
    rows.forEach(r => {{
        if (r.style.display !== 'none') {{
            const chk = r.querySelector('.sign-check');
            if (chk) chk.checked = state;
        }}
    }});
    updateStats();
}}

function updateStats() {{
    const checkboxes = document.querySelectorAll('.sign-check');
    let bl = 0;
    checkboxes.forEach(c => {{
        const row = c.closest('tr');
        const statusPill = row.querySelector('.status-pill');
        if (c.checked) {{
            bl++;
            row.classList.add('row-blacklisted');
            row.classList.remove('row-active');
            statusPill.innerText = 'Blacklisted';
            statusPill.className = 'status-pill pill-bl';
        }} else {{
            row.classList.remove('row-blacklisted');
            row.classList.add('row-active');
            statusPill.innerText = 'Keep';
            statusPill.className = 'status-pill pill-ok';
        }}
    }});
    const total = checkboxes.length;
    document.getElementById('stat-blacklisted').innerText = bl;
    document.getElementById('stat-kept').innerText = (total - bl);
    saveToLocalStorage();
}}

function getBlacklistLines() {{
    const checkboxes = document.querySelectorAll('.sign-check');
    const lines = [];
    checkboxes.forEach(c => {{
        if (c.checked) {{
            const cid = c.getAttribute('data-cid');
            const name = c.getAttribute('data-name');
            lines.push(cid + ': ' + name);
        }}
    }});
    return lines;
}}

function exportBlacklistTxt() {{
    const lines = getBlacklistLines();
    const blob = new Blob([lines.join('\\n')], {{ type: 'text/plain' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = '{dataset_name}_blacklist.txt';
    a.click();
    URL.revokeObjectURL(url);
    showToast('Downloaded {dataset_name}_blacklist.txt (' + lines.length + ' classes)');
}}

function copyBlacklistToClipboard() {{
    const lines = getBlacklistLines();
    navigator.clipboard.writeText(lines.join('\\n')).then(() => {{
        showToast('Copied ' + lines.length + ' blacklist classes to clipboard!');
    }});
}}

function showToast(msg) {{
    const toast = document.getElementById('toast');
    toast.innerText = msg;
    toast.style.display = 'block';
    setTimeout(() => {{ toast.style.display = 'none'; }}, 2500);
}}

window.addEventListener('DOMContentLoaded', () => {{
    loadFromLocalStorage();
    updateStats();
}});
</script>

</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Generated review report: {output_html_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", type=str, default="configs/training/all.yaml")
    parser.add_argument("--crops-dir", type=str, default="runs/sign_crops")
    parser.add_argument("--output-dir", type=str, default="runs/sign_review")
    parser.add_argument("--n-clusters", type=int, default=8)
    args = parser.parse_args()

    _, names, blacklisted = parse_dataset_yaml(args.data_yaml)

    dfg_classes = {cid: name for cid, name in names.items() if 0 <= cid < 200}
    mtsd_classes = {cid: name for cid, name in names.items() if 302 <= cid <= 702}

    logger.info(f"Loaded {len(dfg_classes)} DFG classes and {len(mtsd_classes)} MTSD classes.")

    dfg_report_path = os.path.join(args.output_dir, "report_DFG.html")
    generate_html_report(
        dataset_name="DFG",
        classes=dfg_classes,
        blacklisted=blacklisted,
        crops_dir=args.crops_dir,
        output_html_path=dfg_report_path,
    )

    logger.info("Computing shape features and clustering MTSD traffic signs...")
    mtsd_clusters = cluster_mtsd_shapes(
        classes=mtsd_classes,
        crops_dir=args.crops_dir,
        n_clusters=args.n_clusters,
    )

    mtsd_report_path = os.path.join(args.output_dir, "report_MTSD.html")
    generate_html_report(
        dataset_name="MTSD",
        classes=mtsd_classes,
        blacklisted=blacklisted,
        crops_dir=args.crops_dir,
        output_html_path=mtsd_report_path,
        cluster_results=mtsd_clusters,
    )

    logger.info("Blacklist review reports generated successfully.")


if __name__ == "__main__":
    main()
