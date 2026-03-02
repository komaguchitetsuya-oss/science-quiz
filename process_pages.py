"""
PDF問題集のオレンジ文字を検出し、マスク画像と座標データを生成するスクリプト。

出力:
- pages/page{N}_masked.png : オレンジ部分を白で隠した画像
- pages/regions.json       : 各ページのオレンジ領域座標
"""

import json
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF

PAGES_DIR = os.path.join(os.path.dirname(__file__), "pages")
PDF_PATH = os.path.expanduser("~/Downloads/20260302173326_001.pdf")

# オレンジ検出パラメータ
BLOCK_SIZE = 5         # ピクセルをブロック単位でグループ化（小さい＝細かく分離）
DILATE_BLOCKS = 2      # 隣接ブロックをマージするための膨張量（同一単語内の文字をつなげる）
PADDING = 3            # 領域のパディング（ピクセル）
MIN_REGION_SIZE = 12   # これより小さい領域はノイズとしてスキップ
LINE_Y_TOLERANCE = 20  # この範囲内の領域は「同じ行」として左→右でソート


def detect_orange_mask(img_array):
    """オレンジ色のピクセルを検出して boolean mask を返す"""
    r = img_array[:, :, 0].astype(np.int16)
    g = img_array[:, :, 1].astype(np.int16)
    b = img_array[:, :, 2].astype(np.int16)

    # オレンジ判定:
    #   R が高い、R-B の差が大きい、彩度がある、R > G
    mask = (
        (r > 170) &
        ((r - b) > 35) &
        ((r - g) > 10) &
        ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 35)
    )
    return mask


def find_regions(orange_mask, block_size=BLOCK_SIZE, dilate=DILATE_BLOCKS, page_width=None):
    """オレンジマスクからブロック単位で連結領域を見つける"""
    h, w = orange_mask.shape

    # ブロック単位に集約
    bh = (h + block_size - 1) // block_size
    bw = (w + block_size - 1) // block_size
    block_grid = np.zeros((bh, bw), dtype=bool)

    for by in range(bh):
        for bx in range(bw):
            y0 = by * block_size
            y1 = min(y0 + block_size, h)
            x0 = bx * block_size
            x1 = min(x0 + block_size, w)
            if orange_mask[y0:y1, x0:x1].any():
                block_grid[by, bx] = True

    # 膨張（水平方向のみ — 同一行内の文字をつなげるが、別の行はマージしない）
    if dilate > 0:
        dilated = block_grid.copy()
        for _ in range(dilate):
            new = dilated.copy()
            for by in range(bh):
                for bx in range(bw):
                    if dilated[by, bx]:
                        for dx in range(-1, 2):
                            nx = bx + dx
                            if 0 <= nx < bw:
                                new[by, nx] = True
            dilated = new
        block_grid = dilated

    # 連結成分検出（4方向フラッドフィル — 文字内の上下は接続する）
    labels = np.zeros((bh, bw), dtype=int)
    label_id = 0

    for by in range(bh):
        for bx in range(bw):
            if block_grid[by, bx] and labels[by, bx] == 0:
                label_id += 1
                stack = [(by, bx)]
                while stack:
                    cy, cx = stack.pop()
                    if (0 <= cy < bh and 0 <= cx < bw and
                            block_grid[cy, cx] and labels[cy, cx] == 0):
                        labels[cy, cx] = label_id
                        stack.extend([(cy-1, cx), (cy+1, cx),
                                      (cy, cx-1), (cy, cx+1)])

    # 各ラベルの境界ボックスを計算（元のピクセル座標で）
    regions = []
    for lid in range(1, label_id + 1):
        ys, xs = np.where(labels == lid)
        # ブロック座標 → ピクセル座標
        y_min = int(ys.min()) * block_size
        y_max = min(int(ys.max() + 1) * block_size, orange_mask.shape[0])
        x_min = int(xs.min()) * block_size
        x_max = min(int(xs.max() + 1) * block_size, orange_mask.shape[1])

        # 実際のオレンジピクセルでバウンディングボックスを絞る
        region_mask = orange_mask[y_min:y_max, x_min:x_max]
        if not region_mask.any():
            continue
        oy, ox = np.where(region_mask)
        ry_min = y_min + int(oy.min()) - PADDING
        ry_max = y_min + int(oy.max()) + PADDING
        rx_min = x_min + int(ox.min()) - PADDING
        rx_max = x_min + int(ox.max()) + PADDING

        # クリップ
        ry_min = max(0, ry_min)
        ry_max = min(orange_mask.shape[0], ry_max)
        rx_min = max(0, rx_min)
        rx_max = min(orange_mask.shape[1], rx_max)

        # 小さすぎる領域はスキップ（ノイズ）
        if (rx_max - rx_min) < MIN_REGION_SIZE or (ry_max - ry_min) < MIN_REGION_SIZE:
            continue

        regions.append({
            "x": rx_min,
            "y": ry_min,
            "w": rx_max - rx_min,
            "h": ry_max - ry_min,
        })

    # 読み順: 左半分を上→下、次に右半分を上→下（同じ行内は左→右）
    mid_x = (page_width or orange_mask.shape[1]) // 2
    regions.sort(key=lambda r: (
        0 if r["x"] + r["w"] / 2 < mid_x else 1,
        r["y"] // LINE_Y_TOLERANCE,
        r["x"]
    ))

    return regions


def dilate_mask(mask, radius=2):
    """マスクを数ピクセル膨張させてアンチエイリアスのエッジも消す"""
    h, w = mask.shape
    dilated = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(mask)
            sy = max(0, dy)
            ey = min(h, h + dy)
            sx = max(0, dx)
            ex = min(w, w + dx)
            shifted[sy:ey, sx:ex] = mask[sy - dy:ey - dy, sx - dx:ex - dx]
            dilated |= shifted
    return dilated


def create_masked_image(img_array, orange_mask):
    """オレンジピクセル＋周辺エッジを白で隠した画像を作成"""
    expanded = dilate_mask(orange_mask, radius=2)
    masked = img_array.copy()
    masked[expanded] = 255
    return masked


def enhance_image(img):
    """スキャン画像のコントラストとシャープネスを向上させて可読性を上げる"""
    # コントラスト強化（黒文字をより濃く、背景をより白く）
    img = ImageEnhance.Contrast(img).enhance(1.5)
    # シャープネス向上（文字の輪郭をくっきり）
    img = ImageEnhance.Sharpness(img).enhance(1.8)
    return img


def find_gray_labels(img_array, block_size=10, min_blocks=30):
    """灰色背景の吹き出し領域をラベリング（大きな灰色領域のみ）"""
    h, w = img_array.shape[:2]
    r = img_array[:, :, 0].astype(np.float32)
    g = img_array[:, :, 1].astype(np.float32)
    b = img_array[:, :, 2].astype(np.float32)
    avg = (r + g + b) / 3.0
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

    # 灰色判定: 中間輝度 & 低彩度（白や黒ではない灰色背景）
    gray_mask = (avg > 160) & (avg < 235) & (sat < 30)

    bh = (h + block_size - 1) // block_size
    bw = (w + block_size - 1) // block_size
    block_grid = np.zeros((bh, bw), dtype=bool)

    for by in range(bh):
        for bx in range(bw):
            y0 = by * block_size
            y1 = min(y0 + block_size, h)
            x0 = bx * block_size
            x1 = min(x0 + block_size, w)
            block_area = (y1 - y0) * (x1 - x0)
            if block_area > 0 and gray_mask[y0:y1, x0:x1].sum() > block_area * 0.4:
                block_grid[by, bx] = True

    # 連結成分のラベリング
    labels = np.zeros((bh, bw), dtype=int)
    label_id = 0
    label_counts = {}

    for by in range(bh):
        for bx in range(bw):
            if block_grid[by, bx] and labels[by, bx] == 0:
                label_id += 1
                count = 0
                stack = [(by, bx)]
                while stack:
                    cy, cx = stack.pop()
                    if (0 <= cy < bh and 0 <= cx < bw and
                            block_grid[cy, cx] and labels[cy, cx] == 0):
                        labels[cy, cx] = label_id
                        count += 1
                        stack.extend([(cy-1, cx), (cy+1, cx),
                                      (cy, cx-1), (cy, cx+1)])
                label_counts[label_id] = count

    # 小さい灰色領域はノイズとして除外
    for lid, count in label_counts.items():
        if count < min_blocks:
            labels[labels == lid] = 0

    return labels, block_size


def merge_regions_in_bubbles(regions, gray_labels, block_size):
    """灰色背景の吹き出し内にある複数の領域を1つにマージ"""
    bh, bw = gray_labels.shape
    groups = {}
    ungrouped = []

    for i, r in enumerate(regions):
        cx = r["x"] + r["w"] // 2
        cy = r["y"] + r["h"] // 2
        by = min(cy // block_size, bh - 1)
        bx = min(cx // block_size, bw - 1)
        label = int(gray_labels[by, bx])
        if label > 0:
            groups.setdefault(label, []).append(i)
        else:
            ungrouped.append(i)

    merged = [regions[i] for i in ungrouped]

    for label, indices in groups.items():
        if len(indices) == 1:
            merged.append(regions[indices[0]])
        else:
            group = [regions[i] for i in indices]
            x_min = min(r["x"] for r in group)
            y_min = min(r["y"] for r in group)
            x_max = max(r["x"] + r["w"] for r in group)
            y_max = max(r["y"] + r["h"] for r in group)
            merged.append({
                "x": x_min, "y": y_min,
                "w": x_max - x_min, "h": y_max - y_min,
            })

    return merged


def auto_crop(img_array, margin=10, threshold=240):
    """白い余白を検出してトリミング。marginピクセルの余白を残す"""
    # グレースケール化して白でない領域を検出
    gray = np.mean(img_array[:, :, :3], axis=2)
    non_white = gray < threshold

    # 行・列ごとの非白ピクセル数
    row_has_content = non_white.any(axis=1)
    col_has_content = non_white.any(axis=0)

    if not row_has_content.any() or not col_has_content.any():
        return img_array, 0, 0  # 全部白ならトリミングしない

    rows = np.where(row_has_content)[0]
    cols = np.where(col_has_content)[0]

    y_min = max(0, rows[0] - margin)
    y_max = min(img_array.shape[0], rows[-1] + margin)
    x_min = max(0, cols[0] - margin)
    x_max = min(img_array.shape[1], cols[-1] + margin)

    return img_array[y_min:y_max, x_min:x_max], x_min, y_min


def auto_rotate(img):
    """スキャンPDFを日本語横書きとして読めるよう90°時計回りに回転"""
    return img.transpose(Image.ROTATE_270)


def process_pdf(pdf_path):
    """PDFの全ページを処理"""
    doc = fitz.open(pdf_path)
    all_pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        w, h = pix.width, pix.height

        # numpy配列に変換
        raw_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(h, w, pix.n)
        if pix.n == 4:  # RGBA → RGB
            raw_array = raw_array[:, :, :3]

        # 必要に応じて回転（横長→90°CW）
        pil_img = Image.fromarray(raw_array)
        pil_img = auto_rotate(pil_img)
        img_array = np.array(pil_img)
        h, w = img_array.shape[:2]

        # オレンジ検出（トリミング前に実施）
        orange_mask = detect_orange_mask(img_array)
        orange_count = orange_mask.sum()

        # 領域検出（トリミング前の座標）
        regions = find_regions(orange_mask, page_width=w)

        # 灰色背景の吹き出し内の領域をマージ
        gray_labels, gb_bs = find_gray_labels(img_array)
        regions = merge_regions_in_bubbles(regions, gray_labels, gb_bs)
        # マージ後に再ソート（同じ行は左→右）
        mid_x = w // 2
        regions.sort(key=lambda r: (
            0 if r["x"] + r["w"] / 2 < mid_x else 1,
            r["y"] // LINE_Y_TOLERANCE,
            r["x"]
        ))

        # 白余白トリミング
        img_array, crop_x, crop_y = auto_crop(img_array)
        orange_mask = orange_mask[crop_y:crop_y + img_array.shape[0],
                                  crop_x:crop_x + img_array.shape[1]]
        h, w = img_array.shape[:2]

        # 領域座標をトリミング分オフセット
        for r in regions:
            r["x"] -= crop_x
            r["y"] -= crop_y

        # コントラスト・シャープネス強化
        pil_img = Image.fromarray(img_array)
        enhanced_img = enhance_image(pil_img)
        enhanced_array = np.array(enhanced_img)

        # トリミング＋強化オリジナル画像を保存
        orig_path = os.path.join(PAGES_DIR, f"page{page_num + 1}.png")
        enhanced_img.save(orig_path, optimize=True)

        print(f"Page {page_num + 1}: {w}x{h} (cropped from {crop_x},{crop_y})")
        print(f"  Orange pixels: {orange_count}")
        print(f"  Regions found: {len(regions)}")

        # マスク画像を生成・保存（強化済み画像ベース）
        masked = create_masked_image(enhanced_array, orange_mask)
        masked_img = Image.fromarray(masked)
        masked_path = os.path.join(PAGES_DIR, f"page{page_num + 1}_masked.png")
        masked_img.save(masked_path, optimize=True)

        # 領域にIDを付与
        for i, r in enumerate(regions):
            r["id"] = f"p{page_num + 1}-r{i + 1}"

        all_pages.append({
            "page": page_num + 1,
            "width": w,
            "height": h,
            "regions": regions,
        })

        for i, r in enumerate(regions):
            print(f"    Region {i+1}: x={r['x']}, y={r['y']}, "
                  f"w={r['w']}, h={r['h']}")

    doc.close()

    # JSON保存
    json_path = os.path.join(PAGES_DIR, "regions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"pages": all_pages}, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Regions saved to {json_path}")
    return all_pages


def generate_split_pages(full_pages_data):
    """各ページを左右に分割して split/pages/ に保存"""
    split_dir = os.path.join(os.path.dirname(__file__), "split", "pages")
    os.makedirs(split_dir, exist_ok=True)

    split_pages = []
    split_num = 0

    for page_data in full_pages_data:
        pn = page_data["page"]
        orig_w = page_data["width"]
        mid_x = orig_w // 2

        # 元画像とマスク画像を読み込み
        orig = np.array(Image.open(os.path.join(PAGES_DIR, f"page{pn}.png")))
        masked = np.array(Image.open(os.path.join(PAGES_DIR, f"page{pn}_masked.png")))

        for side, label in [("left", "L"), ("right", "R")]:
            split_num += 1
            if side == "left":
                x_start, x_end = 0, mid_x
            else:
                x_start, x_end = mid_x, orig_w

            # 画像を切り出し
            orig_half = orig[:, x_start:x_end]
            masked_half = masked[:, x_start:x_end]
            sh, sw = orig_half.shape[:2]

            Image.fromarray(orig_half).save(
                os.path.join(split_dir, f"page{split_num}.png"), optimize=True)
            Image.fromarray(masked_half).save(
                os.path.join(split_dir, f"page{split_num}_masked.png"), optimize=True)

            # 該当する領域をフィルタ＆座標調整
            half_regions = []
            for r in page_data["regions"]:
                center_x = r["x"] + r["w"] / 2
                if side == "left" and center_x < mid_x:
                    half_regions.append({
                        "x": r["x"],
                        "y": r["y"],
                        "w": r["w"],
                        "h": r["h"],
                        "id": f"s{split_num}-r{len(half_regions) + 1}",
                    })
                elif side == "right" and center_x >= mid_x:
                    half_regions.append({
                        "x": r["x"] - x_start,
                        "y": r["y"],
                        "w": r["w"],
                        "h": r["h"],
                        "id": f"s{split_num}-r{len(half_regions) + 1}",
                    })

            # y座標でソート（同じ行内は左→右）
            half_regions.sort(key=lambda r: (r["y"] // LINE_Y_TOLERANCE, r["x"]))

            split_pages.append({
                "page": split_num,
                "label": f"P{pn}{label}",
                "width": sw,
                "height": sh,
                "regions": half_regions,
            })

            print(f"  Split {split_num} (P{pn}{label}): {sw}x{sh}, "
                  f"{len(half_regions)} regions")

    # JSON保存
    json_path = os.path.join(split_dir, "regions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"pages": split_pages}, f, ensure_ascii=False, indent=2)

    print(f"\nSplit pages saved to {split_dir}")


if __name__ == "__main__":
    full_data = process_pdf(PDF_PATH)
    print("\n--- Generating split pages ---")
    generate_split_pages(full_data)
