from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# 1. 路径配置
# ============================================================

MODEL_PATH =

TEST_IMAGES = Path(

)

TEST_LABELS = Path(

)

OUTPUT_DIR = Path(

)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 评估设置
# ============================================================

IMGSZ = 512
DEVICE = 0

# Reviewer 要求比较的置信度阈值
CONF_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]

# TP / FP 匹配采用 IoU = 0.5
MATCH_IOU = 0.5

MAX_DET = 300

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


# ============================================================
# 3. 你已有的 target-free evaluation 结果
# ============================================================
# 来自 88 张无麻坑影像
#
# conf=0.5: 57 FP, 34 images with FP
# conf=0.6: 34 FP, 23 images with FP
# conf=0.7: 12 FP, 11 images with FP
# conf=0.8:  4 FP,  4 images with FP

TARGET_FREE_METRICS = {
    0.5: {
        "target_free_fp": 57,
        "fp_per_image": 0.6477,
        "fpir_percent": 38.64,
    },
    0.6: {
        "target_free_fp": 34,
        "fp_per_image": 0.3864,
        "fpir_percent": 26.14,
    },
    0.7: {
        "target_free_fp": 12,
        "fp_per_image": 0.1364,
        "fpir_percent": 12.50,
    },
    0.8: {
        "target_free_fp": 4,
        "fp_per_image": 0.0455,
        "fpir_percent": 4.55,
    },
}


# ============================================================
# 4. 加载模型
# ============================================================

def load_model(model_path):
    """
    优先使用 Ultralytics RTDETR。
    如果当前环境中的模型需要 YOLO 类加载，则自动回退。
    """
    try:
        from ultralytics import RTDETR
        print("Loading model with ultralytics.RTDETR ...")
        return RTDETR(model_path)
    except Exception as e:
        print(f"RTDETR loader failed: {e}")
        print("Trying ultralytics.YOLO ...")

        from ultralytics import YOLO
        return YOLO(model_path)


# ============================================================
# 5. YOLO GT 标签读取
# ============================================================

def read_yolo_labels(label_path, img_width, img_height):
    """
    YOLO格式:
        class x_center y_center width height

    坐标均为归一化坐标。

    返回:
        boxes: (N, 4), xyxy像素坐标
        classes: (N,)
    """

    if not label_path.exists():
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int64)
        )

    lines = label_path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).strip().splitlines()

    boxes = []
    classes = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            print(
                f"[Warning] Invalid label line:\n"
                f"  File: {label_path}\n"
                f"  Line: {line}"
            )
            continue

        cls_id = int(float(parts[0]))

        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        # YOLO normalized xywh -> pixel xyxy
        x1 = (xc - w / 2.0) * img_width
        y1 = (yc - h / 2.0) * img_height
        x2 = (xc + w / 2.0) * img_width
        y2 = (yc + h / 2.0) * img_height

        # 防止坐标超出图像
        x1 = max(0.0, min(x1, img_width))
        y1 = max(0.0, min(y1, img_height))
        x2 = max(0.0, min(x2, img_width))
        y2 = max(0.0, min(y2, img_height))

        boxes.append([x1, y1, x2, y2])
        classes.append(cls_id)

    if len(boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int64)
        )

    return (
        np.asarray(boxes, dtype=np.float32),
        np.asarray(classes, dtype=np.int64)
    )


# ============================================================
# 6. IoU
# ============================================================

def box_iou_single(box, boxes):
    """
    一个预测框和多个GT框计算IoU。

    box:
        shape (4,)

    boxes:
        shape (N, 4)

    返回:
        shape (N,)
    """

    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)

    intersection = inter_w * inter_h

    pred_area = max(
        0.0,
        (box[2] - box[0]) * (box[3] - box[1])
    )

    gt_areas = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) *
        np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )

    union = pred_area + gt_areas - intersection

    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0
    )

    return iou


# ============================================================
# 7. 单张图像的一对一匹配
# ============================================================

def match_predictions(
    pred_boxes,
    pred_scores,
    pred_classes,
    gt_boxes,
    gt_classes,
    iou_threshold=0.5
):
    """
    使用 confidence 从高到低，对预测框与GT进行一对一匹配。

    匹配要求：
        1. 类别相同
        2. IoU >= threshold
        3. 一个GT只能匹配一次

    返回:
        TP, FP, FN
    """

    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)

    if num_pred == 0:
        return 0, 0, num_gt

    if num_gt == 0:
        return 0, num_pred, 0

    # confidence从高到低
    order = np.argsort(-pred_scores)

    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    pred_classes = pred_classes[order]

    gt_matched = np.zeros(num_gt, dtype=bool)

    tp = 0
    fp = 0

    for pred_box, pred_cls in zip(
        pred_boxes,
        pred_classes
    ):

        # 只考虑：
        # 1. 类别相同
        # 2. 尚未匹配的GT
        candidate_indices = np.where(
            (gt_classes == pred_cls) &
            (~gt_matched)
        )[0]

        if len(candidate_indices) == 0:
            fp += 1
            continue

        candidate_gt = gt_boxes[candidate_indices]

        ious = box_iou_single(
            pred_box,
            candidate_gt
        )

        best_local_idx = int(np.argmax(ious))
        best_iou = float(ious[best_local_idx])

        if best_iou >= iou_threshold:

            matched_gt_idx = candidate_indices[
                best_local_idx
            ]

            gt_matched[matched_gt_idx] = True

            tp += 1

        else:

            fp += 1

    fn = num_gt - int(gt_matched.sum())

    return tp, fp, fn


# ============================================================
# 8. Precision / Recall / F1
# ============================================================

def calculate_metrics(tp, fp, fn):

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    if precision + recall > 0:

        f1 = (
            2.0 *
            precision *
            recall /
            (precision + recall)
        )

    else:

        f1 = 0.0

    return precision, recall, f1


# ============================================================
# 9. 主程序
# ============================================================

def main():

    # --------------------------------------------------------
    # 检查路径
    # --------------------------------------------------------

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not TEST_IMAGES.exists():
        raise FileNotFoundError(
            f"Test image directory not found:\n"
            f"{TEST_IMAGES}"
        )

    if not TEST_LABELS.exists():
        raise FileNotFoundError(
            f"Test label directory not found:\n"
            f"{TEST_LABELS}"
        )

    # --------------------------------------------------------
    # 获取测试图片
    # --------------------------------------------------------

    image_files = sorted([
        p for p in TEST_IMAGES.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    print("=" * 70)
    print("Reviewer threshold trade-off evaluation")
    print("=" * 70)

    print(f"\nModel:")
    print(MODEL_PATH)

    print(f"\nTest images:")
    print(TEST_IMAGES)

    print(f"\nTest labels:")
    print(TEST_LABELS)

    print(f"\nNumber of test images: {len(image_files)}")
    print(f"IoU matching threshold: {MATCH_IOU}")

    if len(image_files) != 88:
        print(
            f"\n[Warning] Expected 88 test images, "
            f"but found {len(image_files)}."
        )

    # --------------------------------------------------------
    # 模型
    # --------------------------------------------------------

    model = load_model(MODEL_PATH)

    # --------------------------------------------------------
    # 一次推理即可
    #
    # 最低threshold=0.5。
    # 之后0.6/0.7/0.8直接从预测结果过滤。
    # --------------------------------------------------------

    min_conf = min(CONF_THRESHOLDS)

    print("\nRunning inference...")
    print(
        f"Base inference confidence threshold = "
        f"{min_conf}"
    )

    results = model.predict(
        source=str(TEST_IMAGES),
        imgsz=IMGSZ,
        conf=min_conf,
        device=DEVICE,
        max_det=MAX_DET,
        save=False,
        verbose=False,
        stream=False
    )

    print(
        f"Inference finished. "
        f"Results returned: {len(results)}"
    )

    # --------------------------------------------------------
    # 各 threshold 的累计结果
    # --------------------------------------------------------

    totals = {
        conf: {
            "TP": 0,
            "FP": 0,
            "FN": 0
        }
        for conf in CONF_THRESHOLDS
    }

    per_image_rows = []

    total_gt_count = 0

    # --------------------------------------------------------
    # 遍历每张测试图
    # --------------------------------------------------------

    for idx, result in enumerate(results, start=1):

        image_path = Path(result.path)

        label_path = (
            TEST_LABELS /
            f"{image_path.stem}.txt"
        )

        # 原始图像尺寸
        img_h, img_w = result.orig_shape

        gt_boxes, gt_classes = read_yolo_labels(
            label_path,
            img_width=img_w,
            img_height=img_h
        )

        total_gt_count += len(gt_boxes)

        # ----------------------------------------------------
        # 获取模型预测
        # ----------------------------------------------------

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):

            all_pred_boxes = np.empty(
                (0, 4),
                dtype=np.float32
            )

            all_pred_scores = np.empty(
                (0,),
                dtype=np.float32
            )

            all_pred_classes = np.empty(
                (0,),
                dtype=np.int64
            )

        else:

            all_pred_boxes = (
                result.boxes.xyxy
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            all_pred_scores = (
                result.boxes.conf
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            all_pred_classes = (
                result.boxes.cls
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )

        # ----------------------------------------------------
        # 每个 confidence threshold 单独计算
        # ----------------------------------------------------

        for conf_threshold in CONF_THRESHOLDS:

            keep = (
                all_pred_scores >=
                conf_threshold
            )

            pred_boxes = all_pred_boxes[keep]
            pred_scores = all_pred_scores[keep]
            pred_classes = all_pred_classes[keep]

            tp, fp, fn = match_predictions(
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_classes=pred_classes,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                iou_threshold=MATCH_IOU
            )

            totals[conf_threshold]["TP"] += tp
            totals[conf_threshold]["FP"] += fp
            totals[conf_threshold]["FN"] += fn

            p, r, f1 = calculate_metrics(
                tp,
                fp,
                fn
            )

            per_image_rows.append({
                "image": image_path.name,
                "confidence_threshold":
                    conf_threshold,
                "num_gt": len(gt_boxes),
                "num_predictions":
                    len(pred_boxes),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Precision": p,
                "Recall": r,
                "F1": f1,
            })

        if idx % 10 == 0 or idx == len(results):
            print(
                f"Processed "
                f"{idx}/{len(results)} images"
            )

    # ========================================================
    # 10. 汇总指标
    # ========================================================

    summary_rows = []

    for conf_threshold in CONF_THRESHOLDS:

        tp = totals[conf_threshold]["TP"]
        fp = totals[conf_threshold]["FP"]
        fn = totals[conf_threshold]["FN"]

        precision, recall, f1 = calculate_metrics(
            tp,
            fp,
            fn
        )

        tf = TARGET_FREE_METRICS[
            conf_threshold
        ]

        summary_rows.append({
            "Confidence":
                conf_threshold,

            "TP":
                tp,

            "FP_positive_test":
                fp,

            "FN":
                fn,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1,

            "Target_free_FP":
                tf["target_free_fp"],

            "FP_per_image":
                tf["fp_per_image"],

            "FPIR_percent":
                tf["fpir_percent"],
        })

    summary_df = pd.DataFrame(summary_rows)

    per_image_df = pd.DataFrame(
        per_image_rows
    )

    # ========================================================
    # 11. 保存CSV
    # ========================================================

    summary_csv = (
        OUTPUT_DIR /
        "threshold_tradeoff_summary.csv"
    )

    per_image_csv = (
        OUTPUT_DIR /
        "threshold_tradeoff_per_image.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    per_image_df.to_csv(
        per_image_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 12. 打印结果
    # ========================================================

    print("\n")
    print("=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    print(f"\nTotal GT objects in test set: "
          f"{total_gt_count}")

    display_df = summary_df.copy()

    for col in [
        "Precision",
        "Recall",
        "F1",
        "FP_per_image"
    ]:

        display_df[col] = (
            display_df[col]
            .map(lambda x: f"{x:.4f}")
        )

    display_df["FPIR_percent"] = (
        display_df["FPIR_percent"]
        .map(lambda x: f"{x:.2f}")
    )

    print("\n")
    print(
        display_df.to_string(index=False)
    )

    # ========================================================
    # 13. 审稿人论文表格格式
    # ========================================================

    print("\n")
    print("=" * 100)
    print("TABLE FOR THE MANUSCRIPT")
    print("=" * 100)

    reviewer_table = summary_df[
        [
            "Confidence",
            "Precision",
            "Recall",
            "F1",
            "FP_per_image",
            "FPIR_percent"
        ]
    ].copy()

    reviewer_table["Precision"] = (
        reviewer_table["Precision"]
        .map(lambda x: f"{x:.4f}")
    )

    reviewer_table["Recall"] = (
        reviewer_table["Recall"]
        .map(lambda x: f"{x:.4f}")
    )

    reviewer_table["F1"] = (
        reviewer_table["F1"]
        .map(lambda x: f"{x:.4f}")
    )

    reviewer_table["FP_per_image"] = (
        reviewer_table["FP_per_image"]
        .map(lambda x: f"{x:.4f}")
    )

    reviewer_table["FPIR_percent"] = (
        reviewer_table["FPIR_percent"]
        .map(lambda x: f"{x:.2f}")
    )

    print("\n")
    print(
        reviewer_table.to_string(index=False)
    )

    reviewer_csv = (
        OUTPUT_DIR /
        "reviewer_tradeoff_table.csv"
    )

    reviewer_table.to_csv(
        reviewer_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n")
    print("=" * 100)
    print("Files saved")
    print("=" * 100)

    print(f"\nSummary CSV:")
    print(summary_csv)

    print(f"\nPer-image CSV:")
    print(per_image_csv)

    print(f"\nReviewer table CSV:")
    print(reviewer_csv)

    print("\nEvaluation completed.")


# ============================================================
# 14. Run
# ============================================================

if __name__ == "__main__":
    main()