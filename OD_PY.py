#!/usr/bin/env python3
"""
Convert Label Studio (object detection) JSON export -> OCI Vision Custom Model JSONL.

- Supports Label Studio results of type "rectanglelabels"
- Converts percentage-based rectangles into OCI's normalized polygon (4 vertices)
- Produces:
  * One header line describing the dataset (labelsSet, formats, bucket, etc.)
  * One record line per image with annotations

Usage:
  python ls_to_oci_jsonl.py \
    --input /path/to/label_studio_export.json \
    --output /path/to/oci_vision_dataset.jsonl \
    --compartment-ocid ocid1.compartment.oc1..example \
    --namespace your_namespace \
    --bucket your_bucket \
    --display-name od \
    --image-prefix images/

Notes:
- If your images are stored at a different prefix/path in Object Storage, adjust --image-prefix.
- If your LS export uses `data.image` URLs/paths, we take the basename to form
  `image-prefix + basename`.
- The header's OCIDs for dataset/record/annotation are synthetically generated and
  don't need to be real for training to work.
"""

import argparse
import datetime
import json
import os
import sys
import uuid
from typing import Any, Dict, List, Optional


def make_ocid(prefix: str, region: str = "phx") -> str:
    """Generate a synthetic OCID-like string."""
    return f"ocid1.{prefix}.oc1.{region}.{uuid.uuid4().hex[:48]}"


def get_image_basename(task: Dict[str, Any]) -> Optional[str]:
    """
    Extract a filename for the image from a Label Studio task.
    Priority:
      1) task['file_upload']
      2) task['data']['image'] or task['data']['img']
    Returns basename or None if not found.
    """
    if "file_upload" in task and task["file_upload"]:
        return os.path.basename(str(task["file_upload"]))
    data = task.get("data", {})
    for key in ("image", "img"):
        if key in data and data[key]:
            return os.path.basename(str(data[key]))
    return None


def collect_labels(tasks: List[Dict[str, Any]]) -> List[str]:
    """Collect distinct label names from rectanglelabels results."""
    seen = []
    for t in tasks:
        for ann in t.get("annotations", []):
            for r in ann.get("result", []):
                if r.get("type") == "rectanglelabels":
                    for lab in r.get("value", {}).get("rectanglelabels", []):
                        if lab not in seen:
                            seen.append(lab)
    return seen


def normalize_rect(value: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Label Studio rectangle values are in percentages of the image:
      x, y, width, height in [0..100]
    Convert them to normalized [0..1] polygon with 4 vertices in OCI order:
      (x_left, y_top) -> (x_left, y_bottom) -> (x_right, y_bottom) -> (x_right, y_top)
    """
    try:
        x = float(value["x"]) / 100.0
        y = float(value["y"]) / 100.0
        w = float(value["width"]) / 100.0
        h = float(value["height"]) / 100.0
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid rectangle value: {value!r} ({e})")

    x1 = max(0.0, min(1.0, x))
    y1 = max(0.0, min(1.0, y))
    x2 = max(0.0, min(1.0, x + w))
    y2 = max(0.0, min(1.0, y + h))

    return [
        {"x": f"{x1}", "y": f"{y1}"},
        {"x": f"{x1}", "y": f"{y2}"},
        {"x": f"{x2}", "y": f"{y2}"},
        {"x": f"{x2}", "y": f"{y1}"},
    ]


def build_header(
    compartment_ocid: str,
    display_name: str,
    namespace: str,
    bucket: str,
    labels: List[str],
    annotation_format: str = "BOUNDING_BOX",
    dataset_format: str = "IMAGE",
) -> Dict[str, Any]:
    """Create the first header line for OCI Vision dataset JSONL."""
    return {
        "id": make_ocid("datalabelingdataset"),
        "compartmentId": compartment_ocid,
        "displayName": display_name,
        "labelsSet": [{"name": name} for name in labels],
        "annotationFormat": annotation_format,
        "datasetSourceDetails": {
            "namespace": namespace,
            "bucket": bucket,
        },
        "datasetFormatDetails": {
            "formatType": dataset_format,
        },
    }


def task_to_record(
    task: Dict[str, Any],
    image_prefix: str,
    now_str: str,
) -> Optional[Dict[str, Any]]:
    """
    Convert a single Label Studio task to an OCI DLS-style 'record' object.
    Returns None if the task has no usable image or no annotations.
    """
    base = get_image_basename(task)
    if not base:
        return None

    record = {
        "id": make_ocid("datalabelingrecord"),
        "timeCreated": now_str,
        "sourceDetails": {
            "sourceType": "OBJECT_STORAGE",
            "path": f"{image_prefix}{base}",
        },
        "annotations": [],
    }

    any_entities = False

    for ann in task.get("annotations", []):
        entities = []
        for r in ann.get("result", []):
            if r.get("type") != "rectanglelabels":
                continue
            rect_value = r.get("value", {})
            labels = rect_value.get("rectanglelabels", [])
            if not labels:
                continue

            polygon = normalize_rect(rect_value)
            entities.append(
                {
                    "entityType": "IMAGEOBJECTSELECTION",
                    "labels": [{"label_name": lbl} for lbl in labels],
                    "boundingPolygon": {"normalizedVertices": polygon},
                }
            )

        if entities:
            any_entities = True
            record["annotations"].append(
                {
                    "id": make_ocid("datalabelingannotation"),
                    "timeCreated": now_str,
                    "createdBy": make_ocid("user"),
                    "entities": entities,
                }
            )

    if not any_entities:
        return None
    return record


def convert(
    ls_json_path: str,
    output_path: str,
    compartment_ocid: str,
    namespace: str,
    bucket: str,
    display_name: str = "od",
    image_prefix: str = "images/",
    annotation_format: str = "BOUNDING_BOX",
    dataset_format: str = "IMAGE",
) -> None:
    """Run full conversion and write JSONL file."""
    try:
        with open(ls_json_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        raise SystemExit(f"Failed to read Label Studio export: {e}")

    if not isinstance(tasks, list):
        raise SystemExit("Label Studio export root must be a JSON array.")

    labels = collect_labels(tasks)
    header = build_header(
        compartment_ocid=compartment_ocid,
        display_name=display_name,
        namespace=namespace,
        bucket=bucket,
        labels=labels,
        annotation_format=annotation_format,
        dataset_format=dataset_format,
    )
    now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = [json.dumps(header, ensure_ascii=False)]

    count_records = 0
    for task in tasks:
        rec = task_to_record(task, image_prefix=image_prefix, now_str=now_str)
        if rec:
            lines.append(json.dumps(rec, ensure_ascii=False))
            count_records += 1

    if count_records == 0:
        print("Warning: No annotations found; only header will be written.", file=sys.stderr)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Wrote {count_records} record(s) + header to: {output_path}")


def main():
    p = argparse.ArgumentParser(
        description="Convert Label Studio OD JSON export to OCI Vision JSONL."
    )
    p.add_argument("--input", required=True, help="Path to Label Studio JSON export")
    p.add_argument("--output", required=True, help="Path to write OCI JSONL")
    p.add_argument("--compartment-ocid", required=True, help="OCI Compartment OCID")
    p.add_argument("--namespace", required=True, help="Object Storage namespace")
    p.add_argument("--bucket", required=True, help="Object Storage bucket name")
    p.add_argument("--display-name", default="od", help="Dataset display name")
    p.add_argument("--image-prefix", default="images/", help="Prefix for object paths")
    p.add_argument(
        "--annotation-format",
        default="BOUNDING_BOX",
        choices=["BOUNDING_BOX"],
        help="OCI annotation format (BOUNDING_BOX only for OD).",
    )
    p.add_argument(
        "--dataset-format",
        default="IMAGE",
        choices=["IMAGE"],
        help="OCI dataset format (IMAGE for OD).",
    )
    args = p.parse_args()

    convert(
        ls_json_path=args.input,
        output_path=args.output,
        compartment_ocid=args.compartment_ocid,
        namespace=args.namespace,
        bucket=args.bucket,
        display_name=args.display_name,
        image_prefix=args.image_prefix,
        annotation_format=args.annotation_format,
        dataset_format=args.dataset_format,
    )


if __name__ == "__main__":
    main()

