# src/data/preprocessing.py

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict
import yaml
from pathlib import Path
from typing import List, Tuple

# BCCD class mapping
CLASS_MAPPING = {"WBC": 0, "RBC": 1, "Platelets": 2}


def parse_voc_xml(xml_path: Path) -> Tuple[str, int, int, List[Dict]]:
    """
    Parse Pascal VOC XML file and extract filename, image size, and objects.

    Args:
        xml_path: Path to the XML annotation file.

    Returns:
        filename (str): Name of the image file.
        width (int): Image width.
        height (int): Image height.
        objects (list): List of dicts with 'class_id' and 'bbox' [xmin, ymin, xmax, ymax].
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        # Skip if class is not in mapping (just in case)
        if name not in CLASS_MAPPING:
            continue
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        objects.append(
            {
                "class_id": CLASS_MAPPING[name],
                "bbox": [xmin, ymin, xmax, ymax],
            }
        )

    return filename, width, height, objects


def convert_voc_to_yolo(
    bbox: List[float], img_width: int, img_height: int
) -> List[float]:
    """
    Convert absolute VOC coordinates to normalized YOLO format.

    Args:
        bbox: [xmin, ymin, xmax, ymax].
        img_width: Width of the image.
        img_height: Height of the image.

    Returns:
        list: [center_x, center_y, width, height] normalized to [0, 1].
    """
    xmin, ymin, xmax, ymax = bbox

    # Calculate center and dimensions in absolute pixels
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    # Normalize
    center_x /= img_width
    center_y /= img_height
    bbox_width /= img_width
    bbox_height /= img_height

    # Clamp values to avoid rounding errors that exceed 1.0
    center_x = min(max(center_x, 0.0), 1.0)
    center_y = min(max(center_y, 0.0), 1.0)
    bbox_width = min(max(bbox_width, 0.0), 1.0)
    bbox_height = min(max(bbox_height, 0.0), 1.0)

    return [center_x, center_y, bbox_width, bbox_height]



def generate_yolo_data_yaml(
    images_dir: Path,
    labels_dir: Path,
    train_list_path: Path,
    val_list_path: Path,
    class_names: List[str],
    output_path: Path,
) -> None:
    
    data_config = {
        'path': str(images_dir.parent.resolve()),  # Root directory of the dataset
        'train': str(train_list_path.resolve()),   # Absolute path to train.txt
        'val': str(val_list_path.resolve()),       # Absolute path to val.txt
        'nc': len(class_names),                    # Number of classes
        'names': class_names,                      # List of class names
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"[SUCCESS] data.yaml saved to {output_path}")


def load_yolo_labels(label_path: Path, img_width: int, img_height: int) -> List[Tuple[int, Tuple[int, int, int, int]]]:
    
    boxes = []
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            center_x = float(parts[1]) * img_width
            center_y = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            xmin = int(center_x - width / 2)
            ymin = int(center_y - height / 2)
            xmax = int(center_x + width / 2)
            ymax = int(center_y + height / 2)
            
            boxes.append((class_id, (xmin, ymin, xmax, ymax)))
    
    return boxes