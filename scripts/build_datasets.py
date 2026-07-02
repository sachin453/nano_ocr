import os
import cv2
import json
import numpy as np
import scipy.io as sio
import glob


def crop_quadrilateral(img, pts):
    """
    Takes an image and 8 points (4 corners) and performs a perspective warp
    to flatten the word into a perfect rectangle.
    """
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)

    # Compute width and height of the new cropped rectangle
    width_A = np.linalg.norm(pts[2] - pts[3])
    width_B = np.linalg.norm(pts[1] - pts[0])
    maxWidth = max(int(width_A), int(width_B))

    height_A = np.linalg.norm(pts[1] - pts[2])
    height_B = np.linalg.norm(pts[0] - pts[3])
    maxHeight = max(int(height_A), int(height_B))

    # Destination points for a flat rectangle
    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype=np.float32,
    )

    # Warp it
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped


def process_icdar15_detection(icdar_dir, output_crop_dir):
    dataset = []
    os.makedirs(output_crop_dir, exist_ok=True)

    img_dir = os.path.join(icdar_dir, "ch4_training_images")
    gt_dir = os.path.join(
        icdar_dir, "ch4_training_localization_transcription_gt"
    )

    img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    crop_counter = 0

    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # ICDAR GT files are named 'gt_img_X.txt'
        gt_name = "gt_" + img_name.replace(".jpg", ".txt")
        gt_path = os.path.join(gt_dir, gt_name)

        if not os.path.exists(gt_path):
            continue

        with open(gt_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 9:
                    # First 8 parts are the X,Y coordinates of the 4 corners
                    coords = [int(p) for p in parts[:8]]

                    # The rest is the text (sometimes contains commas, so we join it back)
                    label = ",".join(parts[8:]).strip('"')

                    if label == "###":  # ### means unreadable/blurred in ICDAR
                        continue

                    try:
                        # Extract the word
                        cropped = crop_quadrilateral(img, coords)

                        # Save the crop
                        crop_filename = f"crop_{crop_counter:06d}.jpg"
                        crop_filepath = os.path.join(
                            output_crop_dir, crop_filename
                        )
                        cv2.imwrite(crop_filepath, cropped)

                        # Add to JSON list
                        dataset.append(
                            {
                                "path": os.path.abspath(crop_filepath),
                                "label": label,
                            }
                        )
                        crop_counter += 1
                    except Exception as e:
                        pass  # Skip invalid polygons

    return dataset


def process_iiit5k(iiit5k_dir):
    dataset = []
    mat_path = os.path.join(iiit5k_dir, "traindata.mat")

    mat = sio.loadmat(mat_path)
    for item in mat["traindata"][0]:
        img_name = item["ImgName"][0]
        label = item["GroundTruth"][0]

        img_path = os.path.abspath(os.path.join(iiit5k_dir, img_name))
        dataset.append({"path": img_path, "label": label})

    return dataset


if __name__ == "__main__":
    # Point these to your specific data directories
    icdar_folder = "./data/icdar2015/"
    iiit5k_folder = "./data/iiit5kwords/"
    icdar_crops_folder = "./data/icdar2015_crops/"  # Where extracted words will go

    unified_data = []

    print("Extracting and Cropping ICDAR 2015...")
    unified_data.extend(
        process_icdar15_detection(icdar_folder, icdar_crops_folder)
    )
    print(f"Extracted {len(unified_data)} ICDAR crops.")

    print("Parsing IIIT5K...")
    iiit_data = process_iiit5k(iiit5k_folder)
    unified_data.extend(iiit_data)
    print(f"Parsed {len(iiit_data)} IIIT5K images.")

    output_file = "train_labels.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unified_data, f, indent=4)

    print(
        f"Success! {len(unified_data)} total real-world images indexed into {output_file}."
    )