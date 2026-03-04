import os
import cv2
import numpy as np

def enhance_landscape_image(img_rgb, apply_clahe=False, apply_gamma=False, gamma=1.2, clahe_limit=2.0, clahe_grid=(8, 8)):
    result = img_rgb.copy()

    if apply_clahe:
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=clahe_grid)
        v = clahe.apply(v)
        hsv = cv2.merge((h, s, v))
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if apply_gamma:
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v = np.power(v / 255.0, gamma)
        v = np.uint8(np.clip(v * 255, 0, 255))
        hsv = cv2.merge((h, s, v))
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return result

def enhance_dataset_images(dataset_path):
    output_path = os.path.join(dataset_path, 'dataset_enhanced')
    os.makedirs(output_path, exist_ok=True)

    for fname in os.listdir(dataset_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_file = os.path.join(dataset_path, fname)
            output_file = os.path.join(output_path, fname)

            img_bgr = cv2.imread(input_file)
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            enhanced = enhance_landscape_image(
                img_rgb,
                apply_clahe=True,
                clahe_limit=1.0,
                clahe_grid=(8, 8),
                apply_gamma=True,
                gamma=1.05
            )
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_file, enhanced_bgr)