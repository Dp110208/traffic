import cv2


def enhance_plate(plate):
    # 1. Upscale
    plate = cv2.resize(
        plate,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # 2. Convert to grayscale
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # 3. Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    # 4. Sharpen
    sharpened = cv2.detailEnhance(
        cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
        sigma_s=10,
        sigma_r=0.15
    )
    

    return sharpened