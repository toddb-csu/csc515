# Todd Bartoszkiewicz
# CSC515: Foundations of Computer Vision
# Module 5 Option 1 Critical Thinking
# Fingerprints are tiny patterns on the tip of every finger. The uniqueness of fingerprints has been studied and it is
# well established that the probability of two fingerprints matching is extremely small. Since they are unique,
# fingerprints can be used to identify or confirm the identity of an individual.
#
# Fingerprint recognition refers to the automated method of identifying or confirming the identity of an individual
# based on the comparison of two fingerprints. These comparisons can be made based on the type and the position of the
# ridge characteristics. The performance of fingerprint recognition applications heavily rely on the input fingerprint
# image quality.
#
# A latent fingerprint is left on a surface by deposits of oils and/or perspiration from the finger. It is not usually
# visible but may be detected with special techniques such as dusting for fingerprints. In order to reduce rejection
# rates in most cases the acquired latent fingerprints have to be enhanced prior to matching to reduce the degradation,
# noise, or incompleteness. Enhancement can be achieved using morphological image processing.
#
# Acquire an image of a latent fingerprint. In OpenCV, write algorithms to process the image using morphological
# operations (dilation, erosion, opening, and closing).
#
# Next, write a 2-3 page summary of your observed results. Include in your summary, the following:
#
# Describe in detail what enhancements did each morphological operation make on the image and how beneficial these
# enhancements are for fingerprint recognition.
# Did the enhancement also result in data loss of other features? Explain.
# Research morphological operations for fingerprint enhancements and include whether your results were similar with the
# findings in these. Be sure to cite them in your summary using correct APA styling.
# Your submission should be one executable Python script and one summary of 2-3 pages in length that conforms to CSU
# Global Writing Center. Include at least two scholarly references in addition to the course textbook. The CSU Global
# Library is a good place to find these references. The Writing Center and Library can be accessed by clicking on the
# tabs in the course navigation panel.
import cv2
import matplotlib.pyplot as plt


if __name__ == "__main__":
    img = cv2.imread('fingerprint.jpg', cv2.IMREAD_GRAYSCALE)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    block = 3
    C = 7
    binary_clahe = cv2.adaptiveThreshold(
        img_clahe, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block, C=C
    )
    cv2.imwrite("binary_clahe.png", binary_clahe)

    k = 3
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    dilated_clahe = cv2.dilate(img_clahe, se, iterations=1)
    eroded_clahe = cv2.erode(img_clahe,  se, iterations=1)
    opened_clahe = cv2.morphologyEx(img_clahe, cv2.MORPH_OPEN,  se, iterations=1)
    closed_clahe = cv2.morphologyEx(img_clahe, cv2.MORPH_CLOSE, se, iterations=1)
    close_then_open_clahe = cv2.morphologyEx(closed_clahe, cv2.MORPH_OPEN, se, iterations=1)

    cv2.imwrite("dilate_clahe.png", dilated_clahe)
    cv2.imwrite("erode_clahe.png",  eroded_clahe)
    cv2.imwrite("open_clahe.png",   opened_clahe)
    cv2.imwrite("close_clahe.png",  closed_clahe)
    cv2.imwrite("close_then_open_clahe.png", close_then_open_clahe)

    dilated = cv2.dilate(img, se, iterations=1)
    eroded = cv2.erode(img,  se, iterations=1)
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN,  se, iterations=1)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se, iterations=1)
    close_then_open = cv2.morphologyEx(closed, cv2.MORPH_OPEN, se, iterations=1)

    cv2.imwrite("dilate.png", dilated)
    cv2.imwrite("erode.png",  eroded)
    cv2.imwrite("open.png",   opened)
    cv2.imwrite("close.png",  closed)
    cv2.imwrite("close_then_open.png", close_then_open)

    images = [
        (img, "Original (grayscale)"),
        (img_clahe, "CLAHE"),
        (binary_clahe, f"Binary CLAHE (Adaptive: block={block}, C={C})"),
        (dilated_clahe, f"Dilation CLAHE (SE={k}×{k} ellipse)"),
        (eroded_clahe, f"Erosion CLAHE (SE={k}×{k} ellipse)"),
        (opened_clahe, f"Opening CLAHE (SE={k}×{k} ellipse)"),
        (closed_clahe, f"Closing CLAHE (SE={k}×{k} ellipse)"),
        (close_then_open_clahe, f"Close→Open CLAHE (SE={k}×{k} ellipse)"),
        (dilated, f"Dilation (SE={k}×{k} ellipse)"),
        (eroded, f"Erosion (SE={k}×{k} ellipse)"),
        (opened, f"Opening (SE={k}×{k} ellipse)"),
        (closed, f"Closing (SE={k}×{k} ellipse)"),
        (close_then_open, f"Close→Open (SE={k}×{k} ellipse)")
    ]

    fig, axes = plt.subplots(3, 5, figsize=(16, 8), constrained_layout=True)
    for idx, (im, title) in enumerate(images):
        r, c = divmod(idx, 5)
        axes[r, c].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[r, c].set_title(title, fontsize=11)
        axes[r, c].axis("off")

    fig.suptitle("Morphology for Fingerprint Image Enhancement", fontsize=15, fontweight="bold")
    fig.savefig('fingerprint3.jpg', dpi=200)
