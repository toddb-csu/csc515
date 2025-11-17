# Todd Bartoszkiewicz
# CSC515: Foundations of Computer Vision
# Module 6 Option 1 Critical Thinking
# If an image has been preprocessed properly to remove noise a key step that is generally used when interpreting that
# image is segmentation.  Image segmentation is a process in which regions or features sharing similar characteristics
# are identified and grouped together.
#
# The thresholds in the algorithms discussed in this module were chosen by the designer. In order to make segmentation
# stronger to variations in the scene, the algorithm should be able to select an appropriate threshold automatically
# using the amount of intensity present in the image. The knowledge about the gray values of objects should not be
# hard-coded into an algorithm. The algorithm should use knowledge about the relative characteristics of gray values to
# select the appropriate threshold.  A thresholding scheme that uses such knowledge and selects a proper threshold value
# for each image without human intervention is called an adaptive thresholding scheme.
#
# Select two objects, one light and one dark. Put both objects on a flat surface. Illuminate the scene so that light
# comes from the direction of the light object. Take a picture of the scene.  Develop a suitable adaptive thresholding
# scheme to segment the image as best as you can. Now, change the direction of illumination by moving your light source
# so that the illumination is from the direction of the dark object. Again, take a picture of the scene.  Using this
# image, see whether your algorithm still works. If not, revise the algorithm to make it work.
#
# Your submission should be one executable Python script and one summary of 2-3 pages in length that conforms to CSU
# Global Writing Center. Include at least two scholarly references in addition to the course textbook. The CSU Global
# Library is a good place to find these references. The Writing Center and Library can be accessed by clicking on the
# tabs in the course navigation panel.
##############################
# Apply three thresholding methods to each image:
# - Adaptive Mean Thresholding
# - Adaptive Gaussian Thresholding
# - OTSU's Thresholding
import cv2
import matplotlib.pyplot as plt

images = {
    "indoor": "indoor.png",
    "outdoor": "outdoor.png",
    "closeup": "closeup.png"
}

# ============================
# Run Thresholding on each image
# ============================
for name, image in images.items():
    print(f"Processing {name} image...")
    img = cv2.imread(image)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocessing — light smoothing
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Adaptive Mean Thresholding
    mean_thres = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        21,  # block size
        5    # constant subtracted from mean
    )

    # Adaptive Gaussian Thresholding
    gauss_thres = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        5
    )

    # OTSU Thresholding
    _, otsu_thres = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Save images
    cv2.imwrite(f"{name}_adaptive_mean.png", mean_thres)
    cv2.imwrite(f"{name}_adaptive_gaussian.png", gauss_thres)
    cv2.imwrite(f"{name}_otsu.png", otsu_thres)

    # Show images after thresholding
    plt.figure(figsize=(10, 4))
    plt.suptitle(f"Thresholding Results: {name.capitalize()}")

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")

    plt.subplot(1, 4, 2)
    plt.imshow(mean_thres, cmap='gray')
    plt.title("Adaptive Mean")

    plt.subplot(1, 4, 3)
    plt.imshow(gauss_thres, cmap='gray')
    plt.title("Adaptive Gaussian")

    plt.subplot(1, 4, 4)
    plt.imshow(otsu_thres, cmap='gray')
    plt.title("OTSU Threshold")

    plt.tight_layout()
    plt.show()

print("Complete")
