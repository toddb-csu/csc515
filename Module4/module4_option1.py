# Todd Bartoszkiewicz
# CSC515: Foundations of Computer Vision
# Module 4 Option 1 Critical Thinking
# Image filtering involves the application of window operations that perform useful functions, such as noise removal
# and image enhancement. Compare the effects of mean, median, and Gaussian filters on an image for different kernel
# windows.
#
# This image contains impulse noise. In OpenCV, write algorithms for this image to do the following:
#
# Apply mean, median, and Gaussian filters using a 3x3 kernel. Additionally, for Gaussian, select two different values
# of sigma. Think about how to select good values of sigma for optimal results.
# Apply mean, median, and Gaussian filters using a 5x5 kernel. For Gaussian, use the same values of sigma you selected
# in the above step.
# Apply mean, median, and Gaussian filters using a 7x7 kernel. For Gaussian, use the same values of sigma you selected
# in the above step.
# Output your filter results as 3 x 4 side-by-side subplots to make comparisons easy to inspect visually. That is, your
# subplot should have 3 rows (1 for each kernel size) and 4 columns (1 for each filter type, 2 for Gaussian). Be sure
# to include row and column labels.
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    img = cv2.imread('Mod4CT1.jpg')

    kernel_sizes = [3, 5, 7]
    results_by_row = []
    for k in kernel_sizes:
        # Mean filter
        mean_img = cv2.blur(img, (k, k))
        # Median filter
        median_img = cv2.medianBlur(img, k)
        # Gaussian filters
        gauss1 = cv2.GaussianBlur(img, (k, k), sigmaX=0.8, sigmaY=0.8, borderType=cv2.BORDER_REFLECT)
        gauss2 = cv2.GaussianBlur(img, (k, k), sigmaX=2.0, sigmaY=2.0, borderType=cv2.BORDER_REFLECT)
        results_by_row.append((k, [mean_img, median_img, gauss1, gauss2]))

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)

    # Column titles
    axes[0, 0].set_title("Mean", fontsize=13, fontweight='bold')
    axes[0, 1].set_title("Median", fontsize=13, fontweight='bold')
    axes[0, 2].set_title("Gaussian (sigma=0.8)", fontsize=13, fontweight='bold')
    axes[0, 3].set_title("Gaussian (sigma=2.0)", fontsize=13, fontweight='bold')

    row_titles = [
        "3×3 kernel",
        "5×5 kernel",
        "7×7 kernel"
    ]
    for i, (k, imgs) in enumerate(results_by_row):
        # Row label
        axes[i, 0].set_ylabel(row_titles[i], fontsize=13, fontweight='bold')
        for j, im in enumerate(imgs):
            ax = axes[i, j]
            disp = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            ax.imshow(disp)

    fig.suptitle("Mean vs. Median vs. Gaussian Filtering", fontsize=16, fontweight='bold')
    fig.savefig('filter_comparison.jpg', dpi=200)
    plt.show()
    plt.close(fig)
