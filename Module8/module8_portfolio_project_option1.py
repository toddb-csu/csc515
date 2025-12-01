# Todd Bartoszkiewicz
# CSC515: Foundations of Computer Vision
# Module 8 Portfolio Project Option #1: License Plate Detection and Faked Content
# Reliable recognition of objects is a challenging task for artificial intelligence, yet humans are able to perform
# these kinds of tasks with ease.
#
# The goal of this project is to write algorithms for license plate detection and license plate character recognition.
# Select three color images from the internet that meet the following requirements:
#
# Two images containing vehicles with Russian license plates and one image of vehicles with a non-Russian plate.
# All images should include the entire vehicle and not just the license plate.
# At least one image with Russian plates should display the license plate far away.
# At least one image should include multiple vehicles.
# All images should vary in light illumination and color intensity.
# First, using the appropriate trained cascade classifierLinks to an external site., write one algorithm to detect the
# Russian license plate in the gray scaled versions of the original images.  Put a red boundary box around the detected
# plate in the image in order to see what region the classifier deemed as a license plate.  If expected results are not
# achieved on the unprocessed images, apply processing steps before implementing the classifier for optimal results.
#
# After the license plates have been successfully detected, you will want to process only the extracted plate region
# before applying character recognition on it.  Although the license plate number classifierLinks to an external site.
# is fairly accurate, it is important that all license plates are rotated and scaled so that they are horizontally
# aligned. If expected results are not achieved, implement more image processing for optimal character recognition.
#
# Inspect your results and write a summary describing the techniques you used to detect and identify characters of
# Russian license plates in images. Reflect on the challenges you faced and how you overcame these challenges.
# Furthermore, discuss in your summary, the accuracy of your results for all three images and techniques you used to
# improve the accuracy after each repeated experiment.
#
# Your submission should be one executable Python script and one summary of 2-3 pages in length.
import cv2
import os
import numpy as np
import easyocr
import pytesseract


if __name__ == "__main__":
    # Load cascade
    plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    image_files = [
        'russian_plate_1.jpg',
        'russian_plate_2_far.jpg',
        'non_russian.jpg'
    ]

    # Russian and English
    reader = easyocr.Reader(['ru', 'en'])
    results = []

    original_width = 0
    original_height = 0
    scale_factor = 0.75

    for img_file in image_files:
        # Read image
        img = cv2.imread(img_file)

        orig_img = img.copy()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Equalize histogram for better contrast
        gray_img = cv2.equalizeHist(gray_img)

        # Detect plates
        plates = plate_cascade.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 15),
            maxSize=(300, 100)
        )

        print(f"\nProcessing {img_file}...")
        print(f"Detected {len(plates)} plate(s)")

        for i, (x, y, w, h) in enumerate(plates):
            # Draw red rectangle around detected plate
            cv2.rectangle(gray_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Extract plate region
            plate_roi = gray_img[y:y + h, x:x + w]

            # Align and deskew
            edges = cv2.Canny(plate_roi, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)

            angle = 0
            if lines is not None:
                # Estimate dominant angle
                angles = [line[0][1] * 180 / np.pi for line in lines[:10]]
                mean_angle = np.mean(angles)

                # Convert Hough angle to rotation correction
                if 80 < mean_angle < 100:
                    angle = mean_angle - 90

            center = (plate_roi.shape[1] // 2, plate_roi.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(plate_roi, M, (plate_roi.shape[1], plate_roi.shape[0]))
            enhanced_img = rotated.copy()

            if original_width == 0:
                dimensions = enhanced_img.shape
                original_height = dimensions[0]
                original_width = dimensions[1]

            dimensions = enhanced_img.shape
            if dimensions[1] > original_width:
                enhanced_img = cv2.resize(enhanced_img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_AREA)

            enhanced_img = cv2.medianBlur(enhanced_img, 3)
            enhanced_img = cv2.adaptiveThreshold(enhanced_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 4)
            enhanced_img = cv2.fastNlMeansDenoising(enhanced_img, None, 10, 7, 21)
            _, enhanced_img = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            enhanced_img = cv2.fastNlMeansDenoising(enhanced_img, None, 10, 7, 21)
            enhanced_img = cv2.addWeighted(enhanced_img, 1, enhanced_img, 1, 1)
            enhanced_img = cv2.medianBlur(enhanced_img, 3)

            sharpening_kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])
            enhanced_img = cv2.filter2D(enhanced_img, -1, sharpening_kernel)

            blurred = cv2.GaussianBlur(enhanced_img, (3, 3), 1.0)
            sharpened = float(1.0 + 1) * enhanced_img - float(1.0) * blurred
            sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
            sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
            enhanced_img = sharpened.round().astype(np.uint8)

            enhanced_img = cv2.convertScaleAbs(enhanced_img, 0.0, 100)
            enhanced_img = cv2.bilateralFilter(enhanced_img, 15, 15, 15, 15)

            # Display result
            cv2.imshow("Color License Plate Detection", orig_img)
            cv2.imshow("Grayscale License Plate Detection", gray_img)
            cv2.imshow("Rotated", rotated)

            # Recognize text
            #
            text = pytesseract.image_to_string(rotated, lang='rus')
            if not text:
                text = reader.readtext(rotated, detail=0)
            results.append(' '.join(text))  # Join detected characters

            # Save image files
            cv2.imwrite(f"output/{os.path.splitext(img_file)[0]}_plate_grayscale_{i}.jpg", gray_img)
            cv2.imwrite(f"output/{os.path.splitext(img_file)[0]}_plate_color_{i}.jpg", orig_img)
            cv2.imwrite(f"output/{os.path.splitext(img_file)[0]}_rotated_{i}.jpg", rotated)
            cv2.imwrite(f"output/{os.path.splitext(img_file)[0]}_enhanced_{i}.jpg", enhanced_img)

            print(f"  Plate {i + 1}: Found {len(text)} potential characters: {text}")

            # Save annotated image
            cv2.imwrite(f"output/{os.path.splitext(img_file)[0]}_annotated.jpg", img)

        # Display result
        display_img = cv2.resize(gray_img, (800, 600))
        cv2.imshow("License Plate Detection", display_img)

        print("Press any key to continue...")
        cv2.waitKey(0)

    cv2.destroyAllWindows()
