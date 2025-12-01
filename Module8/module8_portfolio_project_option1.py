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


if __name__ == "__main__":
    # Load cascade
    plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    image_files = [
        'russian_plate_1.jpg',
        'russian_plate_2_far.jpg',
        'non_russian.jpg'
    ]

    reader = easyocr.Reader(['ru', 'en'])  # Supports Russian and English
    results = []

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

            # Display result
            cv2.imshow("Color License Plate Detection", orig_img)
            cv2.imshow("Grayscale License Plate Detection", gray_img)
            cv2.imshow("Rotated", rotated)

            # Preprocessing: Resize to standard size, threshold, and rotate if needed
            # plate = cv2.resize(plate, (200, 50))  # Scale to horizontal
            # _, plate = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Detect skew and rotate
            # coords = np.column_stack(np.where(plate > 0))
            # angle = cv2.minAreaRect(coords)[-1]
            # if angle < -45:
            #     angle = -(90 + angle)
            # else:
            #     angle = -angle
            # plate = imutils.rotate_bound(plate, angle)

            # Recognize text
            text = reader.readtext(rotated, detail=0)
            results.append(' '.join(text))  # Join detected characters

            # Insert processed plate back into full image (optional overlay)
            # Or just save separately
            cv2.imwrite(f"output/{os.path.splitext(img_file)[0]}_plate_grayscale_{i}.jpg", gray_img)
            cv2.imwrite(f"output/{os.path.splitext(img_file)[0]}_plate_color_{i}.jpg", orig_img)
            cv2.imwrite(f"output/{os.path.splitext(img_file)[0]}_rotated_{i}.jpg", rotated)

            print(f"  Plate {i + 1}: Found {len(text)} potential characters: {text}")

            # Save annotated image
            cv2.imwrite(f"output/{os.path.splitext(img_file)[0]}_annotated.jpg", img)

        # Display result
        display_img = cv2.resize(gray_img, (800, 600))
        cv2.imshow("License Plate Detection", display_img)

        print("Press any key to continue...")
        cv2.waitKey(0)

    cv2.destroyAllWindows()
