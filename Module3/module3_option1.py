# Todd Bartoszkiewicz
# CSC515: Foundations of Computer Vision
# Module 3 Option 1 Portfolio Project
import cv2

if __name__ == '__main__':
    try:
        my_picture = cv2.imread('IMG_8706.jpg')
        # The picture is pretty large, so let's resize it
        resized_image = cv2.resize(my_picture, None, fx=0.15, fy=0.15)

        # Load the Haar Cascade classifiers
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        # Convert the selfie image to grayscale for the detectors
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        # scaleFactor: How much the image size is reduced at each image scale.
        # minNeighbors: How many neighbors each candidate rectangle should have to retain it.
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Sort faces by area and take the largest one
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            main_face = faces[0]
            (face_x, face_y, face_w, face_h) = main_face

            # Draw a green circle around the face
            # Calculate the center and radius of the circle
            center_x = face_x + face_w // 2
            center_y = face_y + face_h // 2
            # Make the circle slightly larger than the face width
            radius = int(face_w / 2 * 1.2)

            # cv2.circle(image, center_coordinates, radius, color, thickness)
            cv2.circle(resized_image, (center_x, center_y), radius, (0, 255, 0), 3)

            # Draw red bounding boxes for the eyes
            # Create a Region of Interest (ROI) for the face to search for eyes
            roi_gray = gray[face_y:face_y + face_h, face_x:face_x + face_w]
            roi_color = resized_image[face_y:face_y + face_h, face_x:face_x + face_w]

            # Detect eyes within the face ROI
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

            for (eyes_x, eyes_y, eyes_w, eyes_h) in eyes:
                # cv2.rectangle(image, start_point, end_point, color, thickness)
                # The coordinates are relative to the ROI, so we add the face's top-left corner (fx, fy)
                cv2.rectangle(roi_color, (eyes_x, eyes_y), (eyes_x + eyes_w, eyes_y + eyes_h), (0, 0, 255), 2)

            # Tag the image with text
            text = "this is me"
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Position text just below the face
            text_position = (face_x, face_y + face_h + 30)
            font_scale = 0.8
            font_color = (255, 255, 255)
            thickness = 2

            cv2.putText(resized_image, text, text_position, font, font_scale, font_color, thickness, cv2.LINE_AA)

        else:
            print("Warning: No face was detected in the picture.")

        # Save the final image
        cv2.imwrite('altered_picture.jpg', resized_image)

        # Display the final image for a few seconds
        cv2.imshow('Final Image', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error loading cascade files: {e}")
