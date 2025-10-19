# Todd Bartoszkiewicz
# CSC515: Foundations of Computer Vision
# Module 2 Option 1 Critical Thinking
import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('shutterstock215592034--250.jpg')

    print('Shape of the image: {}'.format(img.shape))
    print('Image Height: {}'.format(img.shape[0]))
    print('Image Width: {}'.format(img.shape[1]))
    print('Image Dimension: {}'.format(img.ndim))

    print('Value of only the B channel {}'.format(img[100, 50, 0]))
    print('Value of only the G channel {}'.format(img[100, 50, 1]))
    print('Value of only the R channel {}'.format(img[100, 50, 2]))

    b_channel, g_channel, r_channel = cv2.split(img)
    print(f"\nShape of blue channel matrix: {b_channel.shape}")
    print(f"\nShape of green channel matrix: {g_channel.shape}")
    print(f"\nShape of red channel matrix: {r_channel.shape}")

    cv2.imshow("Blue Channel (as Grayscale)", b_channel)
    cv2.imshow("Green Channel (as Grayscale)", g_channel)
    cv2.imshow("Red Channel (as Grayscale)", r_channel)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for c, ax in zip(range(3), ax):
        # create zeros matrix
        split_img = np.zeros(img.shape, dtype='uint8')
        # access each channel
        split_img[:, :, c] = img[:, :, c]
        # print each channel
        # print(split_img)
        print('Shape of the image: {}'.format(split_img))
        # display each channel
        ax.imshow(split_img)
        cv2.imwrite('split_img_' + str(c) + '.jpg', split_img)

    plt.show()

    # Step 2: Extract individual color channels (B, G, R)
    blue_channel = img[:, :, 0]  # Blue channel (1st channel in BGR)
    green_channel = img[:, :, 1]  # Green channel (2nd channel in BGR)
    red_channel = img[:, :, 2]  # Red channel (3rd channel in BGR)

    # Step 3: Merge channels back to BGR (original order)
    merged_bgr = cv2.merge((blue_channel, green_channel, red_channel))

    # Step 4: Merge channels with swapped R and G (GRB order)
    swapped_channels = (green_channel, red_channel, blue_channel)
    merged_grb = cv2.merge(swapped_channels)

    # Step 5: Display all images
    plt.figure(figsize=(15, 10))

    # Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    # Blue Channel (grayscale)
    plt.subplot(2, 2, 2)
    plt.imshow(blue_channel, cmap="gray")
    plt.title("Blue Channel")
    plt.axis("off")

    # Green Channel (grayscale)
    plt.subplot(2, 2, 3)
    plt.imshow(green_channel, cmap="gray")
    plt.title("Green Channel")
    plt.axis("off")

    # Red Channel (grayscale)
    plt.subplot(2, 2, 4)
    plt.imshow(red_channel, cmap="gray")
    plt.title("Red Channel")
    plt.axis("off")

    # Merged BGR (Original)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(merged_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Merged BGR (Original)")
    plt.axis("off")

    # Merged GRB (Swapped R and G)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(merged_grb, cv2.COLOR_BGR2RGB))
    plt.title("Merged GRB (Swapped R and G)")
    plt.axis("off")

    plt.show()

    # cv2.imshow('image_window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
