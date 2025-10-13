# Todd Bartoszkiewicz
# CSC515: Foundations of Computer Vision
# Module 1: Portfolio Milestone Option #2
import cv2
import tkinter as tk
from tkinter import filedialog

if __name__ == '__main__':
    img = cv2.imread('shutterstock130285649--250.jpg')
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("JPG files", "*.jpg")],
        title="Save File As"
    )

    if file_path:
        try:
            cv2.imwrite(file_path, img)
            print(f"File saved to: {file_path}")
        except IOError as ioe:
            print(f"Error saving file: {ioe}")
    else:
        cv2.imwrite('numbers.jpg', img)

    cv2.imshow('image_window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
