import cv2
import numpy as np

def sharpen_image(image):
    # Define the sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])

    # Apply the kernel to the image
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image

# Load an image from file
image_path = "warpTesting/warpColored.jpg"
original_image = cv2.imread(image_path)

# Ensure the image is not None
if original_image is not None:
    # Convert the image to grayscale if it's a color image
    if len(original_image.shape) == 3:
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = original_image

    # Sharpen the image
    sharpened_image = sharpen_image(gray_image)

    # Display the original and sharpened images
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Sharpened Image", sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image not found.")
