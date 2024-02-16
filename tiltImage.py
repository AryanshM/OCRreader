import cv2
import numpy as np

def deskew_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve Hough Line Transform
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Hough Line Transform to detect lines in the image
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    # Calculate the average angle of detected lines
    angles = []
    for line in lines:
        for rho, theta in line:
            angles.append(theta)

    average_angle = np.mean(angles)

    # Rotate the image to deskew it
    rotated_image = ndimage.rotate(img, np.degrees(average_angle))

    # Display the original and deskewed images
    cv2.imshow("Original Image", img)
    cv2.imshow("Deskewed Image", rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = '/tilted.png'
deskew_image(image_path)
