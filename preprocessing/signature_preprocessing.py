from utils.utilities import Utilities
import cv2

class ImageProcessor:
    def read_and_preprocess(location):
        # Load an image from the specified location
        utils = Utilities()
        signature = utils.load_image(location)

        # Check if image is loaded correctly
        if signature is None:
            return None
        else:
            # Step 1: Display the original image
            # Step 2: Convert to grayscale (Otsu needs grayscale input)
            img_gray = cv2.cvtColor(signature, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('Converted Signature in grayscale', img_gray)
            # Wait until any key is pressed
            #cv2.waitKey(0)
            # Close the window
            #cv2.destroyAllWindows()

            # Step 3: Apply Otsu's thresholding to find threshold and convert to binary
            threshold_value, img_binary = cv2.threshold(
                img_gray, 
                0,                   # Threshold ignored for Otsu
                255,                 # Max value for the binary image
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            #print(f"Optimal threshold found by Otsu's method: {threshold_value}")

            # Step 4: img_binary is your Otsu thresholded image (0 and 255 pixels)
            img_inverted = cv2.bitwise_not(img_binary)
            #cv2.imshow('Converted Signature in binary', img_inverted)
            # Wait until any key is pressed
            #cv2.waitKey(0)
            # Close the window
            #cv2.destroyAllWindows()

            # Step 5: Crop the image to remove unnecessary white space
            img_cropped = utils.crop_and_resize_signature(img_inverted)
            #cv2.imshow('Cropped Signature', img_cropped)   
            # Wait until any key is pressed
            #cv2.waitKey(0)
            # Close the window
            #cv2.destroyAllWindows()
            # Median Filter - Assuming `binary_img` is your binary image array (with 0 and 255)
            filtered_img = cv2.medianBlur(img_cropped, 3)  # kernel size can be 3, 5, 7, etc.
            #cv2.imshow('Filtered Signature', filtered_img)   
            # Wait until any key is pressed
            #cv2.waitKey(0)
            # Close the window
            #cv2.destroyAllWindows()
            return filtered_img