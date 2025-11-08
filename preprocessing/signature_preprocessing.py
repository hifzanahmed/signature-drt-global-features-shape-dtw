from utils.utilities import Utilities
import numpy as np
import cv2

class ImageProcessor:
    def read_and_preprocess(image_path, enhance_contrast=True, apply_median=True,):
        """
        Complete preprocessing pipeline for signature images.
        Steps:
            1. Load image
            2. Convert to grayscale
            3. Optional CLAHE contrast enhancement
            4. Normalize to [0,1]
            5. Crop and resize while preserving aspect ratio
            6. Optional median filtering
        Returns:
            np.ndarray: Preprocessed image as float32 in [0,1]
        """
        utils = Utilities()
    
        # Step 1: Load image
        signature = utils.load_image(image_path)
        if signature is None:
            return None
        # Step 2: Convert to grayscale
        img_gray = cv2.cvtColor(signature, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('Converted Signature in grayscale', img_gray)

        # Step 3: Optional contrast enhancement (CLAHE)
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_gray = clahe.apply(img_gray)
        # Step 4: Normalize to [0,1]
        img_gray = img_gray.astype(np.float32) / 255.0
        # Step 5: Crop and resize with aspect ratio preserved
        img_cropped = utils.crop_and_resize_signature(img_gray)
    
        # Step 6: Optional median filter to reduce noise
        if apply_median:
            img_filtered = cv2.medianBlur((img_cropped * 255).astype(np.uint8), 3)
            img_filtered = img_filtered.astype(np.float32) / 255.0
        else:
            img_filtered = img_cropped
    
        #cv2.imshow('Converted Signature in grayscale', img_filtered)
        # Wait until any key is pressed
        #cv2.waitKey(0)
        # Close the window
        #cv2.destroyAllWindows()
        return img_filtered