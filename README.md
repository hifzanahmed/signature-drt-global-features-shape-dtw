# signature-drt-global-features-shape-dtw
Static signature recognition using DRT and ShapeDTW

pip install opencv-python 
pip install numpy
pip install shapedtw

Step 1: Preprocess Images
Convert images to grayscale.
Resize to a common size if needed (e.g., 128×128). 300*150 would be preferred our size as we have less data to train.

Step 2: Extract DRT (Discrete Radon Transform) Features
Decide on angles: e.g., 10–20 angles from 0° to 180°. Since you have only 6 training images, I suggest starting with around 12‑18 angles (e.g., every 10° from 0° to 170° → 18 angles) and see if results are stable.
You can later try increasing to ~30 angles (every 6°) and check if performance improves significantly — if not, stick with the simpler version.
Flatten or normalize projections if needed.
At this stage, you have global features for each image, one sequence per projection angle

Step 3: Compute Local Shape Descriptors for ShapeDTW
For ShapeDTW, each point in the projection sequence should have a local descriptor, e.g., a small window of surrounding values.

Step 4: Compute ShapeDTW Distances
For each pair of images (or query vs templates), compute ShapeDTW distance between their DRT sequences.
This gives a robust similarity score accounting for local shape variations.

Step 5: Classification / Matching
With 6 images, you likely do nearest-neighbor matching:
Compute ShapeDTW distance from query image to each template.
Assign class of closest template.
Optionally, augment your templates with rotations / flips to improve robustness.

Step 6: Optional — Rotation Augmentation
Rotate each image by multiples of 45° or 90°.
Compute DRT for each rotated image.

Summary:
Input image → preprocess → DRT (global projections) → local descriptors → ShapeDTW distance
Global features: from DRT.
Local alignment: ShapeDTW.
