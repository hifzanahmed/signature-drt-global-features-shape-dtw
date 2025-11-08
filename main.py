from training.signature_training import SignatureTraining
from verification.signature_verification import SignatureVerificationTraining
import os

def main():
    location_of_training_signature = 'C:/Users/hifza/workspace/Signature Dataset/signatures_4/original_4_'
    size_of_training_signature = 6
    location_of_test_signature = 'C:/Users/hifza/workspace/Signature Dataset/signatures_4/forgeries_4_'
    # Training Phase
    s1 = SignatureTraining.training_genuine_with_shape_dtw(location_of_training_signature, size_of_training_signature)  
    # Verification Phase of input test signature 
    i = 1
    while True:
        test_signature = f"{location_of_test_signature}{i}.png"
        test_signature_path = os.path.join(test_signature)
    
        # Stop if the image does not exist
        if not os.path.exists(test_signature_path):
            break
        s2 = SignatureVerificationTraining.verifiy_test_signature_with_shape_dtw(test_signature_path)
        i += 1
        # Decision Making: calculating the score and comparing it with a threshold value
        score = s2 / s1
        if score <= 1.15:
            print(f"{score};Genuine")
        else:
            print(f"{score};Forged")

if __name__ == "__main__":
    main()