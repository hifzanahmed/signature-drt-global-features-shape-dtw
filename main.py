from training.signature_training import SignatureTraining
from verification.signature_verification import SignatureVerificationTraining
import os

def main():
    location_of_training_signature = 'C:/Users/hifza/workspace/Signature Dataset/signatures_1/original_1_'
    size_of_training_signature = 6
    location_of_test_signature = 'C:/Users/hifza/workspace/Signature Dataset/signatures_1/original_1_'
    # Training Phase
    s1 = SignatureTraining.training_genuine_with_shape_dtw(location_of_training_signature, size_of_training_signature)  
    # Verification Phase of input test signature 
    i = 1
    while False:
        test_signature = f"{location_of_test_signature}{i}.png"
        test_signature_path = os.path.join(test_signature)
    
        # Stop if the image does not exist
        if not os.path.exists(test_signature_path):
            break
        s2 = SignatureVerificationTraining.verifiy_test_signature_with_shape_dtw(test_signature_path)
        # Decision Making: calculating the score and comparing it with a threshold value
        i += 1
        score_ratio = abs(s2) / abs(s1)
        if score_ratio > 1:  
            print(f"{score_ratio};Genuine")
        else:
            print(f"{score_ratio};Forged")

if __name__ == "__main__":
    main()