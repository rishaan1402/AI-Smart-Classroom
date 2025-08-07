#!/usr/bin/env python3
"""
Create Demo Face Images
Generates synthetic face-like images for testing enrollment and identification
"""

import os
import cv2
import numpy as np

def create_demo_face(color, variation=0):
    """
    Create a synthetic face-like image
    
    Args:
        color: Base color tuple (B, G, R)
        variation: Variation factor for randomness
        
    Returns:
        Generated face image
    """
    # Create base image
    img = np.ones((200, 200, 3), dtype=np.uint8) * 50
    
    # Add some random background noise
    if variation > 0:
        noise = np.random.randint(-variation, variation, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Face outline (circle)
    face_center = (100, 100)
    face_radius = 80 + np.random.randint(-5, 5) if variation > 0 else 80
    cv2.circle(img, face_center, face_radius, color, -1)
    
    # Eyes
    eye_y = 80 + np.random.randint(-5, 5) if variation > 0 else 80
    left_eye = (80 + np.random.randint(-3, 3) if variation > 0 else 80, eye_y)
    right_eye = (120 + np.random.randint(-3, 3) if variation > 0 else 120, eye_y)
    
    eye_size = 10 + np.random.randint(-2, 2) if variation > 0 else 10
    cv2.circle(img, left_eye, eye_size, (0, 0, 0), -1)
    cv2.circle(img, right_eye, eye_size, (0, 0, 0), -1)
    
    # Nose
    nose_center = (100 + np.random.randint(-3, 3) if variation > 0 else 100, 
                   100 + np.random.randint(-3, 3) if variation > 0 else 100)
    nose_size = 5 + np.random.randint(-1, 1) if variation > 0 else 5
    cv2.circle(img, nose_center, nose_size, (50, 50, 50), -1)
    
    # Mouth
    mouth_center = (100 + np.random.randint(-3, 3) if variation > 0 else 100,
                    120 + np.random.randint(-3, 3) if variation > 0 else 120)
    mouth_width = 15 + np.random.randint(-2, 2) if variation > 0 else 15
    mouth_height = 8 + np.random.randint(-1, 1) if variation > 0 else 8
    cv2.ellipse(img, mouth_center, (mouth_width, mouth_height), 0, 0, 180, (0, 0, 0), -1)
    
    # Add some texture to make it more realistic
    if variation > 0:
        # Add some random lines for texture
        for _ in range(5):
            pt1 = (np.random.randint(0, 200), np.random.randint(0, 200))
            pt2 = (np.random.randint(0, 200), np.random.randint(0, 200))
            cv2.line(img, pt1, pt2, (color[0]//2, color[1]//2, color[2]//2), 1)
    
    return img

def create_demo_dataset():
    """Create a dataset of demo faces"""
    # Create output directory
    output_dir = "demo_faces"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define demo persons with different colors
    demo_persons = {
        "Alice": (150, 100, 100),    # Reddish
        "Bob": (100, 150, 100),      # Greenish  
        "Charlie": (100, 100, 150),  # Bluish
        "Diana": (150, 150, 100),    # Yellowish
    }
    
    print("Creating demo face dataset...")
    
    for person_name, base_color in demo_persons.items():
        person_dir = os.path.join(output_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Create 5 variations for each person
        for i in range(5):
            # Add some variation to the base color
            color_variation = 20
            color = (
                max(0, min(255, base_color[0] + np.random.randint(-color_variation, color_variation))),
                max(0, min(255, base_color[1] + np.random.randint(-color_variation, color_variation))),
                max(0, min(255, base_color[2] + np.random.randint(-color_variation, color_variation)))
            )
            
            # Create face image with variation
            face_img = create_demo_face(color, variation=10)
            
            # Save image
            filename = os.path.join(person_dir, f"{person_name}_{i+1}.jpg")
            cv2.imwrite(filename, face_img)
            
        print(f"Created {5} images for {person_name} in {person_dir}")
    
    print(f"\nDemo dataset created in '{output_dir}' directory")
    print("\nYou can now enroll these demo persons:")
    for person_name in demo_persons.keys():
        print(f"  python3 simple_enrollment.py enroll --name {person_name} --folder {output_dir}/{person_name}")
    
    return output_dir

if __name__ == "__main__":
    create_demo_dataset() 