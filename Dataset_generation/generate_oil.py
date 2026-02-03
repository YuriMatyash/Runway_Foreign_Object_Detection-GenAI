import os
import cv2
import torch
import random
import numpy as np
from PIL import Image, ImageDraw
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL

# ==========================================
#               CONFIGURATION
# ==========================================

OUTPUT_DIR = "./main/results"                    # Final dataset directory
OUTPUT_FILENAME_PREFIX = 'oil'
NUM_IMAGES = 200                                 # Total images to generate

YOLO_LABEL_INDEX = 0                             # Class ID for oil

PROMPT = (
    "raw aerial photograph, ultra-realistic satellite view of a weathered asphalt airport runway, "
    "bordered by lush green grass or deep brown dirt, realistic soil texture, "
    "granular heavy grain texture, tire skid marks, oil stains, deep black puddle, "
    "sharp focus, harsh daylight"
)

# UPDATED NEGATIVE PROMPT: Explicitly banning vehicles and aircraft
NEGATIVE_PROMPT = (
    "planes, aircraft, airplanes, vehicles, cars, trucks, buses, "
    "bluish grass, blue dirt, smooth, digital painting, cartoon, "
    "drawing, 3d render, blur, low res, clean, plastic"
)

AI_STRENGTH = 0.55
IMG_SIZE = 1024
# ==========================================
#              HELPER FUNCTIONS
# ==========================================

def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)

def add_noise_and_blur(img_cv):
    """ Adds grit to the OpenCV drawing so AI doesn't make it cartoonish. """
    img_blurred = cv2.GaussianBlur(img_cv, (3, 3), 0)
    noise = np.random.randint(0, 45, img_blurred.shape, dtype='uint8')
    img_noisy = cv2.addWeighted(img_blurred, 0.85, noise, 0.15, 0)
    return img_noisy

def get_ellipse_bbox(center, axes, angle):
    """ Calculates the axis-aligned bounding box for a rotated ellipse. """
    cx, cy = center
    a, b = axes 
    theta = np.radians(angle)
    ux, vx = a * np.cos(theta), b * np.sin(theta)
    uy, vy = a * np.sin(theta), b * np.cos(theta)
    
    # Calculate extents based on parametric equations
    width_ext = np.sqrt(ux**2 + vx**2)
    height_ext = np.sqrt(uy**2 + vy**2)

    return (cx - width_ext, cy - height_ext, cx + width_ext, cy + height_ext)

def save_yolo_label(path, bbox, img_w, img_h, class_id):
    """ Saves a .txt file with normalized YOLO format. """
    min_x, min_y, max_x, max_y = bbox
    w, h = max_x - min_x, max_y - min_y
    center_x = min_x + (w / 2)
    center_y = min_y + (h / 2)
    
    # Normalize and clamp to [0, 1]
    norm_cx = min(max(center_x / img_w, 0), 1)
    norm_cy = min(max(center_y / img_h, 0), 1)
    norm_w = min(max(w / img_w, 0), 1)
    norm_h = min(max(h / img_h, 0), 1)
    
    line = f"{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n"
    with open(path, 'w') as f:
        f.write(line)

# ==========================================
#            VARIATION LOGIC
# ==========================================

def draw_burnt_rubber(img, w, h, left_bound, right_bound):
    """ Simulates tire skid marks within the asphalt boundaries. """
    overlay = img.copy()
    num_skids = random.randint(5, 15)
    for _ in range(num_skids):
        start_x = random.randint(left_bound, right_bound)
        end_x = start_x + random.randint(-20, 20)
        cv2.line(overlay, (start_x, 0), (end_x, h), (10, 10, 10), random.randint(2, 8))
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

def create_base_layout(w=1024, h=1024):
    """ Generates a randomized runway layout with shoulders and markings. """
    # 1. Determine Shoulder Widths (5-15% on each side)
    left_perc = random.uniform(0.05, 0.15)
    right_perc = random.uniform(0.05, 0.15)
    left_bound = int(w * left_perc)
    right_bound = int(w * (1.0 - right_perc))
    
    # 2. Base Asphalt
    base_val = random.randint(50, 90)
    img_base = np.full((h, w, 3), (base_val, base_val, base_val), dtype=np.uint8)
    
    # 3. Randomized Shoulder Color (Green or Earthy Brown)
    if random.random() > 0.5:
        shoulder_color = (random.randint(30, 60), random.randint(70, 100), random.randint(20, 40))
    else:
        shoulder_color = (random.randint(80, 120), random.randint(70, 100), random.randint(30, 60))
        
    cv2.rectangle(img_base, (0, 0), (left_bound, h), shoulder_color, -1)
    cv2.rectangle(img_base, (right_bound, 0), (w, h), shoulder_color, -1)
    
    # 4. Randomized Markings
    mark_type = random.choice(['white', 'yellow', 'none'])
    if mark_type != 'none':
        if mark_type == 'white': mark_color = (random.randint(200, 230),)*3
        else: mark_color = (random.randint(180, 210), random.randint(160, 190), 30)
        
        dash_w, dash_h = int(w * 0.03), int(h * 0.15)
        asphalt_center = left_bound + (right_bound - left_bound) // 2
        
        if random.random() > 0.5: # Centerline
            cv2.rectangle(img_base, (asphalt_center - dash_w//2, int(h*0.1)), (asphalt_center + dash_w//2, int(h*0.1)+dash_h), mark_color, -1)
            cv2.rectangle(img_base, (asphalt_center - dash_w//2, int(h*0.7)), (asphalt_center + dash_w//2, int(h*0.7)+dash_h), mark_color, -1)
        else: # Side Line near shoulder
            cv2.line(img_base, (left_bound + int(w*0.02), 0), (left_bound + int(w*0.02), h), mark_color, int(w*0.01))

    # 5. Tire Skids
    img_base = draw_burnt_rubber(img_base, w, h, left_bound, right_bound)

    # 6. Oil Blob
    blob_w, blob_h = random.randint(w//12, w//8), random.randint(h//12, h//8)
    center_x = random.randint(left_bound + blob_w, right_bound - blob_w)
    center_y = random.randint(h//4, 3*h//4)
    angle = random.randint(0, 180)
    
    puddle_color = (random.randint(5, 15), random.randint(5, 15), random.randint(5, 15))
    cv2.ellipse(img_base, (center_x, center_y), (blob_w, blob_h), angle, 0, 360, puddle_color, -1)
    
    bbox = get_ellipse_bbox((center_x, center_y), (blob_w, blob_h), angle)
    img_gritty = add_noise_and_blur(img_base)
    
    return Image.fromarray(cv2.cvtColor(img_gritty, cv2.COLOR_BGR2RGB)), bbox

# ==========================================

def main():
    ensure_dir(OUTPUT_DIR)
    
    print("⏳ Loading SDXL and Stable VAE...")
    # Use Stable VAE fix for FP16 stability
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    
    # Optimizations for performance
    pipe.enable_model_cpu_offload() 
    pipe.enable_attention_slicing()

    print(f"✅ Ready. Generating {NUM_IMAGES} images with variety...")

    for i in range(NUM_IMAGES):
        print(f"[{i+1}/{NUM_IMAGES}] Generative Step...")
        
        # Get randomized base and bbox
        base_image, bbox = create_base_layout(IMG_SIZE, IMG_SIZE)
        
        with torch.inference_mode(), torch.autocast("cuda"):
            final_image = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=base_image,
                strength=AI_STRENGTH,
                num_inference_steps=35,
                guidance_scale=13.0               # Higher for prompt adherence
            ).images[0]

        filename_base = f"{OUTPUT_FILENAME_PREFIX}_{i+1:03d}"
        save_path_img = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
        save_path_txt = os.path.join(OUTPUT_DIR, f"{filename_base}.txt")
        
        # Save results
        final_image.save(save_path_img)
        save_yolo_label(save_path_txt, bbox, IMG_SIZE, IMG_SIZE, YOLO_LABEL_INDEX)
        
        print(f"Saved: {filename_base}")
        
    print(f"✅ Done. Dataset ready in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()