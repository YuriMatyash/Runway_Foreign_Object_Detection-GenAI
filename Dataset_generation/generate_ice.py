import os
import cv2
import torch
import random
import numpy as np
import math
from PIL import Image, ImageDraw
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL

# ==========================================
#               CONFIGURATION
# ==========================================

OUTPUT_DIR = "./main/results"
OUTPUT_FILENAME_PREFIX = 'ice'
NUM_IMAGES = 200

YOLO_LABEL_INDEX = 1  # Class ID for Ice

# ICE PROMPT: Specific satellite/winter context
PROMPT = (
    "raw aerial photograph, ultra-realistic satellite view of a weathered asphalt airport runway, "
    "bordered by lush green grass and deep brown dirt, realistic soil texture, "
    "a large solid chunk of jagged ice sitting on the asphalt, frozen block, "
    "translucent white and blue ice, granular heavy grain texture, tire skid marks, "
    "sharp focus, harsh daylight, cold winter lighting"
)

NEGATIVE_PROMPT = (
    "planes, aircraft, airplanes, vehicles, cars, trucks, buses, "
    "bluish grass, blue dirt, oil, puddle, melted, liquid water, "
    "smooth, digital painting, cartoon, drawing, 3d render, blur, low res, clean"
)

AI_STRENGTH = 0.60
IMG_SIZE = 1024  # SDXL native resolution

# ==========================================
#              HELPER FUNCTIONS
# ==========================================

def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)

def add_noise_and_blur(img_cv):
    img_blurred = cv2.GaussianBlur(img_cv, (3, 3), 0)
    noise = np.random.randint(0, 45, img_blurred.shape, dtype='uint8')
    img_noisy = cv2.addWeighted(img_blurred, 0.85, noise, 0.15, 0)
    return img_noisy

def get_polygon_bbox(pts):
    """ Calculates the axis-aligned bounding box from polygon points. """
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    return (np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords))

def save_yolo_label(path, bbox, img_w, img_h, class_id):
    min_x, min_y, max_x, max_y = bbox
    w, h = max_x - min_x, max_y - min_y
    center_x = min_x + (w / 2)
    center_y = min_y + (h / 2)
    
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
    overlay = img.copy()
    num_skids = random.randint(5, 15)
    for _ in range(num_skids):
        start_x = random.randint(left_bound, right_bound)
        end_x = start_x + random.randint(-20, 20)
        cv2.line(overlay, (start_x, 0), (end_x, h), (10, 10, 10), random.randint(2, 8))
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

def draw_ice_chunk(img, left_bound, right_bound, h):
    """ Draws a blocky, jagged ice chunk proxy. """
    # Scale radius for 1024px
    base_radius = random.randint(50, 90)
    center_x = random.randint(left_bound + base_radius, right_bound - base_radius)
    center_y = random.randint(int(h * 0.3), int(h * 0.7))
    
    num_points = 10 
    points = []
    for i in range(num_points):
        angle = (2 * math.pi * i) / num_points
        r = base_radius * random.uniform(0.6, 1.5)
        x = int(center_x + r * math.cos(angle))
        y = int(center_y + r * math.sin(angle))
        points.append([x, y])

    pts = np.array(points, np.int32)
    # Light blue/white ice color
    ice_color = (255, 250, 240) 
    cv2.fillPoly(img, [pts], ice_color)
    cv2.polylines(img, [pts], True, (200, 200, 220), 2)
    
    return get_polygon_bbox(pts)

def create_base_layout(w=1024, h=1024):
    # 1. Shoulder Logic (5-15%)
    left_perc, right_perc = random.uniform(0.05, 0.15), random.uniform(0.05, 0.15)
    l_bound, r_bound = int(w * left_perc), int(w * (1.0 - right_perc))
    
    # 2. Base Asphalt
    base_val = random.randint(50, 90)
    img = np.full((h, w, 3), (base_val, base_val, base_val), dtype=np.uint8)
    
    # 3. Shoulder Colors (Green/Brown)
    if random.random() > 0.5:
        shoulder_color = (random.randint(30, 60), random.randint(70, 100), random.randint(20, 40))
    else:
        shoulder_color = (random.randint(80, 120), random.randint(70, 100), random.randint(30, 60))
    cv2.rectangle(img, (0, 0), (l_bound, h), shoulder_color, -1)
    cv2.rectangle(img, (r_bound, 0), (w, h), shoulder_color, -1)
    
    # 4. Markings
    mark_type = random.choice(['white', 'yellow', 'none'])
    if mark_type != 'none':
        mark_color = (210, 210, 210) if mark_type == 'white' else (180, 160, 30)
        dash_w, dash_h = int(w * 0.03), int(h * 0.15)
        center_x = l_bound + (r_bound - l_bound) // 2
        cv2.rectangle(img, (center_x - dash_w//2, int(h*0.1)), (center_x + dash_w//2, int(h*0.25)), mark_color, -1)
        cv2.rectangle(img, (center_x - dash_w//2, int(h*0.7)), (center_x + dash_w//2, int(h*0.85)), mark_color, -1)

    # 5. Tire Skids
    img = draw_burnt_rubber(img, w, h, l_bound, r_bound)

    # 6. Ice Chunk
    bbox = draw_ice_chunk(img, l_bound, r_bound, h)
    
    # 7. Realism noise
    img_gritty = add_noise_and_blur(img)
    
    return Image.fromarray(cv2.cvtColor(img_gritty, cv2.COLOR_BGR2RGB)), bbox

# ==========================================

def main():
    ensure_dir(OUTPUT_DIR)
    
    print("⏳ Loading SDXL and Stable VAE...")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    
    pipe.enable_model_cpu_offload() 
    pipe.enable_attention_slicing()

    print(f"✅ Ready. Generating {NUM_IMAGES} Ice Chunk images...")

    for i in range(NUM_IMAGES):
        print(f"[{i+1}/{NUM_IMAGES}] Generative Step...")
        base_image, bbox = create_base_layout(IMG_SIZE, IMG_SIZE)
        
        with torch.inference_mode(), torch.autocast("cuda"):
            final_image = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=base_image,
                strength=AI_STRENGTH,
                num_inference_steps=35,
                guidance_scale=13.0
            ).images[0]

        filename_base = f"{OUTPUT_FILENAME_PREFIX}_{i+1:03d}"
        save_path_img = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
        save_path_txt = os.path.join(OUTPUT_DIR, f"{filename_base}.txt")
        
        final_image.save(save_path_img)
        save_yolo_label(save_path_txt, bbox, IMG_SIZE, IMG_SIZE, YOLO_LABEL_INDEX)
        
        print(f"Saved: {filename_base}")
    
    print(f"✅ Done. Ice dataset ready in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()