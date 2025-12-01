import cv2
import numpy as np
import os
import glob
from geometry_classes import get_scene_planes
from tqdm import tqdm 
from PIL import Image
from collections import deque

# ==========================================
# 1. CONFIGURATION
# ==========================================
# WORKING RESOLUTION - Resize everything to this
TARGET_WIDTH = 1920  # Change this to your desired resolution
# TARGET_WIDTH = 1280  # Or use this for faster processing

# Original coordinates (at full resolution)
VP_ORIGINAL = (2825, 5185)
BACK_TL_ORIGINAL = (2399, 4600)
BACK_BR_ORIGINAL = (3304, 5358)

# Files 
IMG_PATH = 'images/iiit.jpg'
CODED_MAP_PATH = 'coded_id_map_iiit.npy'
ALPHA_MASK_PATH = 'images/mask_iiit.jpeg' 
SKY_FOLDER = 'sky2' 

OUTPUT_FOLDER = 'iiit'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- SIMPLE SETTINGS ---
IMAGE_SELECTION_STEP = 2
SHADOW_DARKNESS = 0.4
SHADOW_BLUR = 21
FLOOR_SAMPLE_RATE = 1

# --- OUTPUT ---
GIF_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, 'animation.gif')
VIDEO_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, 'animation.mp4')
GIF_DURATION = 100
VIDEO_FPS = 24

# ==========================================
# 2. LOAD AND RESIZE ALL INPUTS
# ==========================================
print("--- LOADING AND RESIZING INPUTS ---")

# Load original base image
base_img_original = cv2.imread(IMG_PATH).astype(np.float32) / 255.0
h_orig, w_orig = base_img_original.shape[:2]
print(f"Original image size: {w_orig}x{h_orig}")

# Calculate scale factor
scale = TARGET_WIDTH / w_orig
h_target = int(h_orig * scale)
print(f"Target size: {TARGET_WIDTH}x{h_target} (scale: {scale:.3f})")

# Resize base image
base_img = cv2.resize(base_img_original, (TARGET_WIDTH, h_target), interpolation=cv2.INTER_LINEAR)
h, w = base_img.shape[:2]

# Resize alpha mask
alpha_mask_original = cv2.imread(ALPHA_MASK_PATH, 0)
alpha_mask = cv2.resize(alpha_mask_original, (TARGET_WIDTH, h_target), interpolation=cv2.INTER_LINEAR)
alpha_mask = 255 - alpha_mask  # Invert
alpha_mask = alpha_mask.astype(np.float32) / 255.0

# Resize coded map
coded_map_original = np.load(CODED_MAP_PATH)
coded_map = cv2.resize(coded_map_original, (TARGET_WIDTH, h_target), interpolation=cv2.INTER_NEAREST)

# Scale geometry coordinates
VP = (int(VP_ORIGINAL[0] * scale), int(VP_ORIGINAL[1] * scale))
BACK_TL = (int(BACK_TL_ORIGINAL[0] * scale), int(BACK_TL_ORIGINAL[1] * scale))
BACK_BR = (int(BACK_BR_ORIGINAL[0] * scale), int(BACK_BR_ORIGINAL[1] * scale))

print(f"Scaled VP: {VP}")
print(f"Scaled BACK_TL: {BACK_TL}")
print(f"Scaled BACK_BR: {BACK_BR}")

# Get scene planes with scaled coordinates
planes, f, (cx, cy) = get_scene_planes(w, h, VP, BACK_TL, BACK_BR)

print(f"Focal length: {f:.2f}")
print(f"Principal point: ({cx}, {cy})")

# ==========================================
# 3. PRECOMPUTE FLOOR 3D POSITIONS
# ==========================================
print("\n--- Precomputing floor 3D ---")

camera_origin = np.array([0, 0, 0], dtype=np.float32)
floor_pixels_3d = {}
floor_coords = np.argwhere(coded_map == 3)[::FLOOR_SAMPLE_RATE]

for py, px in tqdm(floor_coords, desc="Floor 3D", leave=False):
    pixel_ray = np.array([px - cx, py - cy, f], dtype=np.float32)
    pixel_ray = pixel_ray / np.linalg.norm(pixel_ray)
    
    t_hit = planes['floor'].intersect_ray(camera_origin, pixel_ray)
    if t_hit and t_hit > 0:
        floor_pixels_3d[(py, px)] = camera_origin + t_hit * pixel_ray

print(f"Computed {len(floor_pixels_3d)} floor points")

# ==========================================
# 4. NORMALS
# ==========================================
normal_map = np.zeros((h, w, 3), dtype=np.float32)
normal_map[coded_map == 3] = [0, -1, 0]
normal_map[coded_map == 1] = [1, 0, 0]
normal_map[coded_map == 2] = [-1, 0, 0]
normal_map[coded_map == 4] = [0, 0, 1]

# ==========================================
# 5. FUNCTIONS
# ==========================================
def extract_sun_direction(sky_img):
    """Extract sun direction from equirectangular sky"""
    sky_h, sky_w = sky_img.shape[:2]
    gray = np.mean(sky_img, axis=2)
    
    _, _, _, max_loc = cv2.minMaxLoc(gray)
    sun_u, sun_v = max_loc
    
    phi = (sun_u / sky_w) * 2 * np.pi
    theta = (sun_v / sky_h) * np.pi
    
    sun_x = np.sin(theta) * np.cos(phi)
    sun_y = np.cos(theta)
    sun_z = np.sin(theta) * np.sin(phi)
    
    sun_dir = np.array([sun_x, sun_y, sun_z], dtype=np.float32)
    return sun_dir / np.linalg.norm(sun_dir)

def compute_shadows(floor_pixels_3d, sun_dir, planes, alpha_mask, coded_map, h, w, f, cx, cy):
    """Shadow raytracing with projective mask lookup"""
    shadow_map = np.zeros((h, w), dtype=np.float32)
    
    occluders = [planes['left'], planes['right'], planes['back']]
    BIAS = 2.0
    MIN_Z = 1.0
    
    for (py, px), P_floor in tqdm(floor_pixels_3d.items(), desc="  Shadows", leave=False):
        P_start = P_floor + sun_dir * BIAS
        is_shadowed = False
        
        for plane in occluders:
            t = plane.intersect_ray(P_start, sun_dir)
            
            if t is None or t <= 0.01 or not np.isfinite(t):
                continue
            
            P_hit = P_start + t * sun_dir
            
            if not np.all(np.isfinite(P_hit)):
                continue
            
            if abs(P_hit[2]) < MIN_Z:
                continue
            
            u = (P_hit[0] / P_hit[2]) * f + cx
            v = (P_hit[1] / P_hit[2]) * f + cy
            
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            
            u_int = int(round(u))
            v_int = int(round(v))
            
            if 0 <= u_int < w and 0 <= v_int < h:
                if alpha_mask[v_int, u_int] > 0.5:
                    is_shadowed = True
                    break
            else:
                is_shadowed = True
                break
        
        if is_shadowed:
            shadow_map[py, px] = 1.0
    
    if SHADOW_BLUR > 0:
        s = SHADOW_BLUR if SHADOW_BLUR % 2 == 1 else SHADOW_BLUR + 1
        shadow_map = cv2.GaussianBlur(shadow_map, (s, s), 0)
    
    return shadow_map * (coded_map == 3)

# ==========================================
# 6. PROCESS FRAMES
# ==========================================
sky_files = glob.glob(os.path.join(SKY_FOLDER, "*.*"))
valid_exts = ['.jpg', '.jpeg', '.png', '.hdr', '.exr']
sky_files = [f for f in sky_files if os.path.splitext(f)[1].lower() in valid_exts]

try:
    sky_files = sorted(sky_files, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
except:
    sky_files = sorted(sky_files)

sky_files = sky_files[::IMAGE_SELECTION_STEP]

print(f"\nFound {len(sky_files)} sky images")
print(f"Processing at {TARGET_WIDTH}x{h_target}\n")

processed_images = []
processed_frames_bgr = []

for idx, sky_path in enumerate(sky_files):
    print(f"[{idx+1}/{len(sky_files)}] {os.path.basename(sky_path)}")
    
    sky = cv2.imread(sky_path, cv2.IMREAD_UNCHANGED)
    if sky is None: 
        print("  ⚠ Failed to load, skipping")
        continue
    
    if len(sky.shape) == 2:
        sky = cv2.cvtColor(sky, cv2.COLOR_GRAY2BGR)
    elif sky.shape[2] == 4:
        sky = sky[:, :, :3]
    
    sky = cv2.cvtColor(sky, cv2.COLOR_BGR2RGB).astype(np.float32)
    if sky.max() > 1.0: 
        sky = sky / 255.0
    
    # Resize sky to match working resolution
    sky_resized = cv2.resize(sky, (w, h), interpolation=cv2.INTER_LINEAR)
    
    sun_dir = extract_sun_direction(sky)
    sun_dir[1] = -sun_dir[1]
    
    print(f"  Sun: [{sun_dir[0]:.2f}, {sun_dir[1]:.2f}, {sun_dir[2]:.2f}]")
    
    shadow_map = compute_shadows(
        floor_pixels_3d, sun_dir, planes, alpha_mask, coded_map, h, w, f, cx, cy
    )
    
    # Lighting
    sun_dir_broadcast = sun_dir.reshape(1, 1, 3)
    n_dot_l = np.sum(normal_map * sun_dir_broadcast, axis=2)
    n_dot_l = np.clip(n_dot_l, 0, 1)
    
    shadow_factor = np.where(shadow_map > 0.5, SHADOW_DARKNESS, 1.0)
    light_factor = n_dot_l * shadow_factor
    lit_image = base_img * light_factor[:, :, None]
    
    # Composite
    building_mask = (alpha_mask > 0.5)[:, :, None]
    final = lit_image * building_mask
    final = final + sky_resized * (1 - building_mask)
    
    final_uint8 = (np.clip(final, 0, 1) * 255).astype(np.uint8)
    
    processed_images.append(Image.fromarray(final_uint8))
    processed_frames_bgr.append(cv2.cvtColor(final_uint8, cv2.COLOR_RGB2BGR))
    
    print(f"  ✓ Done")

# ==========================================
# 7. SAVE
# ==========================================
if processed_images:
    print(f"\n{'='*50}")
    print(f"Saving {len(processed_images)} frames...")
    print(f"{'='*50}\n")
    
    processed_images[0].save(
        GIF_OUTPUT_PATH, save_all=True, append_images=processed_images[1:],
        duration=GIF_DURATION, loop=0, optimize=False
    )
    print(f"✓ GIF saved: {GIF_OUTPUT_PATH}")
    
    h_out, w_out = processed_frames_bgr[0].shape[:2]
    out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (w_out, h_out))
    for frame in processed_frames_bgr:
        out.write(frame)
    out.release()
    print(f"✓ Video saved: {VIDEO_OUTPUT_PATH}")
    
    print("\nDONE! 🎬")
else:
    print("\n⚠ No frames processed!")