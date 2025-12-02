import cv2
import numpy as np
import os
import sys
import glob
from tqdm import tqdm
from PIL import Image 

# ==========================================
# 0. SETUP
# ==========================================

try:
    from geometry_classes import get_scene_planes
except ImportError:
    print("❌ Error: 'geometry_classes.py' not found.")
    sys.exit(1)

# ==========================================
# 1. CONFIGURATION
# ==========================================
# VP = (427, 393)
# BACK_TL = (362, 322)
# BACK_BR = (509, 404)

VP = (283,568)
BACK_BR = (356,590)
BACK_TL = (232,462)



IMG_PATH = 'images/research_street.jpg'
CODED_MAP_PATH = 'coded_id_map_research.npy'
ALPHA_MASK_PATH = 'images/research_street_mask_inverted.png' 
# FOLDER CONTAINING REAL SKY IMAGES
SKY_FOLDER = 'frames_latlongvideo1' 

OUTPUT_FOLDER = 'timelapse_real_sky_output_research'

# --- TIMELAPSE SETTINGS ---
NUM_FRAMES = 291         
PEAK_ELEVATION = 70      
ARC_START_AZIMUTH = -110 
ARC_END_AZIMUTH = 110    

# --- QUALITY SETTINGS ---
SHADOW_OPACITY = 0.90
SHADOW_BLUR = 15         
SAMPLE_RATE = 2          

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# 2. HELPER FUNCTIONS (Geometry & Color)
# ==========================================
def get_sun_vector(azimuth_deg, elevation_deg):
    """Calculates Sun Position based on angles."""
    phi = np.radians(azimuth_deg)     
    theta = np.radians(elevation_deg) 
    
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta)
    z = np.cos(theta) * np.cos(phi)
    
    sun_dir = np.array([x, y, z], dtype=np.float32)
    sun_dir = sun_dir / np.linalg.norm(sun_dir)
    sun_dir[1] = -abs(sun_dir[1]) 
    return sun_dir

def get_daylight_color(elevation):
    """Returns color (R,G,B) and Intensity based on sun height."""
    color_horizon = np.array([1.0, 0.55, 0.2]) # Deep Orange
    color_noon    = np.array([1.0, 1.0, 1.0])  # White
    
    factor = np.clip(elevation / 40.0, 0.0, 1.0)
    current_color = (color_horizon * (1.0 - factor)) + (color_noon * factor)
    
    # Intensity Curve: 1.0x at sunrise -> 3.0x at noon
    intensity = 1.0 + (2.0 * np.sin(factor * np.pi / 2)) 
    
    return current_color, intensity

def compute_full_scene_shadows(scene_pixels_3d, sun_dir, planes, alpha_mask, h, w, f, cx, cy):
    """Raytraces from ANY scene point (Floor OR Wall) towards the sun."""
    shadow_map = np.zeros((h, w), dtype=np.float32)
    occluders = [planes['left'], planes['right'], planes['back']]
    BIAS_DIST = 1.0 

    for (py, px), P_start_raw in scene_pixels_3d.items():
        P_start = P_start_raw + (sun_dir * BIAS_DIST)
        is_shadowed = False
        for wall_plane in occluders:
            t = wall_plane.intersect_ray(P_start, sun_dir)
            if t is not None and t > 0.01 and np.isfinite(t):
                P_hit = P_start + t * sun_dir
                if abs(P_hit[2]) > 0.1 and np.all(np.isfinite(P_hit)):
                    u_proj = (P_hit[0] / P_hit[2]) * f + cx
                    v_proj = (P_hit[1] / P_hit[2]) * f + cy
                    if np.isfinite(u_proj) and np.isfinite(v_proj):
                        u_int, v_int = int(u_proj), int(v_proj)
                        if 0 <= u_int < w and 0 <= v_int < h:
                            if alpha_mask[v_int, u_int] > 0.5:
                                is_shadowed = True; break 
        if is_shadowed: shadow_map[py, px] = 1.0

    if SAMPLE_RATE > 1:
        k = np.ones((SAMPLE_RATE*2+1, SAMPLE_RATE*2+1), np.uint8)
        shadow_map = cv2.dilate(shadow_map, k, iterations=1)
    if SHADOW_BLUR > 0:
        s = SHADOW_BLUR if SHADOW_BLUR % 2 == 1 else SHADOW_BLUR + 1
        shadow_map = cv2.GaussianBlur(shadow_map, (s, s), 0)
    return shadow_map

# ==========================================
# 3. PRE-COMPUTATION
# ==========================================
print("--- LOADING ASSETS ---")
base_img = cv2.imread(IMG_PATH).astype(np.float32) / 255.0
h, w = base_img.shape[:2]
# Load mask: White = Building, Black = Sky
alpha_mask = cv2.imread(ALPHA_MASK_PATH, 0).astype(np.float32) / 255.0
coded_map = np.load(CODED_MAP_PATH)
planes, f, (cx, cy) = get_scene_planes(w, h, VP, BACK_TL, BACK_BR)

# # --- LOAD SKY FILES ---
# print(f"--- PREPARING SKIES FROM: {SKY_FOLDER} ---")
# sky_files = sorted(glob.glob(os.path.join(SKY_FOLDER, "*.*")))
# valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
# sky_files = [f for f in sky_files if os.path.splitext(f)[1].lower() in valid_exts]

# --- LOAD SKY FILES ---
print(f"--- PREPARING SKIES FROM: {SKY_FOLDER} ---")
sky_files = glob.glob(os.path.join(SKY_FOLDER, "*.*"))
valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
sky_files = [f for f in sky_files if os.path.splitext(f)[1].lower() in valid_exts]

# --- THE FIX: Sort by the integer value of the filename ---
sky_files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))

if not sky_files:
    print("❌ Error: No valid image files found in sky folder.")
    sys.exit(1)

# Check to confirm
print(f"   > First file: {os.path.basename(sky_files[0])}")
print(f"   > Last file:  {os.path.basename(sky_files[-1])}")

if not sky_files:
    print("❌ Error: No valid image files found in sky folder.")
    sys.exit(1)
print(f"   > Found {len(sky_files)} sky frames.")

# --- PRECOMPUTE 3D GEOMETRY ---
print("--- PRECOMPUTING 3D GEOMETRY ---")
scene_pixels_3d = {} 
cam_origin = np.array([0, 0, 0], dtype=np.float32)
target_ids = [1, 2, 3, 4] # Left, Right, Floor, Back

for target_id in target_ids:
    if target_id == 3: plane = planes['floor']
    elif target_id == 1: plane = planes['left']
    elif target_id == 2: plane = planes['right']
    elif target_id == 4: plane = planes['back']
    else: continue

    coords = np.argwhere(coded_map == target_id)
    for py, px in coords[::SAMPLE_RATE]:
        ray = np.array([px - cx, py - cy, f], dtype=np.float32)
        ray = ray / np.linalg.norm(ray)
        t = plane.intersect_ray(cam_origin, ray)
        if t is not None and t > 0 and np.isfinite(t):
            scene_pixels_3d[(py, px)] = cam_origin + t * ray
print(f"   > Points Tracked: {len(scene_pixels_3d)}")

# ==========================================
# 4. TIMELAPSE LOOP
# ==========================================
output_frames = []
normal_map = np.zeros((h, w, 3), dtype=np.float32)
normal_map[coded_map == 3] = [0, -1, 0] 
normal_map[coded_map == 1] = [1, 0, 0]  
normal_map[coded_map == 2] = [-1, 0, 0] 
normal_map[coded_map == 4] = [0, 0, 1]  

print(f"\n--- RENDERING {NUM_FRAMES} FRAMES ---")

for i in range(NUM_FRAMES):
    progress = i / (NUM_FRAMES - 1)
    
    # --- A. SUN PHYSICS ---
    curr_azimuth = ARC_START_AZIMUTH + (progress * (ARC_END_AZIMUTH - ARC_START_AZIMUTH))
    curr_elevation = 5 + (PEAK_ELEVATION * np.sin(progress * np.pi))
    
    if i % 10 == 0: print(f"   Frame {i+1}/{NUM_FRAMES} | Elev: {curr_elevation:.1f}°")
    
    sun_dir = get_sun_vector(curr_azimuth, curr_elevation)
    sun_color, sun_intensity = get_daylight_color(curr_elevation)
    
    # --- B. COMPUTE SHADOWS & LIGHTING ---
    shadow_map = compute_full_scene_shadows(
        scene_pixels_3d, sun_dir, planes, alpha_mask, h, w, f, cx, cy
    )
    
    sun_vec_view = sun_dir.reshape(1, 1, 3)
    n_dot_l = np.sum(normal_map * sun_vec_view, axis=2)
    n_dot_l = np.clip(n_dot_l, 0, 1) 
    
    valid_geo_mask = (coded_map > 0).astype(np.float32)
    shadow_factor = 1.0 - (shadow_map * SHADOW_OPACITY * valid_geo_mask)
    
    ambient_strength = 0.15 + (0.15 * np.sin(progress * np.pi)) 
    ambient = sun_color * ambient_strength 
    direct = sun_color * sun_intensity * n_dot_l[:,:,None] * shadow_factor[:,:,None]
    
    # The relit building (Linear Space)
    lit_building = base_img * (ambient + direct)
    
    # --- C. LOAD & PREPARE REAL SKY ---
    # Pick sky frame, looping if we run out
    sky_idx = i % len(sky_files)
    ldr_sky = cv2.imread(sky_files[sky_idx])
    
    # Resize to match building image
    ldr_sky_resized = cv2.resize(ldr_sky, (w, h))
    
    # Convert to float and Linearize (removing Gamma 2.2)
    # This is crucial so the sky lighting matches the building lighting
    linear_sky = (ldr_sky_resized.astype(np.float32) / 255.0) ** 2.2
        
    # --- D. COMPOSITING WITH MASK ---
    # Mask: 1.0 = Building area, 0.0 = Sky area
    building_mask_3ch = (alpha_mask > 0.5)[:, :, None]
    sky_mask_3ch = 1.0 - building_mask_3ch
    
    # Final = (Lit Building pixels) + (Real Sky pixels)
    final_linear = (lit_building * building_mask_3ch) + (linear_sky * sky_mask_3ch)
    
    # --- E. SAVE ---
    # Apply Gamma Correction (Linear -> sRGB) at the very end
    final_srgb = np.power(np.clip(final_linear, 0, 1), 1/2.2)
    final_uint8 = (final_srgb * 255).astype(np.uint8)
    frame_rgb = cv2.cvtColor(final_uint8, cv2.COLOR_BGR2RGB)
    output_frames.append(Image.fromarray(frame_rgb))

# ==========================================
# 5. SAVE GIF
# ==========================================
gif_path = os.path.join(OUTPUT_FOLDER, "timelapse_real_sky.gif")
print(f"\nSaving GIF to {gif_path}...")
output_frames[0].save(
    gif_path, save_all=True, append_images=output_frames[1:], 
    duration=40, loop=0 
)
print("✅ DONE!")