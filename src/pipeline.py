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
VP = (427, 393)
BACK_TL = (362, 322)
BACK_BR = (509, 404)

# ================
# Files
# ================
IMG_PATH = 'images/img.jpeg'
CODED_MAP_PATH = 'coded_id_map.npy'
ALPHA_MASK_PATH = 'images/mask.jpeg' 
SKY_FOLDER = 'sky3' 

OUTPUT_FOLDER = 'final_output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


#################################
# --- SETTINGS ---
#################################
GLOBAL_BRIGHTNESS = 1.0        
IMAGE_SELECTION_STEP = 1
SHADOW_BLUR = 21
SHADOW_OPACITY = 0.75
FLOOR_SAMPLE_RATE = 2

# --- VIRTUAL SUN (Now separate intensity and color) ---
SUN_INTENSITY = 2.5            # Pure intensity multiplier
SUN_COLOR = np.array([1.0, 0.95, 0.85])  # Warm sunlight tint (not multiplied by intensity yet)

# --- AMBIENT LIGHT ---
AMBIENT_INTENSITY = 0.3        # Brighter ambient for better color preservation
AMBIENT_COLOR = np.array([0.6, 0.65, 0.7])  # Cool ambient (sky-like)

# --- EXPOSURE & SATURATION ---
EXPOSURE_COMPENSATION = 0.5    # Adjust overall brightness (0 = neutral, + = brighter)
SATURATION_BOOST = 1.2         # Boost color saturation (1.0 = original)

# --- TEMPORAL STABILIZATION ---
TEMPORAL_SMOOTHING = False
TEMPORAL_WINDOW = 5
TEMPORAL_WEIGHT_CURRENT = 0.6

# --- DEBUG & OUTPUT ---
DEBUG_MODE = False
TEST_CONSTANT_SUN = False
CONSTANT_SUN_DIR = np.array([0.5, 0.8, 0.0])

GIF_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, 'sky3.gif')
VIDEO_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, 'sky3.mp4')
GIF_DURATION = 100 
VIDEO_FPS = 24

# ==========================================
# 1.5 TEMPORAL BUFFER CLASS
# ==========================================
class TemporalBuffer:
    def __init__(self, window_size=3, current_weight=0.5):
        self.window_size = window_size
        self.current_weight = current_weight
        self.buffer = deque(maxlen=window_size)
        if window_size > 1:
            remaining_weight = 1.0 - current_weight
            self.history_weights = [remaining_weight / (window_size - 1)] * (window_size - 1)
        else:
            self.history_weights = []
    
    def add_frame(self, frame):
        self.buffer.append(frame.copy())
    
    def get_smoothed_frame(self):
        if len(self.buffer) == 0: return None
        if len(self.buffer) == 1 or not TEMPORAL_SMOOTHING: return self.buffer[-1]
        
        result = self.buffer[-1] * self.current_weight
        for i in range(len(self.buffer) - 1):
            weight = self.history_weights[min(i, len(self.history_weights) - 1)]
            result += self.buffer[i] * weight
        return result

# ==========================================
# 2. LOAD STATIC GEOMETRY
# ==========================================
print("--- LOADING GEOMETRY ---")

# Load image in LINEAR space (proper way)
base_img_srgb = cv2.imread(IMG_PATH).astype(np.float32) / 255.0
base_img = np.power(base_img_srgb, 2.2)  # Convert sRGB to linear

h, w = base_img.shape[:2]

alpha_mask = cv2.imread(ALPHA_MASK_PATH, 0)
kernel = np.ones((3,3), np.uint8)
alpha_mask = cv2.erode(alpha_mask, kernel, iterations=2) 
alpha_mask = alpha_mask.astype(np.float32) / 255.0

coded_map = np.load(CODED_MAP_PATH)
planes, f, (cx, cy) = get_scene_planes(w, h, VP, BACK_TL, BACK_BR)

# ==========================================
# 3. PRECOMPUTE FLOOR
# ==========================================
print("--- Precomputing floor 3D positions ---")
camera_origin = np.array([0, 0, 0], dtype=np.float32)
floor_pixels_3d = {}
floor_mask = (coded_map == 3)
floor_coords = np.argwhere(floor_mask)
floor_coords_sampled = floor_coords[::FLOOR_SAMPLE_RATE]

for py, px in tqdm(floor_coords_sampled, desc="Computing floor 3D", leave=False):
    pixel_ray = np.array([px - cx, py - cy, f], dtype=np.float32)
    pixel_ray = pixel_ray / (np.linalg.norm(pixel_ray) + 1e-8)
    t_hit = planes['floor'].intersect_ray(camera_origin, pixel_ray)
    if t_hit is not None and t_hit > 0 and np.isfinite(t_hit):
        floor_pixels_3d[(py, px)] = camera_origin + t_hit * pixel_ray

# ==========================================
# 4. PRECOMPUTE NORMALS
# ==========================================
normal_map = np.zeros((h, w, 3), dtype=np.float32)
normal_map[coded_map == 3] = [0, -1, 0]   # Floor faces Sun (-Y)
normal_map[coded_map == 1] = [1, 0, 0]    # Left wall
normal_map[coded_map == 2] = [-1, 0, 0]   # Right wall
normal_map[coded_map == 4] = [0, 0, 1]    # Back wall

# ==========================================
# 5. FUNCTIONS
# ==========================================
def extract_dominant_light(sky_img):
    sky_h, sky_w = sky_img.shape[:2]
    gray = np.mean(sky_img, axis=2)
    threshold_value = max(0.9, np.percentile(gray, 95))
    sun_mask = (gray >= threshold_value).astype(np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sun_mask, connectivity=8)
    if num_labels > 1:
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        sun_u, sun_v = centroids[largest_component]
    else:
        _, _, _, max_loc = cv2.minMaxLoc(gray)
        sun_u, sun_v = max_loc
    
    phi = (sun_u / sky_w) * 2 * np.pi
    theta = (sun_v / sky_h) * np.pi
    
    sun_x = np.sin(theta) * np.cos(phi)
    sun_y = np.cos(theta)
    sun_z = np.sin(theta) * np.sin(phi)
    
    sun_dir = np.array([sun_x, sun_y, sun_z], dtype=np.float32)
    sun_dir = sun_dir / (np.linalg.norm(sun_dir) + 1e-8)
    
    ambient = np.array([0.2, 0.2, 0.2]) 
        
    return sun_dir, ambient, (int(sun_u), int(sun_v))

def compute_shadow_map_raytracing(floor_pixels_3d, sun_dir, planes, alpha_mask, coded_map, h, w, f, cx, cy):
    shadow_map = np.zeros((h, w), dtype=np.float32)
    occluder_planes = [('left', planes['left'], 1), ('right', planes['right'], 2), ('back', planes['back'], 4)]
    
    BIAS_DIST = 2.0 
    
    if np.dot(planes['floor'].n, sun_dir) <= 0:
        return np.ones((h, w), dtype=np.float32) * (coded_map == 3)

    for (py, px), P_floor in tqdm(floor_pixels_3d.items(), desc="    Ray Tracing", leave=False):
        P_start = P_floor + (sun_dir * BIAS_DIST)
        is_shadowed = False
        
        for wall_name, plane, wall_id in occluder_planes:
            t = plane.intersect_ray(P_start, sun_dir)
            if t is not None and t > 0.01 and np.isfinite(t):
                P_hit = P_start + t * sun_dir
                if abs(P_hit[2]) > 0.1:
                    u_proj = (P_hit[0] / P_hit[2]) * f + cx
                    v_proj = (P_hit[1] / P_hit[2]) * f + cy
                    u_int, v_int = int(np.round(u_proj)), int(np.round(v_proj))
                    
                    if 0 <= u_int < w and 0 <= v_int < h:
                        if alpha_mask[v_int, u_int] > 0.5:
                            is_shadowed = True; break 
                    else:
                        is_shadowed = True; break
        
        if is_shadowed: shadow_map[py, px] = 1.0

    if FLOOR_SAMPLE_RATE > 1:
        kernel = np.ones((FLOOR_SAMPLE_RATE * 2 + 1, FLOOR_SAMPLE_RATE * 2 + 1), np.uint8)
        shadow_map = cv2.dilate(shadow_map, kernel, iterations=1)
    
    if SHADOW_BLUR > 0:
        s = SHADOW_BLUR if SHADOW_BLUR % 2 == 1 else SHADOW_BLUR + 1
        shadow_map = cv2.GaussianBlur(shadow_map, (s, s), 0)
    
    return shadow_map * (coded_map == 3).astype(np.float32)

def apply_saturation(img, saturation):
    """Apply saturation boost while preserving luminance"""
    # Convert to HSV-like representation
    luminance = 0.2126 * img[:,:,2] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,0]  # Standard luminance
    luminance = np.maximum(luminance, 1e-8)[:, :, None]
    
    # Normalize to get chromaticity
    chroma = img / luminance
    
    # Boost saturation
    chroma = 1.0 + (chroma - 1.0) * saturation
    
    # Reconstruct with original luminance
    result = chroma * luminance
    return result

# ==========================================
# 6. PROCESSING LOOP
# ==========================================
sky_files = glob.glob(os.path.join(SKY_FOLDER, "*.*"))
valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.hdr', '.exr']
sky_files = [f for f in sky_files if os.path.splitext(f)[1].lower() in valid_exts]
sky_files = sorted(sky_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split()[-1]))[::IMAGE_SELECTION_STEP]
print(f"Found {len(sky_files)} sky images for processing.")
print("-----------------------------------")
print("Starting processing loop...")
print("Top-5 sky files to be used:", sky_files[:5])
print("-----------------------------------")
print(f"check folder: {OUTPUT_FOLDER}")

temporal_buffer = TemporalBuffer(window_size=TEMPORAL_WINDOW, current_weight=TEMPORAL_WEIGHT_CURRENT)
processed_images = []
processed_frames_bgr = []

print(f"\nProcessing {len(sky_files)} frames...")

for sky_idx, sky_path in enumerate(sky_files):
    print(f"\n[{sky_idx+1}/{len(sky_files)}] Processing Frame...")

    # --- LOAD SKY ---
    ldr_sky = cv2.imread(sky_path, cv2.IMREAD_UNCHANGED)
    if ldr_sky is None: continue
    if len(ldr_sky.shape) == 2: ldr_sky = cv2.cvtColor(ldr_sky, cv2.COLOR_GRAY2BGR)
    elif ldr_sky.shape[2] == 4: ldr_sky = ldr_sky[:, :, :3]
    ldr_sky = cv2.cvtColor(ldr_sky, cv2.COLOR_BGR2RGB).astype(np.float32)
    if ldr_sky.max() > 1.0: ldr_sky = ldr_sky / 255.0
    linear_sky = np.power(np.clip(ldr_sky, 0, 1), 2.2)
    
    # --- EXTRACT LIGHTING ---
    if TEST_CONSTANT_SUN:
        sun_dir = CONSTANT_SUN_DIR / np.linalg.norm(CONSTANT_SUN_DIR)
    else:
        sun_dir, _, _ = extract_dominant_light(linear_sky)
    
    sun_dir[1] = -sun_dir[1]  # Invert Y
    
    # --- COMPUTE SHADOWS ---
    shadow_map = compute_shadow_map_raytracing(
        floor_pixels_3d, sun_dir, planes, alpha_mask, coded_map, h, w, f, cx, cy
    )
    shadow_map = 1.0 - shadow_map  # Invert for correct shadow direction
    
    # ==========================================
    # IMPROVED LIGHTING MODEL (COLOR-PRESERVING)
    # ==========================================
    print("  💡 Applying color-preserving lighting...")
    
    # 1. Calculate lighting coefficients (intensity only, no color yet)
    sun_dir_broadcast = sun_dir.reshape(1, 1, 3)
    n_dot_l = np.sum(normal_map * sun_dir_broadcast, axis=2)
    n_dot_l = np.clip(n_dot_l, 0, 1)
    
    # 2. Shadow attenuation
    shadow_term = 1.0 - (shadow_map * SHADOW_OPACITY)
    
    # 3. Build lighting intensity map (grayscale)
    direct_intensity = SUN_INTENSITY * n_dot_l * shadow_term
    total_intensity = AMBIENT_INTENSITY + direct_intensity
    
    # 4. Build color tint map (blend ambient and sun colors based on contribution)
    direct_contribution = direct_intensity / (total_intensity + 1e-8)
    ambient_contribution = AMBIENT_INTENSITY / (total_intensity + 1e-8)
    
    # Normalize contributions
    total_contribution = direct_contribution + ambient_contribution
    direct_contribution = direct_contribution / (total_contribution + 1e-8)
    ambient_contribution = ambient_contribution / (total_contribution + 1e-8)
    
    # Blend sun and ambient colors
    light_color = (direct_contribution[:, :, None] * SUN_COLOR.reshape(1, 1, 3) + 
                   ambient_contribution[:, :, None] * AMBIENT_COLOR.reshape(1, 1, 3))
    
    # 5. Apply lighting to base image (SEPARATE intensity and color)
    # First apply intensity, then tint with light color
    current_frame = base_img * total_intensity[:, :, None] * light_color
    
    # 6. Apply exposure compensation
    current_frame = current_frame * np.power(2.0, EXPOSURE_COMPENSATION)
    
    # 7. Boost saturation to preserve colors
    current_frame = apply_saturation(current_frame, SATURATION_BOOST)
    
    # --- COMPOSITING ---
    temporal_buffer.add_frame(current_frame)
    final_output = temporal_buffer.get_smoothed_frame()
    
    building_mask = (alpha_mask > 0.5)[:, :, None]
    final_output = final_output * building_mask
    
    sky_bg = cv2.resize(linear_sky, (w, h))
    inv_mask = (1.0 - alpha_mask)[:, :, None]
    final_output = final_output + (sky_bg * inv_mask)
    
    # --- IMPROVED TONE MAPPING ---
    # Use ACES filmic tone mapping for better color preservation
    # https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    def aces_tonemap(x):
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
    
    final_output = aces_tonemap(final_output)
    
    # Convert back to sRGB
    final_output = np.power(final_output, 1.0/2.2)
    final_output_uint8 = (np.clip(final_output, 0, 1) * 255).astype(np.uint8)
    
    processed_images.append(Image.fromarray(final_output_uint8))
    processed_frames_bgr.append(cv2.cvtColor(final_output_uint8, cv2.COLOR_RGB2BGR))

# ==========================================
# 7. SAVE OUTPUTS
# ==========================================
if len(processed_images) > 0:
    print(f"\nSaving {len(processed_images)} frames...")
    processed_images[0].save(
        GIF_OUTPUT_PATH, save_all=True, append_images=processed_images[1:],
        duration=GIF_DURATION, loop=0, optimize=False
    )
    
    height, width = processed_frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, VIDEO_FPS, (width, height))
    for frame in processed_frames_bgr:
        out.write(frame)
    out.release()
    print("DONE! 🎬")
else:
    print("\n⚠ No frames processed!")