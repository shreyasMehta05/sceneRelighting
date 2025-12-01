import cv2
import numpy as np

# ==========================================
# 1. SETUP: Coordinates from your filename
# ==========================================
# File: shoebox_(427, 393)_(362, 321)_(509, 404).jpg
# VP = (427, 393)
# BACK_TL = (362, 322)
# BACK_BR = (509, 404)
VP = (2825,5185)
BACK_TL = (2399,4600)
BACK_BR = (3304,5358)
# Image Paths
# Ensure this matches your actual file location
IMAGE_PATH = 'images/iiit.jpg' 
ALPHA_PATH = 'images/mask_iiit.jpeg' # Optional sky mask

# ==========================================
# 2. GEOMETRY FUNCTIONS (UPDATED: OVERSHOOT LOGIC)
# ==========================================

def get_far_point(vp, corner, scale=10000):
    """
    Projects the ray from VP -> Corner far outside the canvas.
    This guarantees the polygon covers the edges without gaps.
    """
    vx, vy = vp
    cx, cy = corner
    dx = cx - vx
    dy = cy - vy
    return (int(cx + dx * scale), int(cy + dy * scale))

def create_masks(h, w, vp, back_tl, back_br):
    """
    Creates masks using the robust overshoot method.
    """
    # 1. Define corners of the back wall
    btl = back_tl                  
    btr = (back_br[0], back_tl[1]) 
    bbr = back_br                  
    bbl = (back_tl[0], back_br[1]) 
    
    # 2. Project these corners FAR outwards
    far_tl = get_far_point(vp, btl)
    far_tr = get_far_point(vp, btr)
    far_br = get_far_point(vp, bbr)
    far_bl = get_far_point(vp, bbl)
    
    masks = {
        'back': np.zeros((h, w), dtype=np.uint8),
        'left': np.zeros((h, w), dtype=np.uint8),
        'right': np.zeros((h, w), dtype=np.uint8),
        'floor': np.zeros((h, w), dtype=np.uint8),
        'ceiling': np.zeros((h, w), dtype=np.uint8)
    }
    
    # --- Create Polygons ---
    
    # Back Wall
    pts_back = np.array([btl, btr, bbr, bbl], dtype=np.int32)
    cv2.fillPoly(masks['back'], [pts_back], 255)
    
    # Left Wall 
    # Defined by: Back-Top-Left -> Back-Bottom-Left -> Far-Bottom-Left -> Far-Top-Left
    pts_left = np.array([btl, bbl, far_bl, far_tl], dtype=np.int32)
    cv2.fillPoly(masks['left'], [pts_left], 255)

    # Right Wall
    # Defined by: Back-Top-Right -> Back-Bottom-Right -> Far-Bottom-Right -> Far-Top-Right
    pts_right = np.array([btr, bbr, far_br, far_tr], dtype=np.int32)
    cv2.fillPoly(masks['right'], [pts_right], 255)
    
    # Floor
    # Defined by: Back-Bottom-Left -> Back-Bottom-Right -> Far-Bottom-Right -> Far-Bottom-Left
    pts_floor = np.array([bbl, bbr, far_br, far_bl], dtype=np.int32)
    cv2.fillPoly(masks['floor'], [pts_floor], 255)
    
    # Ceiling (Optional, often handled as sky, but calculated here just in case)
    pts_ceil = np.array([btl, btr, far_tr, far_tl], dtype=np.int32)
    cv2.fillPoly(masks['ceiling'], [pts_ceil], 255)
    
    # Clean up overlaps: Remove Back Wall from others to keep edges clean
    for key in ['left', 'right', 'floor', 'ceiling']:
        masks[key] = cv2.subtract(masks[key], masks['back'])
        
    return masks

# ==========================================
# 3. MAIN EXECUTION: GENERATE ID MAP
# ==========================================
print(f"Loading image from {IMAGE_PATH}...")
try:
    img = cv2.imread(IMAGE_PATH)
    if img is None: raise FileNotFoundError
    h, w = img.shape[:2]
except Exception as e:
    print(f"Error: Could not load {IMAGE_PATH}. Creating dummy 600x800 image.")
    h, w = 600, 800
    img = np.zeros((h, w, 3), dtype=np.uint8)

# Try to load Alpha Mask
try:
    alpha_mask = cv2.imread(ALPHA_PATH, 0)
    #####################################
    ### NOTE: INVERT MASK LOGIC HERE ### =--> iiit specific
    alpha_mask = cv2.bitwise_not(alpha_mask)
    #######################################
    if alpha_mask is None: raise FileNotFoundError
    # Resize alpha mask to match image in case dimensions differ
    alpha_mask = cv2.resize(alpha_mask, (w, h))
    _, alpha_mask = cv2.threshold(alpha_mask, 127, 255, cv2.THRESH_BINARY)
    print("Alpha mask loaded.")
except:
    print("Warning: Alpha mask not found. Defaulting to full visibility.")
    alpha_mask = np.ones((h, w), dtype=np.uint8) * 255

print("Generating masks using Overshoot logic...")
masks = create_masks(h, w, VP, BACK_TL, BACK_BR)

# Create the ID Map (Single channel, integers)
# 0=Sky, 1=Left, 2=Right, 3=Floor, 4=Back
id_map = np.zeros((h, w), dtype=np.uint8)

# Assign IDs
# Note: id_map is 0 by default, so any area not covered by these masks remains 0 (Sky/Ceiling)
id_map[masks['left'] > 0] = 1
id_map[masks['right'] > 0] = 2
id_map[masks['floor'] > 0] = 3
id_map[masks['back'] > 0] = 4

# Apply the Alpha Mask
# Any pixel that is black (0) in the alpha mask becomes SKY (0) in our map
# This effectively carves out the complex sky shape from the geometric walls
id_map[alpha_mask == 0] = 0

# ==========================================
# 4. SAVE OUTPUTS
# ==========================================

# Save the raw ID map
np.save('coded_id_map_iiit.npy', id_map)
print("Saved 'coded_id_map_iiit.npy'.")

# Save a Visual Check
vis_img = np.zeros((h, w, 3), dtype=np.uint8)
vis_img[id_map == 0] = [0, 0, 0]       # Sky = Black
vis_img[id_map == 1] = [0, 0, 255]     # Left = Red
vis_img[id_map == 2] = [0, 255, 0]     # Right = Green
vis_img[id_map == 3] = [255, 0, 0]     # Floor = Blue
vis_img[id_map == 4] = [0, 255, 255]   # Back = Yellow

cv2.imwrite('coded_vis_check.png', vis_img)
print("Saved 'coded_vis_check.png'. Check this image to verify the result.")