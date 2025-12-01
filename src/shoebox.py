import matplotlib.pyplot as plt
import cv2
import numpy as np

# ==========================================
# PART 1: HELPER FUNCTIONS (Geometry Logic)
# ==========================================

def get_far_point(vp, corner, scale=10000):
    """
    Instead of finding the exact border intersection, this projects 
    the point far outside the image canvas. cv2.fillPoly will 
    handle the clipping automatically.
    """
    vx, vy = vp
    cx, cy = corner
    
    # Vector from VP to Corner
    dx = cx - vx
    dy = cy - vy
    
    # Project far out
    return (int(cx + dx * scale), int(cy + dy * scale))

def create_masks(h, w, vp, back_tl, back_br):
    """
    Returns a dictionary containing 5 binary masks.
    """
    # 1. Define the 4 corners of the back wall
    btl = back_tl                  
    btr = (back_br[0], back_tl[1]) 
    bbr = back_br                  
    bbl = (back_tl[0], back_br[1]) 
    
    # 2. Project these corners FAR outwards from the VP
    far_tl = get_far_point(vp, btl)
    far_tr = get_far_point(vp, btr)
    far_br = get_far_point(vp, bbr)
    far_bl = get_far_point(vp, bbl)
    
    # Initialize masks
    masks = {
        'back': np.zeros((h, w), dtype=np.uint8),
        'left': np.zeros((h, w), dtype=np.uint8),
        'right': np.zeros((h, w), dtype=np.uint8),
        'floor': np.zeros((h, w), dtype=np.uint8),
        'ceiling': np.zeros((h, w), dtype=np.uint8)
    }
    
    # --- Create Polygons ---
    # We define the polygons using the Back Wall corners and the Far points.
    # The order of points matters (clockwise or counter-clockwise).

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
    
    # Ceiling
    # Defined by: Back-Top-Left -> Back-Top-Right -> Far-Top-Right -> Far-Top-Left
    pts_ceil = np.array([btl, btr, far_tr, far_tl], dtype=np.int32)
    cv2.fillPoly(masks['ceiling'], [pts_ceil], 255)

    # Floor
    # Defined by: Back-Bottom-Left -> Back-Bottom-Right -> Far-Bottom-Right -> Far-Bottom-Left
    pts_floor = np.array([bbl, bbr, far_br, far_bl], dtype=np.int32)
    cv2.fillPoly(masks['floor'], [pts_floor], 255)

    # Note: Because the polygons overlap slightly at the edges and extend infinitely,
    # we should mask out the 'back' wall from the others to keep it clean, 
    # although drawing order usually handles this.
    # Explicitly removing back wall overlap:
    masks['left'] = cv2.subtract(masks['left'], masks['back'])
    masks['right'] = cv2.subtract(masks['right'], masks['back'])
    masks['ceiling'] = cv2.subtract(masks['ceiling'], masks['back'])
    masks['floor'] = cv2.subtract(masks['floor'], masks['back'])

    return masks

# ==========================================
# PART 2: MAIN EXECUTION
# ==========================================

# 1. Load Image
try:
    # Update this path to your actual file
    img = cv2.imread('images/iiit.jpg') 
    if img is None: raise FileNotFoundError
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
except Exception as e:
    print("Image not found, creating dummy image...")
    img = np.zeros((600, 800, 3), dtype=np.uint8) + 200 

h, w, _ = img.shape

# 2. Get User Input
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.title("Click: 1. Vanishing Point, 2. Back-Wall Top-Left, 3. Back-Wall Bottom-Right")
print("Please click 3 points on the popup window...")

points = plt.ginput(3, timeout=-1)
plt.close()

if len(points) < 3:
    print("Error: You didn't click 3 points!")
else:
    vp = (int(points[0][0]), int(points[0][1]))
    back_tl = (int(points[1][0]), int(points[1][1]))
    back_br = (int(points[2][0]), int(points[2][1]))

    # hardcoded points for testing without GUI
    # vp = (427, 393)
    # back_tl = (362, 322)
    # back_br = (509, 404)

    print(f"Vanishing Point: {vp}")
    print(f"Back Wall: {back_tl} to {back_br}")

    # 3. Generate Masks
    region_masks = create_masks(h, w, vp, back_tl, back_br)
    
    # 4. Visualize Results (Overlay)
    overlay = img.copy()
    
    colors = {
        'back': [255, 0, 0],    # Red
        'floor': [0, 0, 255],   # Blue
        'ceiling': [0, 255, 0], # Green
        'left': [255, 255, 0],  # Yellow
        'right': [255, 0, 255]  # Magenta
    }

    for region, mask in region_masks.items():
        color_layer = np.zeros_like(img)
        color_layer[mask > 0] = colors[region]
        mask_bool = mask > 0
        overlay[mask_bool] = cv2.addWeighted(img[mask_bool], 0.6, color_layer[mask_bool], 0.4, 0)

    # Draw wireframe lines for clarity
    cv2.line(overlay, vp, back_tl, (255, 255, 255), 2)
    cv2.line(overlay, vp, back_br, (255, 255, 255), 2)
    cv2.line(overlay, vp, (back_br[0], back_tl[1]), (255, 255, 255), 2)
    cv2.line(overlay, vp, (back_tl[0], back_br[1]), (255, 255, 255), 2)
    cv2.rectangle(overlay, back_tl, back_br, (255, 255, 255), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(overlay)
    plt.title("Segmentation Result")
    plt.axis('off')
    # save the image and include the vp , back_tl and back_br points in the name
    plt.savefig(f'segmented_vp_{vp[0]}_{vp[1]}_backtl_{back_tl[0]}_{back_tl[1]}_backbr_{back_br[0]}_{back_br[1]}_iiit.png')
    plt.show()