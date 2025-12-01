import numpy as np

class Plane:
    def __init__(self, name, normal, point):
        self.name = name
        self.n = np.array(normal)
        self.n = self.n / (np.linalg.norm(self.n) + 1e-8)
        self.p = np.array(point)
        self.d = -np.dot(self.n, self.p)

    def intersect_ray(self, origin, direction):
        denom = np.dot(self.n, direction)
        if abs(denom) < 1e-6: 
            return float('inf')
        
        t = -(np.dot(self.n, origin) + self.d) / denom
        
        # FIX: Allow extremely small negative numbers (Self-Intersection tolerance)
        if t < -0.01: 
            return float('inf')
        return t

def get_scene_planes(width, height, vp, back_tl, back_br):
    f = width
    cx, cy = width / 2, height / 2
    
    def to_3d(px, py, z): 
        return np.array([px - cx, py - cy, z], dtype=np.float32)
    
    # 1. Back Wall is deep in the scene (Z = f)
    P_back_tl = to_3d(*back_tl, f)
    P_back_br = to_3d(*back_br, f)
    P_back_bl = to_3d(back_tl[0], back_br[1], f)
    P_back_tr = to_3d(back_br[0], back_tl[1], f)
    
    # 2. Vanishing Point is at the Camera Lens (Z = 0)
    # This difference in Z (f vs 0) is what creates the 3D slope!
    P_vp = to_3d(*vp, 0)
    
    planes = {}
    
    # Back Wall
    planes['back'] = Plane('back', [0, 0, -1], P_back_tl)
    
    # Floor (Defined by Back Edge -> VP at Lens)
    vec_edge = P_back_bl - P_back_br
    vec_vp   = P_vp - P_back_bl
    n_floor  = np.cross(vec_edge, vec_vp)
    if n_floor[1] > 0: n_floor = -n_floor
    planes['floor'] = Plane('floor', n_floor, P_back_bl)
    
    # Left Wall
    vec_edge = P_back_tl - P_back_bl
    vec_vp   = P_vp - P_back_tl
    n_left   = np.cross(vec_edge, vec_vp)
    if n_left[0] < 0: n_left = -n_left
    planes['left'] = Plane('left', n_left, P_back_tl)
    
    # Right Wall
    vec_edge = P_back_br - P_back_tr
    vec_vp   = P_vp - P_back_br
    n_right  = np.cross(vec_edge, vec_vp)
    if n_right[0] > 0: n_right = -n_right
    planes['right'] = Plane('right', n_right, P_back_tr)

    # Ceiling
    vec_edge = P_back_tr - P_back_tl
    vec_vp   = P_vp - P_back_tr
    n_ceil   = np.cross(vec_edge, vec_vp)
    if n_ceil[1] < 0: n_ceil = -n_ceil
    planes['ceil'] = Plane('ceil', n_ceil, P_back_tl)
    
    return planes, f, (cx, cy)