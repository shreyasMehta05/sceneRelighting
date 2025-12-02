"""
Shadow Ray Tracer - Standalone Shadow Computation Visualization

This script demonstrates the ray tracing shadow computation with clear,
visible shadows for analysis and debugging. It creates multiple visualizations
showing the shadow casting process in detail.

Features:
- Adjustable shadow parameters for maximum visibility
- Multiple sun direction presets for testing
- Step-by-step shadow computation visualization
- High-quality shadow maps with soft edges
- Comparison views (with/without shadows)
"""

import cv2
import numpy as np
import os
import glob
from geometry_classes import get_scene_planes
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==========================================
# CONFIGURATION
# ==========================================
VP = (427, 393)
BACK_TL = (362, 322)
BACK_BR = (509, 404)

# Files
IMG_PATH = 'images/img.jpeg'
CODED_MAP_PATH = 'coded_id_map.npy'
ALPHA_MASK_PATH = 'images/mask.jpeg'
SKY_FOLDER = 'sky3'

OUTPUT_FOLDER = 'shadow_analysis'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# SHADOW SETTINGS (Optimized for Visibility)
# ==========================================
FLOOR_SAMPLE_RATE = 1          # Lower = more detailed shadows (1 = every pixel)
SHADOW_BLUR = 31               # Larger = softer shadows
SHADOW_OPACITY = 0.85          # Higher = darker shadows (0-1)
SHADOW_BIAS = 2.0              # Ray offset to prevent self-intersection

# Sun Direction Presets (for testing different lighting scenarios)
SUN_PRESETS = {
    'morning': np.array([0.7, -0.5, 0.3]),      # Low angle, long shadows
    'noon': np.array([0.1, -0.95, 0.1]),        # High angle, short shadows
    'afternoon': np.array([-0.6, -0.6, 0.4]),   # Medium angle
    'evening': np.array([-0.8, -0.3, 0.5]),     # Low angle, dramatic shadows
    'custom': np.array([0.5, -0.7, 0.3]),       # Custom direction
}

USE_PRESET = 'afternoon'  # Change this to test different lighting
USE_SKY_DETECTION = True   # Set to False to use preset instead

# Lighting for final composites
SUN_INTENSITY = 3.0
AMBIENT_INTENSITY = 0.2

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def extract_dominant_light(sky_img):
    """Extract sun direction from sky image"""
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
    
    return sun_dir, (int(sun_u), int(sun_v))

def compute_shadow_map_detailed(floor_pixels_3d, sun_dir, planes, alpha_mask, 
                                coded_map, h, w, f, cx, cy, verbose=True):
    """
    Compute shadow map with detailed progress reporting
    Returns: (shadow_map, debug_info)
    """
    shadow_map = np.zeros((h, w), dtype=np.float32)
    occluder_planes = [
        ('left', planes['left'], 1), 
        ('right', planes['right'], 2), 
        ('back', planes['back'], 4)
    ]
    
    # Check if sun is below horizon
    if np.dot(planes['floor'].n, sun_dir) <= 0:
        if verbose:
            print("  ⚠ Sun below horizon - entire floor in shadow")
        return np.ones((h, w), dtype=np.float32) * (coded_map == 3), {'below_horizon': True}
    
    # Statistics
    total_points = len(floor_pixels_3d)
    shadowed_count = 0
    occluder_hits = {'left': 0, 'right': 0, 'back': 0}
    
    # Ray tracing
    pbar = tqdm(floor_pixels_3d.items(), desc="  Ray Tracing Shadows", 
                disable=not verbose, total=total_points)
    
    for (py, px), P_floor in pbar:
        P_start = P_floor + (sun_dir * SHADOW_BIAS)
        is_shadowed = False
        hit_wall = None
        
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
                            is_shadowed = True
                            hit_wall = wall_name
                            break
                    else:
                        is_shadowed = True
                        hit_wall = wall_name
                        break
        
        if is_shadowed:
            shadow_map[py, px] = 1.0
            shadowed_count += 1
            if hit_wall:
                occluder_hits[hit_wall] += 1
    
    # Post-processing
    if verbose:
        print(f"  📊 Shadow Statistics:")
        print(f"     Total floor points: {total_points}")
        print(f"     Shadowed points: {shadowed_count} ({100*shadowed_count/total_points:.1f}%)")
        print(f"     Occluder hits: {occluder_hits}")
    
    # Dilate if sampled
    if FLOOR_SAMPLE_RATE > 1:
        kernel_size = FLOOR_SAMPLE_RATE * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        shadow_map = cv2.dilate(shadow_map, kernel, iterations=1)
        if verbose:
            print(f"  🔧 Dilated shadow map (kernel: {kernel_size}x{kernel_size})")
    
    # Blur for soft shadows
    if SHADOW_BLUR > 0:
        blur_size = SHADOW_BLUR if SHADOW_BLUR % 2 == 1 else SHADOW_BLUR + 1
        shadow_map = cv2.GaussianBlur(shadow_map, (blur_size, blur_size), 0)
        if verbose:
            print(f"  🔧 Blurred shadow map (kernel: {blur_size}x{blur_size})")
    
    # Mask to floor only
    shadow_map = shadow_map * (coded_map == 3).astype(np.float32)
    
    debug_info = {
        'below_horizon': False,
        'total_points': total_points,
        'shadowed_count': shadowed_count,
        'shadow_percentage': 100 * shadowed_count / total_points if total_points > 0 else 0,
        'occluder_hits': occluder_hits
    }
    
    return shadow_map, debug_info

# ==========================================
# MAIN SHADOW COMPUTATION
# ==========================================
def main():
    print("="*60)
    print("SHADOW RAY TRACER - DETAILED ANALYSIS")
    print("="*60)
    
    # Load geometry
    print("\n📦 Loading scene geometry...")
    base_img_srgb = cv2.imread(IMG_PATH).astype(np.float32) / 255.0
    base_img = np.power(base_img_srgb, 2.2)
    base_img_rgb = cv2.cvtColor((base_img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    h, w = base_img.shape[:2]
    
    alpha_mask = cv2.imread(ALPHA_MASK_PATH, 0)
    kernel = np.ones((3,3), np.uint8)
    alpha_mask = cv2.erode(alpha_mask, kernel, iterations=2)
    alpha_mask = alpha_mask.astype(np.float32) / 255.0
    
    coded_map = np.load(CODED_MAP_PATH)
    planes, f, (cx, cy) = get_scene_planes(w, h, VP, BACK_TL, BACK_BR)
    
    print(f"  ✓ Image size: {w}x{h}")
    print(f"  ✓ Focal length: {f:.1f}")
    print(f"  ✓ Camera center: ({cx:.1f}, {cy:.1f})")
    
    # Precompute floor 3D positions
    print("\n🏗️ Computing floor 3D positions...")
    camera_origin = np.array([0, 0, 0], dtype=np.float32)
    floor_pixels_3d = {}
    floor_mask = (coded_map == 3)
    floor_coords = np.argwhere(floor_mask)
    floor_coords_sampled = floor_coords[::FLOOR_SAMPLE_RATE]
    
    print(f"  Sampling rate: 1/{FLOOR_SAMPLE_RATE} (processing {len(floor_coords_sampled)} points)")
    
    for py, px in tqdm(floor_coords_sampled, desc="  Computing 3D coords"):
        pixel_ray = np.array([px - cx, py - cy, f], dtype=np.float32)
        pixel_ray = pixel_ray / (np.linalg.norm(pixel_ray) + 1e-8)
        t_hit = planes['floor'].intersect_ray(camera_origin, pixel_ray)
        if t_hit is not None and t_hit > 0 and np.isfinite(t_hit):
            floor_pixels_3d[(py, px)] = camera_origin + t_hit * pixel_ray
    
    print(f"  ✓ Computed {len(floor_pixels_3d)} floor points")
    
    # Get sun direction
    print("\n☀️ Determining sun direction...")
    if USE_SKY_DETECTION:
        sky_files = glob.glob(os.path.join(SKY_FOLDER, "*.*"))
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        sky_files = [f for f in sky_files if os.path.splitext(f)[1].lower() in valid_exts]
        sky_files = sorted(sky_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split()[-1]))
        
        if len(sky_files) > 0:
            # Use middle frame
            sky_path = sky_files[len(sky_files) // 2]
            print(f"  Using sky frame: {os.path.basename(sky_path)}")
            
            ldr_sky = cv2.imread(sky_path, cv2.IMREAD_UNCHANGED)
            if len(ldr_sky.shape) == 2:
                ldr_sky = cv2.cvtColor(ldr_sky, cv2.COLOR_GRAY2BGR)
            ldr_sky = cv2.cvtColor(ldr_sky, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            linear_sky = np.power(np.clip(ldr_sky, 0, 1), 2.2)
            
            sun_dir, sun_uv = extract_dominant_light(linear_sky)
            sun_dir[1] = -sun_dir[1]  # Invert Y
            print(f"  Detected sun position: {sun_uv}")
        else:
            print(f"  ⚠ No sky images found, using preset '{USE_PRESET}'")
            sun_dir = SUN_PRESETS[USE_PRESET]
            sun_dir = sun_dir / np.linalg.norm(sun_dir)
    else:
        sun_dir = SUN_PRESETS[USE_PRESET]
        sun_dir = sun_dir / np.linalg.norm(sun_dir)
        print(f"  Using preset: '{USE_PRESET}'")
    
    print(f"  Sun direction: ({sun_dir[0]:.3f}, {sun_dir[1]:.3f}, {sun_dir[2]:.3f})")
    print(f"  Elevation angle: {np.degrees(np.arcsin(-sun_dir[1])):.1f}°")
    
    # Compute shadows
    print("\n🌑 Computing shadows...")
    print(f"  Shadow opacity: {SHADOW_OPACITY}")
    print(f"  Shadow blur: {SHADOW_BLUR}")
    shadow_map, debug_info = compute_shadow_map_detailed(
        floor_pixels_3d, sun_dir, planes, alpha_mask, coded_map, h, w, f, cx, cy
    )
    
    # Invert for rendering
    shadow_map_inv = 1.0 - shadow_map
    
    # Create normal map
    normal_map = np.zeros((h, w, 3), dtype=np.float32)
    normal_map[coded_map == 3] = [0, -1, 0]
    normal_map[coded_map == 1] = [1, 0, 0]
    normal_map[coded_map == 2] = [-1, 0, 0]
    normal_map[coded_map == 4] = [0, 0, 1]
    
    # Compute lighting
    print("\n💡 Computing lighting...")
    sun_dir_broadcast = sun_dir.reshape(1, 1, 3)
    n_dot_l = np.sum(normal_map * sun_dir_broadcast, axis=2)
    n_dot_l = np.clip(n_dot_l, 0, 1)
    
    shadow_term = 1.0 - (shadow_map_inv * SHADOW_OPACITY)
    direct_intensity = SUN_INTENSITY * n_dot_l * shadow_term
    total_intensity = AMBIENT_INTENSITY + direct_intensity
    
    # Apply lighting
    lit_image = base_img * total_intensity[:, :, None]
    lit_image_no_shadow = base_img * (AMBIENT_INTENSITY + SUN_INTENSITY * n_dot_l)[:, :, None]
    
    # Tone map
    def aces_tonemap(x):
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
    
    lit_image = aces_tonemap(lit_image)
    lit_image_no_shadow = aces_tonemap(lit_image_no_shadow)
    
    # Convert to sRGB
    lit_image = np.power(lit_image, 1.0/2.2)
    lit_image_no_shadow = np.power(lit_image_no_shadow, 1.0/2.2)
    lit_image_rgb = (np.clip(lit_image, 0, 1) * 255).astype(np.uint8)
    lit_image_rgb = cv2.cvtColor(lit_image_rgb, cv2.COLOR_BGR2RGB)
    lit_image_no_shadow_rgb = (np.clip(lit_image_no_shadow, 0, 1) * 255).astype(np.uint8)
    lit_image_no_shadow_rgb = cv2.cvtColor(lit_image_no_shadow_rgb, cv2.COLOR_BGR2RGB)
    
    # ==========================================
    # VISUALIZATIONS
    # ==========================================
    print("\n📊 Creating visualizations...")
    
    # Figure 1: Shadow Analysis
    fig1 = plt.figure(figsize=(20, 12), facecolor='white')
    gs = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.3, wspace=0.3)
    
    # Original
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.imshow(base_img_rgb)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Floor mask
    ax2 = fig1.add_subplot(gs[0, 1])
    floor_viz = np.zeros((h, w, 3), dtype=np.uint8)
    floor_viz[coded_map == 3] = [100, 149, 237]  # Cornflower blue
    ax2.imshow(floor_viz)
    ax2.set_title(f'Floor Region\n{len(floor_pixels_3d)} sampled points', 
                 fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Raw shadow map
    ax3 = fig1.add_subplot(gs[0, 2])
    im3 = ax3.imshow(shadow_map, cmap='Reds', vmin=0, vmax=1)
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    shadow_pct = debug_info['shadow_percentage']
    ax3.set_title(f'Shadow Map (Raw)\n{shadow_pct:.1f}% in shadow', 
                 fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Shadow overlay on original
    ax4 = fig1.add_subplot(gs[1, 0])
    shadow_overlay = base_img_rgb.copy().astype(np.float32) / 255.0
    shadow_viz = 1.0 - (shadow_map_inv * 0.7)
    shadow_overlay = shadow_overlay * shadow_viz[:, :, None]
    ax4.imshow(np.clip(shadow_overlay, 0, 1))
    ax4.set_title('Shadow Overlay (70% opacity)', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Without shadows
    ax5 = fig1.add_subplot(gs[1, 1])
    ax5.imshow(lit_image_no_shadow_rgb)
    ax5.set_title('Lit (No Shadows)', fontsize=14, fontweight='bold')
    ax5.axis('off')
    
    # With shadows
    ax6 = fig1.add_subplot(gs[1, 2])
    ax6.imshow(lit_image_rgb)
    ax6.set_title(f'Lit (With Shadows)\nOpacity: {SHADOW_OPACITY}', 
                 fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_FOLDER, 'shadow_analysis.png')
    fig1.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print(f"  ✓ Saved: {output_path}")
    
    # Figure 2: Detailed breakdown
    fig2 = plt.figure(figsize=(18, 10), facecolor='white')
    gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.3, wspace=0.3)
    
    # N·L diffuse
    ax1 = fig2.add_subplot(gs2[0, 0])
    im1 = ax1.imshow(n_dot_l, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    ax1.set_title('N·L (Diffuse Term)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Shadow term
    ax2 = fig2.add_subplot(gs2[0, 1])
    im2 = ax2.imshow(shadow_term, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    ax2.set_title('Shadow Attenuation Term', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Direct intensity
    ax3 = fig2.add_subplot(gs2[0, 2])
    im3 = ax3.imshow(direct_intensity, cmap='hot')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    ax3.set_title('Direct Light Intensity', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Comparison: side by side
    ax4 = fig2.add_subplot(gs2[1, :])
    comparison = np.hstack([lit_image_no_shadow_rgb, lit_image_rgb])
    ax4.imshow(comparison)
    
    # Add dividing line
    ax4.axvline(w, color='yellow', linewidth=3, linestyle='--')
    ax4.text(w/2, h-20, 'WITHOUT SHADOWS', ha='center', fontsize=16, 
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax4.text(w + w/2, h-20, 'WITH SHADOWS', ha='center', fontsize=16, 
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax4.set_title('Direct Comparison', fontsize=16, fontweight='bold', pad=15)
    ax4.axis('off')
    
    plt.tight_layout()
    output_path2 = os.path.join(OUTPUT_FOLDER, 'shadow_breakdown.png')
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"  ✓ Saved: {output_path2}")
    
    # Save individual outputs
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'shadow_map.png'), 
                (shadow_map * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'lit_with_shadows.png'), 
                cv2.cvtColor(lit_image_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'lit_without_shadows.png'), 
                cv2.cvtColor(lit_image_no_shadow_rgb, cv2.COLOR_RGB2BGR))
    
    print("\n" + "="*60)
    print("✅ SHADOW RAY TRACING COMPLETE!")
    print(f"📁 Results saved to: {OUTPUT_FOLDER}/")
    print("="*60)
    print("\nGenerated files:")
    print("  • shadow_analysis.png - Complete shadow analysis")
    print("  • shadow_breakdown.png - Detailed lighting breakdown")
    print("  • shadow_map.png - Raw shadow map")
    print("  • lit_with_shadows.png - Final result with shadows")
    print("  • lit_without_shadows.png - Comparison without shadows")
    print("\nShadow Statistics:")
    print(f"  • Shadow coverage: {shadow_pct:.1f}%")
    print(f"  • Shadow opacity: {SHADOW_OPACITY * 100}%")
    print(f"  • Blur kernel: {SHADOW_BLUR}px")
    print(f"  • Sample rate: 1/{FLOOR_SAMPLE_RATE}")
    print("="*60)

if __name__ == "__main__":
    main()
