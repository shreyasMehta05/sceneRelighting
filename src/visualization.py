"""
Visualization Module for Scene Relighting Pipeline (Enhanced Version)

This module creates comprehensive, aesthetically-enhanced visualizations for each 
step of the relighting pipeline using Seaborn styling and animated GIFs.

VISUALIZATIONS GENERATED:
========================

1. SUN DETECTION (01_sun_detection_frame_XXX.png)
   - Original sky image with detected sun position
   - Grayscale intensity map with threshold visualization
   - Binary sun mask after connected components analysis
   - Complete coordinate conversion diagram: (u,v) → (θ,φ) → (x,y,z)
   - 3D sun direction vector on unit sphere
   - Spherical coordinate projection with grid overlay

2. SUN TRAJECTORY - ENHANCED (Multiple outputs)
   a) 02_sun_trajectory_enhanced.png
      - UV trajectory with gradient coloring (Seaborn palette)
      - Spherical coordinates trajectory with colorbar
      - 3D trajectory on unit sphere with enhanced styling
   
   b) 02_sun_trajectory_frames_grid.png
      - Grid of randomly sampled frames (default: 8 frames)
      - Shows sun position evolution across sequence
      - Includes coordinate info for each frame
   
   c) sun_trajectory_animation.gif (in viz/animations/)
      - Animated 2D trajectory showing sun movement
      - Frame-by-frame progression with info overlay
      - Duration: 100ms per frame
   
   d) sun_trajectory_3d_rotating.gif (in viz/animations/)
      - 360° rotating view of 3D sun trajectory
      - 60 frames, smooth rotation
      - Duration: 50ms per frame

3. SCENE GEOMETRY (03_scene_geometry.png)
   - Color-coded segmentation map
   - Image with geometry annotations (VP, perspective lines)
   - RGB-encoded normal map
   - 3D scene reconstruction with planes (floor, walls)
   - Plane equations with normals and points
   - Camera ray visualization with plane intersections

4. SHADOW COMPUTATION (04_shadow_computation_frame_XXX.png)
   - Original image and floor mask
   - Raw shadow map (sampled ray tracing)
   - Blurred shadow map with soft edges
   - Shadow overlay on image
   - 3D ray tracing visualization (red=occluded, green=lit)

5. LIGHTING (05_lighting_frame_XXX.png) - Enhanced with Seaborn
   - Normal map (RGB encoded)
   - N·L diffuse term with rocket colormap
   - Direct light intensity
   - Total intensity (ambient + direct)
   - Light color tint map
   - Lighting parameters panel
   - Original, final lit, and intensity-only comparisons

FEATURES:
=========
✓ Seaborn aesthetic styling for professional plots
✓ Enhanced color palettes (plasma, rocket, mako)
✓ Animated GIF generation for trajectory visualization
✓ Random frame sampling for grid visualizations
✓ 3D rotating animations
✓ High-DPI output (150-200 DPI)
✓ Comprehensive coordinate conversion visualization
✓ Separate output folders for static images and animations

USAGE:
======
python src/visualization.py

All outputs saved to:
- Static images: viz/
- Animated GIFs: viz/animations/

REQUIREMENTS:
============
- numpy, matplotlib, opencv-python
- seaborn (for enhanced aesthetics)
- tqdm (for progress bars)
- Pillow (for GIF creation)
"""

import cv2
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for GIF generation
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
from geometry_classes import get_scene_planes, Plane
from tqdm import tqdm
from PIL import Image
import random

# Set aesthetic style with Seaborn
sns.set_style("darkgrid")
sns.set_context("notebook", font_scale=1.1)
sns.set_palette("husl")

# ==========================================
# CONFIGURATION (Import from pipeline)
# ==========================================
VP = (427, 393)
BACK_TL = (362, 322)
BACK_BR = (509, 404)

IMG_PATH = 'images/img.jpeg'
CODED_MAP_PATH = 'coded_id_map.npy'
ALPHA_MASK_PATH = 'images/mask.jpeg'
SKY_FOLDER = 'sky3'

VIZ_FOLDER = 'viz3'
GIF_FOLDER = os.path.join(VIZ_FOLDER, 'animations')
os.makedirs(VIZ_FOLDER, exist_ok=True)
os.makedirs(GIF_FOLDER, exist_ok=True)

FLOOR_SAMPLE_RATE = 2

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def save_fig(fig, filename, dpi=150, transparent=False):
    """Save figure to viz folder with enhanced styling"""
    filepath = os.path.join(VIZ_FOLDER, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none', transparent=transparent)
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")

def save_gif(frames, filename, duration=100):
    """Save frames as GIF animation"""
    filepath = os.path.join(GIF_FOLDER, filename)
    frames[0].save(filepath, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    print(f"  ✓ Saved GIF: {filename}")

def load_base_image():
    """Load base image in linear space"""
    img_srgb = cv2.imread(IMG_PATH).astype(np.float32) / 255.0
    img_linear = np.power(img_srgb, 2.2)
    img_rgb = cv2.cvtColor((img_linear * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return img_rgb

def load_coded_map():
    """Load coded segmentation map"""
    return np.load(CODED_MAP_PATH)

def load_alpha_mask():
    """Load alpha mask"""
    alpha = cv2.imread(ALPHA_MASK_PATH, 0)
    kernel = np.ones((3,3), np.uint8)
    alpha = cv2.erode(alpha, kernel, iterations=2)
    return alpha.astype(np.float32) / 255.0

# ==========================================
# 1. SUN DETECTION VISUALIZATION
# ==========================================
def extract_sun_with_debug(sky_img):
    """Extract sun direction with intermediate steps for visualization"""
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
    
    # Coordinate conversion: (u,v) → (θ,φ) → (x,y,z)
    phi = (sun_u / sky_w) * 2 * np.pi
    theta = (sun_v / sky_h) * np.pi
    
    sun_x = np.sin(theta) * np.cos(phi)
    sun_y = np.cos(theta)
    sun_z = np.sin(theta) * np.sin(phi)
    
    sun_dir = np.array([sun_x, sun_y, sun_z], dtype=np.float32)
    sun_dir = sun_dir / (np.linalg.norm(sun_dir) + 1e-8)
    
    # Flip Y for rendering coordinate system
    sun_dir[1] = -sun_dir[1]
    
    return {
        'gray': gray,
        'sun_mask': sun_mask,
        'sun_uv': (int(sun_u), int(sun_v)),
        'phi': phi,
        'theta': theta,
        'sun_dir': sun_dir,
        'threshold': threshold_value
    }

def visualize_sun_detection(sky_path, frame_idx=0):
    """Visualize sun detection process with coordinate conversion"""
    print(f"\n📍 Visualizing Sun Detection (Frame {frame_idx})...")
    
    # Load sky
    ldr_sky = cv2.imread(sky_path, cv2.IMREAD_UNCHANGED)
    if ldr_sky is None:
        print(f"  ✗ Failed to load {sky_path}")
        return None
    
    if len(ldr_sky.shape) == 2:
        ldr_sky = cv2.cvtColor(ldr_sky, cv2.COLOR_GRAY2BGR)
    elif ldr_sky.shape[2] == 4:
        ldr_sky = ldr_sky[:, :, :3]
    
    ldr_sky = cv2.cvtColor(ldr_sky, cv2.COLOR_BGR2RGB).astype(np.float32)
    if ldr_sky.max() > 1.0:
        ldr_sky = ldr_sky / 255.0
    linear_sky = np.power(np.clip(ldr_sky, 0, 1), 2.2)
    
    # Extract sun with debug info
    sun_info = extract_sun_with_debug(linear_sky)
    
    # Create visualization
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Original Sky
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.clip(linear_sky, 0, 1))
    ax1.plot(sun_info['sun_uv'][0], sun_info['sun_uv'][1], 'r+', markersize=20, markeredgewidth=3)
    circle = patches.Circle(sun_info['sun_uv'], 30, fill=False, edgecolor='red', linewidth=2)
    ax1.add_patch(circle)
    ax1.set_title(f'Sky Image\nSun Position: ({sun_info["sun_uv"][0]}, {sun_info["sun_uv"][1]})')
    ax1.axis('off')
    
    # 2. Grayscale Intensity
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(sun_info['gray'], cmap='hot')
    ax2.plot(sun_info['sun_uv'][0], sun_info['sun_uv'][1], 'c+', markersize=20, markeredgewidth=3)
    plt.colorbar(im, ax=ax2, fraction=0.046)
    ax2.set_title(f'Grayscale Intensity\nThreshold: {sun_info["threshold"]:.3f}')
    ax2.axis('off')
    
    # 3. Sun Mask
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(sun_info['sun_mask'], cmap='gray')
    ax3.plot(sun_info['sun_uv'][0], sun_info['sun_uv'][1], 'r+', markersize=20, markeredgewidth=3)
    ax3.set_title('Detected Sun Region\n(Thresholded Mask)')
    ax3.axis('off')
    
    # 4. Coordinate Conversion Diagram
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.text(0.5, 0.9, 'Coordinate Conversion', ha='center', fontsize=14, weight='bold', transform=ax4.transAxes)
    
    conversion_text = f"""
    Step 1: Image Coordinates (u, v)
    u = {sun_info['sun_uv'][0]}
    v = {sun_info['sun_uv'][1]}
    
    Step 2: Spherical Coordinates (θ, φ)
    φ = (u / width) × 2π = {sun_info['phi']:.4f} rad
    θ = (v / height) × π = {sun_info['theta']:.4f} rad
    
    Step 3: Cartesian 3D (x, y, z)
    x = sin(θ) × cos(φ) = {sun_info['sun_dir'][0]:.4f}
    y = cos(θ) [flipped] = {sun_info['sun_dir'][1]:.4f}
    z = sin(θ) × sin(φ) = {sun_info['sun_dir'][2]:.4f}
    
    Normalized: ||dir|| = 1.0
    """
    ax4.text(0.1, 0.5, conversion_text, ha='left', va='center', fontsize=10, 
             family='monospace', transform=ax4.transAxes)
    ax4.axis('off')
    
    # 5. 3D Sun Direction Visualization
    ax5 = fig.add_subplot(gs[1, 1], projection='3d')
    
    # Draw coordinate system
    origin = [0, 0, 0]
    ax5.quiver(*origin, 1, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2, label='X')
    ax5.quiver(*origin, 0, 1, 0, color='g', arrow_length_ratio=0.1, linewidth=2, label='Y')
    ax5.quiver(*origin, 0, 0, 1, color='b', arrow_length_ratio=0.1, linewidth=2, label='Z')
    
    # Draw sun direction
    ax5.quiver(*origin, *sun_info['sun_dir'], color='orange', arrow_length_ratio=0.1, 
               linewidth=3, label='Sun Direction')
    
    # Draw unit sphere (wireframe)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax5.plot_wireframe(x, y, z, color='gray', alpha=0.2, linewidth=0.5)
    
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.set_title('3D Sun Direction Vector')
    ax5.legend(loc='upper left', fontsize=8)
    ax5.set_xlim([-1, 1])
    ax5.set_ylim([-1, 1])
    ax5.set_zlim([-1, 1])
    
    # 6. Spherical Coordinate Visualization
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Draw equirectangular projection
    img_height, img_width = linear_sky.shape[:2]
    ax6.imshow(np.clip(linear_sky, 0, 1), extent=[0, 2*np.pi, np.pi, 0])
    ax6.plot(sun_info['phi'], sun_info['theta'], 'r+', markersize=20, markeredgewidth=3)
    
    # Draw grid
    for phi_line in np.linspace(0, 2*np.pi, 9):
        ax6.axvline(phi_line, color='white', alpha=0.3, linewidth=0.5)
    for theta_line in np.linspace(0, np.pi, 5):
        ax6.axhline(theta_line, color='white', alpha=0.3, linewidth=0.5)
    
    ax6.set_xlabel('φ (azimuth) [radians]')
    ax6.set_ylabel('θ (elevation) [radians]')
    ax6.set_title(f'Spherical Coordinates\nφ={sun_info["phi"]:.3f}, θ={sun_info["theta"]:.3f}')
    ax6.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax6.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax6.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax6.set_yticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])
    
    save_fig(fig, f'01_sun_detection_frame_{frame_idx:03d}.png')
    return sun_info

def visualize_sun_trajectory(sky_folder, max_frames=None, num_sample_frames=8):
    """Visualize sun trajectory across multiple sky frames with enhanced aesthetics"""
    print(f"\n🌞 Visualizing Sun Trajectory (Enhanced with Seaborn)...")
    
    sky_files = glob.glob(os.path.join(sky_folder, "*.*"))
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.hdr', '.exr']
    sky_files = [f for f in sky_files if os.path.splitext(f)[1].lower() in valid_exts]
    sky_files = sorted(sky_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split()[-1]))
    
    # Apply max_frames limit if specified
    if max_frames is not None:
        sky_files = sky_files[:max_frames]
    
    sun_positions_uv = []
    sun_positions_sphere = []
    sun_directions_3d = []
    sky_images = []
    
    for sky_path in tqdm(sky_files, desc="Processing frames"):
        ldr_sky = cv2.imread(sky_path, cv2.IMREAD_UNCHANGED)
        if ldr_sky is None:
            continue
        
        if len(ldr_sky.shape) == 2:
            ldr_sky = cv2.cvtColor(ldr_sky, cv2.COLOR_GRAY2BGR)
        elif ldr_sky.shape[2] == 4:
            ldr_sky = ldr_sky[:, :, :3]
        
        ldr_sky = cv2.cvtColor(ldr_sky, cv2.COLOR_BGR2RGB).astype(np.float32)
        if ldr_sky.max() > 1.0:
            ldr_sky = ldr_sky / 255.0
        linear_sky = np.power(np.clip(ldr_sky, 0, 1), 2.2)
        
        sun_info = extract_sun_with_debug(linear_sky)
        sun_positions_uv.append(sun_info['sun_uv'])
        sun_positions_sphere.append((sun_info['phi'], sun_info['theta']))
        sun_directions_3d.append(sun_info['sun_dir'])
        sky_images.append(np.clip(linear_sky, 0, 1))
    
    sun_positions_uv = np.array(sun_positions_uv)
    sun_positions_sphere = np.array(sun_positions_sphere)
    sun_directions_3d = np.array(sun_directions_3d)
    
    # Select random sample frames for grid visualization
    total_frames = len(sky_files)
    if total_frames > num_sample_frames:
        sample_indices = sorted(random.sample(range(total_frames), num_sample_frames))
    else:
        sample_indices = list(range(total_frames))
    
    # Use seaborn color palette
    colors = sns.color_palette("rocket", total_frames)
    
    # ============================================
    # VISUALIZATION 1: Enhanced Trajectory Plot
    # ============================================
    fig = plt.figure(figsize=(20, 6), facecolor='white')
    
    # 1. UV Trajectory with gradient coloring
    ax1 = fig.add_subplot(1, 3, 1)
    first_sky = sky_images[0]
    ax1.imshow(first_sky, alpha=0.9)
    
    # Draw trajectory with gradient
    for i in range(len(sun_positions_uv) - 1):
        ax1.plot(sun_positions_uv[i:i+2, 0], sun_positions_uv[i:i+2, 1], 
                color=colors[i], linewidth=3, alpha=0.8)
    
    # Add scatter points with seaborn palette
    scatter = ax1.scatter(sun_positions_uv[:, 0], sun_positions_uv[:, 1], 
                         c=range(len(sun_positions_uv)), cmap='plasma', s=50, 
                         edgecolors='white', linewidths=1.5, zorder=5)
    ax1.scatter(sun_positions_uv[0, 0], sun_positions_uv[0, 1], 
               color='lime', s=150, marker='o', edgecolors='white', 
               linewidths=3, label='Start', zorder=10)
    ax1.scatter(sun_positions_uv[-1, 0], sun_positions_uv[-1, 1], 
               color='red', s=150, marker='*', edgecolors='white', 
               linewidths=3, label='End', zorder=10)
    
    ax1.set_title(f'Sun Trajectory (Image Space)\n{len(sky_files)} frames', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax1.axis('off')
    
    # 2. Spherical Coordinates with enhanced styling
    ax2 = fig.add_subplot(1, 3, 2)
    
    # Draw trajectory with gradient
    for i in range(len(sun_positions_sphere) - 1):
        ax2.plot(sun_positions_sphere[i:i+2, 0], sun_positions_sphere[i:i+2, 1], 
                color=colors[i], linewidth=3, alpha=0.8)
    
    scatter2 = ax2.scatter(sun_positions_sphere[:, 0], sun_positions_sphere[:, 1], 
                          c=range(len(sun_positions_sphere)), cmap='plasma', s=50, 
                          edgecolors='white', linewidths=1.5, zorder=5)
    ax2.scatter(sun_positions_sphere[0, 0], sun_positions_sphere[0, 1], 
               color='lime', s=150, marker='o', edgecolors='white', 
               linewidths=3, label='Start', zorder=10)
    ax2.scatter(sun_positions_sphere[-1, 0], sun_positions_sphere[-1, 1], 
               color='red', s=150, marker='*', edgecolors='white', 
               linewidths=3, label='End', zorder=10)
    
    ax2.set_xlabel('φ (azimuth) [radians]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('θ (elevation) [radians]', fontsize=12, fontweight='bold')
    ax2.set_title('Sun Trajectory (Spherical Coordinates)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax2.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    # Add colorbar
    cbar = plt.colorbar(scatter2, ax=ax2)
    cbar.set_label('Frame Number', fontsize=11, fontweight='bold')
    
    # 3. 3D Trajectory with enhanced styling
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.set_facecolor('white')
    
    # Draw coordinate system
    origin = [0, 0, 0]
    ax3.quiver(*origin, 1, 0, 0, color='#E74C3C', arrow_length_ratio=0.1, 
              linewidth=2.5, alpha=0.8, label='X-axis')
    ax3.quiver(*origin, 0, 1, 0, color='#27AE60', arrow_length_ratio=0.1, 
              linewidth=2.5, alpha=0.8, label='Y-axis')
    ax3.quiver(*origin, 0, 0, 1, color='#3498DB', arrow_length_ratio=0.1, 
              linewidth=2.5, alpha=0.8, label='Z-axis')
    
    # Draw unit sphere with better styling
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax3.plot_wireframe(x, y, z, color='lightgray', alpha=0.2, linewidth=0.5)
    
    # Draw trajectory with gradient
    for i in range(len(sun_directions_3d) - 1):
        ax3.plot(sun_directions_3d[i:i+2, 0], sun_directions_3d[i:i+2, 1], 
                sun_directions_3d[i:i+2, 2], color=colors[i], linewidth=3, alpha=0.9)
    
    # Add scatter points
    scatter3 = ax3.scatter(sun_directions_3d[:, 0], sun_directions_3d[:, 1], 
                          sun_directions_3d[:, 2], c=range(len(sun_directions_3d)), 
                          cmap='plasma', s=50, edgecolors='white', linewidths=1.5, zorder=5)
    ax3.scatter(*sun_directions_3d[0], color='lime', s=150, marker='o', 
               edgecolors='white', linewidths=3, zorder=10)
    ax3.scatter(*sun_directions_3d[-1], color='red', s=150, marker='*', 
               edgecolors='white', linewidths=3, zorder=10)
    
    ax3.set_xlabel('X', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Y', fontsize=11, fontweight='bold')
    ax3.set_zlabel('Z', fontsize=11, fontweight='bold')
    ax3.set_title('Sun Trajectory (3D Directions)', fontsize=14, fontweight='bold', pad=15)
    ax3.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    ax3.set_zlim([-1, 1])
    
    plt.tight_layout()
    save_fig(fig, '02_sun_trajectory_enhanced.png', dpi=200)
    
    # ============================================
    # VISUALIZATION 2: Random Frames Grid
    # ============================================
    print(f"  Creating grid visualization with {len(sample_indices)} random frames...")
    
    n_cols = 4
    n_rows = (len(sample_indices) + n_cols - 1) // n_cols
    
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows), facecolor='white')
    axes = axes.flatten() if len(sample_indices) > 1 else [axes]
    
    for idx, ax in enumerate(axes):
        if idx < len(sample_indices):
            frame_idx = sample_indices[idx]
            sky_img = sky_images[frame_idx]
            sun_uv = sun_positions_uv[frame_idx]
            
            ax.imshow(sky_img)
            
            # Draw trajectory up to this frame
            ax.plot(sun_positions_uv[:frame_idx+1, 0], sun_positions_uv[:frame_idx+1, 1], 
                   'yellow', linewidth=2, alpha=0.8, linestyle='--')
            
            # Highlight current position
            ax.scatter(sun_uv[0], sun_uv[1], color='red', s=200, marker='*', 
                      edgecolors='white', linewidths=2, zorder=10)
            
            # Add frame info
            phi, theta = sun_positions_sphere[frame_idx]
            sun_dir = sun_directions_3d[frame_idx]
            
            info_text = f'Frame {frame_idx}\n'
            info_text += f'UV: ({sun_uv[0]:.0f}, {sun_uv[1]:.0f})\n'
            info_text += f'φ={phi:.2f}, θ={theta:.2f}\n'
            info_text += f'Dir: ({sun_dir[0]:.2f}, {sun_dir[1]:.2f}, {sun_dir[2]:.2f})'
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='black', alpha=0.7), color='white', family='monospace')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    save_fig(fig2, '02_sun_trajectory_frames_grid.png', dpi=150)
    
    # ============================================
    # VISUALIZATION 3: Create GIF Animation
    # ============================================
    print(f"  Creating animated GIF...")
    gif_frames = []
    
    for frame_idx in tqdm(range(len(sky_images)), desc="  Generating GIF frames"):
        fig_gif, ax_gif = plt.subplots(figsize=(10, 10), facecolor='white')
        
        # Show sky
        ax_gif.imshow(sky_images[frame_idx])
        
        # Draw full trajectory
        ax_gif.plot(sun_positions_uv[:, 0], sun_positions_uv[:, 1], 
                   'yellow', linewidth=2, alpha=0.5, linestyle='--', label='Full Path')
        
        # Draw trajectory up to current frame with gradient
        if frame_idx > 0:
            for i in range(frame_idx):
                ax_gif.plot(sun_positions_uv[i:i+2, 0], sun_positions_uv[i:i+2, 1], 
                           color=colors[i], linewidth=3, alpha=0.9)
        
        # Highlight current position
        ax_gif.scatter(sun_positions_uv[frame_idx, 0], sun_positions_uv[frame_idx, 1], 
                      color='red', s=300, marker='*', edgecolors='white', 
                      linewidths=3, zorder=10, label=f'Frame {frame_idx}')
        
        # Add info box
        phi, theta = sun_positions_sphere[frame_idx]
        sun_dir = sun_directions_3d[frame_idx]
        
        info_text = f'Frame: {frame_idx}/{len(sky_images)-1}\n'
        info_text += f'Position: ({sun_positions_uv[frame_idx, 0]:.0f}, {sun_positions_uv[frame_idx, 1]:.0f})\n'
        info_text += f'Spherical: φ={phi:.3f}, θ={theta:.3f}\n'
        info_text += f'Direction: ({sun_dir[0]:.3f}, {sun_dir[1]:.3f}, {sun_dir[2]:.3f})'
        
        ax_gif.text(0.02, 0.98, info_text, transform=ax_gif.transAxes, 
                   fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='black', alpha=0.8), color='white', family='monospace',
                   fontweight='bold')
        
        ax_gif.set_title(f'Sun Trajectory Animation', fontsize=16, fontweight='bold', pad=20)
        ax_gif.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax_gif.axis('off')
        
        # Convert to PIL Image
        plt.tight_layout()
        fig_gif.canvas.draw()
        # Use buffer_rgba and convert to RGB
        img_buf = np.asarray(fig_gif.canvas.buffer_rgba())
        img_rgb = img_buf[:, :, :3]  # Drop alpha channel
        gif_frames.append(Image.fromarray(img_rgb))
        plt.close(fig_gif)
    
    # Save GIF
    save_gif(gif_frames, 'sun_trajectory_animation.gif', duration=100)
    
    # ============================================
    # VISUALIZATION 4: 3D Rotating Trajectory GIF
    # ============================================
    print(f"  Creating 3D rotating trajectory GIF...")
    gif_frames_3d = []
    
    # Create 360-degree rotation
    n_rotation_frames = 60
    for angle_idx in tqdm(range(n_rotation_frames), desc="  Generating 3D rotation"):
        fig_3d = plt.figure(figsize=(10, 10), facecolor='white')
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.set_facecolor('white')
        
        # Set viewing angle
        azim = angle_idx * 360 / n_rotation_frames
        ax_3d.view_init(elev=20, azim=azim)
        
        # Draw coordinate system
        origin = [0, 0, 0]
        ax_3d.quiver(*origin, 1, 0, 0, color='#E74C3C', arrow_length_ratio=0.1, 
                    linewidth=2.5, alpha=0.8)
        ax_3d.quiver(*origin, 0, 1, 0, color='#27AE60', arrow_length_ratio=0.1, 
                    linewidth=2.5, alpha=0.8)
        ax_3d.quiver(*origin, 0, 0, 1, color='#3498DB', arrow_length_ratio=0.1, 
                    linewidth=2.5, alpha=0.8)
        
        # Add axis labels at the ends
        ax_3d.text(1.2, 0, 0, 'X', color='#E74C3C', fontsize=14, fontweight='bold')
        ax_3d.text(0, 1.2, 0, 'Y', color='#27AE60', fontsize=14, fontweight='bold')
        ax_3d.text(0, 0, 1.2, 'Z', color='#3498DB', fontsize=14, fontweight='bold')
        
        # Draw unit sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax_3d.plot_wireframe(x, y, z, color='lightgray', alpha=0.2, linewidth=0.5)
        
        # Draw trajectory with gradient
        for i in range(len(sun_directions_3d) - 1):
            ax_3d.plot(sun_directions_3d[i:i+2, 0], sun_directions_3d[i:i+2, 1], 
                      sun_directions_3d[i:i+2, 2], color=colors[i], linewidth=3, alpha=0.9)
        
        # Add scatter points
        ax_3d.scatter(sun_directions_3d[:, 0], sun_directions_3d[:, 1], 
                     sun_directions_3d[:, 2], c=range(len(sun_directions_3d)), 
                     cmap='plasma', s=60, edgecolors='white', linewidths=1.5, zorder=5)
        
        # Highlight start and end
        ax_3d.scatter(*sun_directions_3d[0], color='lime', s=200, marker='o', 
                     edgecolors='white', linewidths=3, zorder=10)
        ax_3d.scatter(*sun_directions_3d[-1], color='red', s=200, marker='*', 
                     edgecolors='white', linewidths=3, zorder=10)
        
        ax_3d.set_xlabel('X', fontsize=12, fontweight='bold', labelpad=10)
        ax_3d.set_ylabel('Y', fontsize=12, fontweight='bold', labelpad=10)
        ax_3d.set_zlabel('Z', fontsize=12, fontweight='bold', labelpad=10)
        ax_3d.set_title(f'3D Sun Trajectory (Rotating View)\\nAngle: {azim:.1f}°', 
                       fontsize=16, fontweight='bold', pad=20)
        ax_3d.set_xlim([-1.2, 1.2])
        ax_3d.set_ylim([-1.2, 1.2])
        ax_3d.set_zlim([-1.2, 1.2])
        
        # Convert to PIL Image
        plt.tight_layout()
        fig_3d.canvas.draw()
        # Use buffer_rgba and convert to RGB
        img_buf = np.asarray(fig_3d.canvas.buffer_rgba())
        img_rgb = img_buf[:, :, :3]  # Drop alpha channel
        gif_frames_3d.append(Image.fromarray(img_rgb))
        plt.close(fig_3d)
    
    # Save 3D rotating GIF
    save_gif(gif_frames_3d, 'sun_trajectory_3d_rotating.gif', duration=50)

# ==========================================
# 2. GEOMETRY VISUALIZATION
# ==========================================
def visualize_scene_geometry():
    """Visualize scene geometry: planes, normals, and coordinate system"""
    print(f"\n🏗️ Visualizing Scene Geometry...")
    
    img = load_base_image()
    coded_map = load_coded_map()
    h, w = img.shape[:2]
    
    planes, f, (cx, cy) = get_scene_planes(w, h, VP, BACK_TL, BACK_BR)
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Segmentation Map
    ax1 = fig.add_subplot(gs[0, 0])
    colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
    colored_seg[coded_map == 1] = [255, 0, 0]     # Left wall - Red
    colored_seg[coded_map == 2] = [0, 255, 0]     # Right wall - Green
    colored_seg[coded_map == 3] = [0, 0, 255]     # Floor - Blue
    colored_seg[coded_map == 4] = [255, 255, 0]   # Back wall - Yellow
    ax1.imshow(colored_seg)
    ax1.plot(*VP, 'w+', markersize=20, markeredgewidth=3)
    ax1.plot(*BACK_TL, 'co', markersize=10)
    ax1.plot(*BACK_BR, 'mo', markersize=10)
    ax1.set_title('Segmentation Map\nRed=Left, Green=Right, Blue=Floor, Yellow=Back')
    ax1.legend(['Vanishing Point', 'Back TL', 'Back BR'], loc='upper right', fontsize=8)
    ax1.axis('off')
    
    # 2. Image with Geometry Annotations
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img)
    ax2.plot(*VP, 'r+', markersize=20, markeredgewidth=3, label='Vanishing Point')
    ax2.plot([BACK_TL[0], BACK_BR[0]], [BACK_TL[1], BACK_BR[1]], 'g-', linewidth=2)
    ax2.plot([BACK_TL[0], BACK_TL[0]], [BACK_TL[1], BACK_BR[1]], 'g-', linewidth=2)
    ax2.plot([BACK_BR[0], BACK_BR[0]], [BACK_TL[1], BACK_BR[1]], 'g-', linewidth=2)
    ax2.plot([BACK_TL[0], BACK_BR[0]], [BACK_BR[1], BACK_BR[1]], 'g-', linewidth=2, label='Back Wall')
    
    # Draw perspective lines to VP
    ax2.plot([BACK_TL[0], VP[0]], [BACK_TL[1], VP[1]], 'y--', alpha=0.6, linewidth=1)
    ax2.plot([BACK_BR[0], VP[0]], [BACK_BR[1], VP[1]], 'y--', alpha=0.6, linewidth=1)
    ax2.plot([BACK_TL[0], VP[0]], [BACK_BR[1], VP[1]], 'y--', alpha=0.6, linewidth=1)
    ax2.plot([BACK_BR[0], VP[0]], [BACK_TL[1], VP[1]], 'y--', alpha=0.6, linewidth=1)
    
    ax2.set_title('Geometry Annotations')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.axis('off')
    
    # 3. Normal Map Visualization
    ax3 = fig.add_subplot(gs[0, 2])
    normal_map = np.zeros((h, w, 3), dtype=np.float32)
    normal_map[coded_map == 3] = [0, -1, 0]   # Floor
    normal_map[coded_map == 1] = [1, 0, 0]    # Left wall
    normal_map[coded_map == 2] = [-1, 0, 0]   # Right wall
    normal_map[coded_map == 4] = [0, 0, 1]    # Back wall
    
    # Convert normals to RGB for visualization
    normal_rgb = (normal_map + 1.0) / 2.0  # Map [-1,1] to [0,1]
    ax3.imshow(normal_rgb)
    ax3.set_title('Normal Map\n(RGB encoded: R=X, G=Y, B=Z)')
    ax3.axis('off')
    
    # 4. 3D Scene Visualization
    ax4 = fig.add_subplot(gs[1, 0], projection='3d')
    
    def to_3d(px, py, z):
        return np.array([px - cx, py - cy, z], dtype=np.float32)
    
    # Back wall corners
    P_back_tl = to_3d(*BACK_TL, f)
    P_back_br = to_3d(*BACK_BR, f)
    P_back_bl = to_3d(BACK_TL[0], BACK_BR[1], f)
    P_back_tr = to_3d(BACK_BR[0], BACK_TL[1], f)
    
    # Vanishing point (camera lens, Z=0)
    P_vp = to_3d(*VP, 0)
    
    # Draw back wall
    back_wall = [P_back_tl, P_back_tr, P_back_br, P_back_bl]
    back_poly = Poly3DCollection([back_wall], alpha=0.3, facecolor='yellow', edgecolor='black', linewidth=2)
    ax4.add_collection3d(back_poly)
    
    # Draw floor
    floor_front_left = P_vp + np.array([-200, 0, 0])
    floor_front_right = P_vp + np.array([200, 0, 0])
    floor = [P_back_bl, P_back_br, floor_front_right, floor_front_left]
    floor_poly = Poly3DCollection([floor], alpha=0.3, facecolor='blue', edgecolor='black', linewidth=2)
    ax4.add_collection3d(floor_poly)
    
    # Draw left wall
    left_wall = [P_back_tl, P_back_bl, floor_front_left, P_vp]
    left_poly = Poly3DCollection([left_wall], alpha=0.3, facecolor='red', edgecolor='black', linewidth=2)
    ax4.add_collection3d(left_poly)
    
    # Draw right wall
    right_wall = [P_back_tr, P_back_br, floor_front_right, P_vp]
    right_poly = Poly3DCollection([right_wall], alpha=0.3, facecolor='green', edgecolor='black', linewidth=2)
    ax4.add_collection3d(right_poly)
    
    # Draw coordinate system at origin
    origin = np.array([0, 0, 0])
    ax4.quiver(*origin, 100, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2, label='X')
    ax4.quiver(*origin, 0, 100, 0, color='g', arrow_length_ratio=0.1, linewidth=2, label='Y')
    ax4.quiver(*origin, 0, 0, 100, color='b', arrow_length_ratio=0.1, linewidth=2, label='Z')
    
    # Mark vanishing point
    ax4.scatter(*P_vp, color='magenta', s=100, marker='*', label='VP (Camera)')
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('3D Scene Geometry')
    ax4.legend(loc='upper left', fontsize=8)
    
    # 5. Plane Equations
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.text(0.5, 0.95, 'Plane Equations', ha='center', fontsize=14, weight='bold', transform=ax5.transAxes)
    
    plane_info = ""
    for name, plane in planes.items():
        plane_info += f"\n{name.upper()} WALL:\n"
        plane_info += f"  Normal: ({plane.n[0]:.4f}, {plane.n[1]:.4f}, {plane.n[2]:.4f})\n"
        plane_info += f"  Point: ({plane.p[0]:.2f}, {plane.p[1]:.2f}, {plane.p[2]:.2f})\n"
        plane_info += f"  Equation: {plane.n[0]:.4f}x + {plane.n[1]:.4f}y + {plane.n[2]:.4f}z + {plane.d:.4f} = 0\n"
    
    ax5.text(0.05, 0.85, plane_info, ha='left', va='top', fontsize=9, 
             family='monospace', transform=ax5.transAxes)
    ax5.axis('off')
    
    # 6. Camera Ray Visualization
    ax6 = fig.add_subplot(gs[1, 2], projection='3d')
    
    # Sample some pixels and draw their rays
    camera_origin = np.array([0, 0, 0])
    sample_points = [
        (w//4, h//2, 'Floor Left'),
        (w//2, h//2, 'Back Wall Center'),
        (3*w//4, h//2, 'Floor Right'),
        (w//2, h//4, 'Back Wall Top'),
    ]
    
    colors = ['cyan', 'yellow', 'magenta', 'orange']
    
    for idx, (px, py, label) in enumerate(sample_points):
        pixel_ray = np.array([px - cx, py - cy, f], dtype=np.float32)
        pixel_ray = pixel_ray / (np.linalg.norm(pixel_ray) + 1e-8)
        
        # Find intersection with planes
        min_t = float('inf')
        hit_point = None
        
        for plane in planes.values():
            t = plane.intersect_ray(camera_origin, pixel_ray)
            if t is not None and 0 < t < min_t and np.isfinite(t):
                min_t = t
                hit_point = camera_origin + t * pixel_ray
        
        if hit_point is not None:
            # Draw ray
            ray_points = np.array([camera_origin, hit_point])
            ax6.plot(ray_points[:, 0], ray_points[:, 1], ray_points[:, 2], 
                     color=colors[idx], linewidth=2, alpha=0.7, label=label)
            ax6.scatter(*hit_point, color=colors[idx], s=50, marker='o')
    
    # Draw simplified scene
    ax6.scatter(*origin, color='black', s=100, marker='o', label='Camera')
    
    # Draw back wall outline
    back_outline = np.array([P_back_tl, P_back_tr, P_back_br, P_back_bl, P_back_tl])
    ax6.plot(back_outline[:, 0], back_outline[:, 1], back_outline[:, 2], 'k-', linewidth=2, alpha=0.5)
    
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Z')
    ax6.set_title('Camera Rays & Plane Intersections')
    ax6.legend(loc='upper left', fontsize=7)
    
    save_fig(fig, '03_scene_geometry.png', dpi=200)

# ==========================================
# 3. SHADOW COMPUTATION VISUALIZATION
# ==========================================
def visualize_shadow_computation(sky_path, frame_idx=0):
    """Visualize shadow ray tracing process"""
    print(f"\n☀️ Visualizing Shadow Computation (Frame {frame_idx})...")
    
    img = load_base_image()
    coded_map = load_coded_map()
    alpha_mask = load_alpha_mask()
    h, w = img.shape[:2]
    
    planes, f, (cx, cy) = get_scene_planes(w, h, VP, BACK_TL, BACK_BR)
    
    # Load sky and extract sun
    ldr_sky = cv2.imread(sky_path, cv2.IMREAD_UNCHANGED)
    if ldr_sky is None:
        print(f"  ✗ Failed to load {sky_path}")
        return
    
    if len(ldr_sky.shape) == 2:
        ldr_sky = cv2.cvtColor(ldr_sky, cv2.COLOR_GRAY2BGR)
    elif ldr_sky.shape[2] == 4:
        ldr_sky = ldr_sky[:, :, :3]
    
    ldr_sky = cv2.cvtColor(ldr_sky, cv2.COLOR_BGR2RGB).astype(np.float32)
    if ldr_sky.max() > 1.0:
        ldr_sky = ldr_sky / 255.0
    linear_sky = np.power(np.clip(ldr_sky, 0, 1), 2.2)
    
    sun_info = extract_sun_with_debug(linear_sky)
    sun_dir = sun_info['sun_dir']
    
    # Compute floor 3D positions (sampled)
    camera_origin = np.array([0, 0, 0], dtype=np.float32)
    floor_pixels_3d = {}
    floor_mask = (coded_map == 3)
    floor_coords = np.argwhere(floor_mask)
    floor_coords_sampled = floor_coords[::FLOOR_SAMPLE_RATE]
    
    for py, px in floor_coords_sampled[:100]:  # Limit for visualization
        pixel_ray = np.array([px - cx, py - cy, f], dtype=np.float32)
        pixel_ray = pixel_ray / (np.linalg.norm(pixel_ray) + 1e-8)
        t_hit = planes['floor'].intersect_ray(camera_origin, pixel_ray)
        if t_hit is not None and t_hit > 0 and np.isfinite(t_hit):
            floor_pixels_3d[(py, px)] = camera_origin + t_hit * pixel_ray
    
    # Compute shadow map
    shadow_map = np.zeros((h, w), dtype=np.float32)
    occluder_planes = [('left', planes['left'], 1), ('right', planes['right'], 2), ('back', planes['back'], 4)]
    
    BIAS_DIST = 2.0
    shadow_rays = []  # For 3D visualization
    
    if np.dot(planes['floor'].n, sun_dir) > 0:
        for (py, px), P_floor in list(floor_pixels_3d.items())[:20]:  # Sample for viz
            P_start = P_floor + (sun_dir * BIAS_DIST)
            is_shadowed = False
            hit_point = None
            
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
                                hit_point = P_hit
                                break
                        else:
                            is_shadowed = True
                            hit_point = P_hit
                            break
            
            if is_shadowed:
                shadow_map[py, px] = 1.0
                if hit_point is not None:
                    shadow_rays.append((P_floor, hit_point, True))
            else:
                shadow_rays.append((P_floor, P_start + sun_dir * 500, False))
    
    # Blur shadow map
    kernel_size = 21
    shadow_map_blurred = cv2.GaussianBlur(shadow_map, (kernel_size, kernel_size), 0)
    
    # Create visualization
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # 2. Floor Mask
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(floor_mask, cmap='gray')
    ax2.set_title('Floor Region (Ray Tracing Target)')
    ax2.axis('off')
    
    # 3. Raw Shadow Map
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(shadow_map, cmap='Reds', vmin=0, vmax=1)
    ax3.set_title(f'Raw Shadow Map\n(Sampled every {FLOOR_SAMPLE_RATE} pixels)')
    ax3.axis('off')
    
    # 4. Blurred Shadow Map
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(shadow_map_blurred, cmap='Reds', vmin=0, vmax=1)
    ax4.set_title(f'Blurred Shadow Map\n(Kernel Size: {kernel_size})')
    ax4.axis('off')
    
    # 5. Shadow Overlay
    ax5 = fig.add_subplot(gs[1, 1])
    img_with_shadow = img.copy().astype(np.float32) / 255.0
    shadow_overlay = 1.0 - (shadow_map_blurred * 0.75)
    img_with_shadow = img_with_shadow * shadow_overlay[:, :, None]
    ax5.imshow(np.clip(img_with_shadow, 0, 1))
    ax5.set_title('Image with Shadows Applied')
    ax5.axis('off')
    
    # 6. 3D Ray Tracing Visualization
    ax6 = fig.add_subplot(gs[1, 2], projection='3d')
    
    # Draw floor points
    floor_points = np.array(list(floor_pixels_3d.values()))
    ax6.scatter(floor_points[:, 0], floor_points[:, 1], floor_points[:, 2], 
                c='blue', s=10, alpha=0.5, label='Floor Points')
    
    # Draw shadow rays
    for P_floor, P_end, is_shadowed in shadow_rays:
        color = 'red' if is_shadowed else 'green'
        alpha = 0.6 if is_shadowed else 0.3
        ray_points = np.array([P_floor, P_end])
        ax6.plot(ray_points[:, 0], ray_points[:, 1], ray_points[:, 2], 
                 color=color, linewidth=1, alpha=alpha)
    
    # Draw sun direction (scaled)
    sun_arrow_start = np.array([0, -100, 300])
    sun_arrow_end = sun_arrow_start + sun_dir * 200
    ax6.quiver(*sun_arrow_start, *(sun_dir * 200), color='orange', 
               arrow_length_ratio=0.1, linewidth=3, label='Sun Direction')
    
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Z')
    ax6.set_title(f'Shadow Ray Tracing\nRed=Occluded, Green=Lit')
    ax6.legend(loc='upper left', fontsize=8)
    
    save_fig(fig, f'04_shadow_computation_frame_{frame_idx:03d}.png', dpi=150)

# ==========================================
# 4. LIGHTING VISUALIZATION
# ==========================================
def visualize_lighting(sky_path, frame_idx=0):
    """Visualize lighting computation and color blending"""
    print(f"\n💡 Visualizing Lighting (Frame {frame_idx})...")
    
    img_srgb = cv2.imread(IMG_PATH).astype(np.float32) / 255.0
    base_img = np.power(img_srgb, 2.2)  # Linear space
    coded_map = load_coded_map()
    h, w = base_img.shape[:2]
    
    planes, f, (cx, cy) = get_scene_planes(w, h, VP, BACK_TL, BACK_BR)
    
    # Load sky and extract sun
    ldr_sky = cv2.imread(sky_path, cv2.IMREAD_UNCHANGED)
    if ldr_sky is None:
        print(f"  ✗ Failed to load {sky_path}")
        return
    
    if len(ldr_sky.shape) == 2:
        ldr_sky = cv2.cvtColor(ldr_sky, cv2.COLOR_GRAY2BGR)
    elif ldr_sky.shape[2] == 4:
        ldr_sky = ldr_sky[:, :, :3]
    
    ldr_sky = cv2.cvtColor(ldr_sky, cv2.COLOR_BGR2RGB).astype(np.float32)
    if ldr_sky.max() > 1.0:
        ldr_sky = ldr_sky / 255.0
    linear_sky = np.power(np.clip(ldr_sky, 0, 1), 2.2)
    
    sun_info = extract_sun_with_debug(linear_sky)
    sun_dir = sun_info['sun_dir']
    
    # Lighting parameters from pipeline
    SUN_INTENSITY = 2.5
    SUN_COLOR = np.array([1.0, 0.95, 0.85])
    AMBIENT_INTENSITY = 0.3
    AMBIENT_COLOR = np.array([0.6, 0.65, 0.7])
    
    # Compute normal map
    normal_map = np.zeros((h, w, 3), dtype=np.float32)
    normal_map[coded_map == 3] = [0, -1, 0]   # Floor
    normal_map[coded_map == 1] = [1, 0, 0]    # Left wall
    normal_map[coded_map == 2] = [-1, 0, 0]   # Right wall
    normal_map[coded_map == 4] = [0, 0, 1]    # Back wall
    
    # Compute lighting
    sun_dir_broadcast = sun_dir.reshape(1, 1, 3)
    n_dot_l = np.sum(normal_map * sun_dir_broadcast, axis=2)
    n_dot_l = np.clip(n_dot_l, 0, 1)
    
    # Simplified (no shadows for clarity)
    direct_intensity = SUN_INTENSITY * n_dot_l
    total_intensity = AMBIENT_INTENSITY + direct_intensity
    
    # Color blending
    direct_contribution = direct_intensity / (total_intensity + 1e-8)
    ambient_contribution = AMBIENT_INTENSITY / (total_intensity + 1e-8)
    total_contribution = direct_contribution + ambient_contribution
    direct_contribution = direct_contribution / (total_contribution + 1e-8)
    ambient_contribution = ambient_contribution / (total_contribution + 1e-8)
    
    light_color = (direct_contribution[:, :, None] * SUN_COLOR.reshape(1, 1, 3) + 
                   ambient_contribution[:, :, None] * AMBIENT_COLOR.reshape(1, 1, 3))
    
    # Apply lighting
    lit_image = base_img * total_intensity[:, :, None] * light_color
    
    # Tone map
    def aces_tonemap(x):
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
    
    lit_image = aces_tonemap(lit_image)
    lit_image = np.power(lit_image, 1.0/2.2)  # Back to sRGB
    
    # Create visualization with enhanced seaborn styling
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Use seaborn color palettes
    cmap_heat = sns.color_palette("rocket", as_cmap=True)
    cmap_cool = sns.color_palette("mako", as_cmap=True)
    
    # 1. Normal Map (RGB encoded)
    ax1 = fig.add_subplot(gs[0, 0])
    normal_rgb = (normal_map + 1.0) / 2.0
    ax1.imshow(normal_rgb)
    ax1.set_title('Normal Map\n(R=X, G=Y, B=Z)', fontsize=12, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # 2. N·L (Diffuse Term)
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(n_dot_l, cmap=cmap_heat, vmin=0, vmax=1)
    cbar2 = plt.colorbar(im, ax=ax2, fraction=0.046)
    cbar2.set_label('Intensity', fontsize=10, fontweight='bold')
    ax2.set_title(f'N·L (Diffuse Term)\nSun: ({sun_dir[0]:.2f}, {sun_dir[1]:.2f}, {sun_dir[2]:.2f})', 
                 fontsize=12, fontweight='bold', pad=10)
    ax2.axis('off')
    
    # 3. Direct Intensity
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(direct_intensity, cmap=cmap_heat)
    cbar3 = plt.colorbar(im, ax=ax3, fraction=0.046)
    cbar3.set_label('Intensity', fontsize=10, fontweight='bold')
    ax3.set_title(f'Direct Light Intensity\n(Sun × N·L = {SUN_INTENSITY} × N·L)', 
                 fontsize=12, fontweight='bold', pad=10)
    ax3.axis('off')
    
    # 4. Total Intensity
    ax4 = fig.add_subplot(gs[1, 0])
    im = ax4.imshow(total_intensity, cmap=cmap_heat)
    cbar4 = plt.colorbar(im, ax=ax4, fraction=0.046)
    cbar4.set_label('Intensity', fontsize=10, fontweight='bold')
    ax4.set_title(f'Total Intensity\n(Ambient {AMBIENT_INTENSITY} + Direct)', 
                 fontsize=12, fontweight='bold', pad=10)
    ax4.axis('off')
    
    # 5. Light Color Map
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(np.clip(light_color, 0, 1))
    ax5.set_title('Light Color Tint\n(Sun + Ambient Blend)', 
                 fontsize=12, fontweight='bold', pad=10)
    ax5.axis('off')
    
    # 6. Color Legend
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.text(0.5, 0.95, 'Lighting Parameters', ha='center', fontsize=13, 
            weight='bold', transform=ax6.transAxes)
    
    params_text = f"""
    SUN LIGHT:
    Intensity: {SUN_INTENSITY}
    Color: RGB({SUN_COLOR[0]:.2f}, {SUN_COLOR[1]:.2f}, {SUN_COLOR[2]:.2f})
    Direction: ({sun_dir[0]:.3f}, {sun_dir[1]:.3f}, {sun_dir[2]:.3f})
    
    AMBIENT LIGHT:
    Intensity: {AMBIENT_INTENSITY}
    Color: RGB({AMBIENT_COLOR[0]:.2f}, {AMBIENT_COLOR[1]:.2f}, {AMBIENT_COLOR[2]:.2f})
    
    BLENDING:
    Direct Contrib: intensity-weighted
    Ambient Contrib: intensity-weighted
    Final = Base × Intensity × Color
    """
    ax6.text(0.05, 0.82, params_text, ha='left', va='top', fontsize=9, 
             family='monospace', transform=ax6.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax6.axis('off')
    
    # 7. Original (Linear)
    ax7 = fig.add_subplot(gs[2, 0])
    base_srgb = np.power(base_img, 1.0/2.2)
    ax7.imshow(np.clip(base_srgb, 0, 1))
    ax7.set_title('Original Image (sRGB)', fontsize=12, fontweight='bold', pad=10)
    ax7.axis('off')
    
    # 8. Lit Result
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.imshow(np.clip(lit_image, 0, 1))
    ax8.set_title('Final Lit Image', fontsize=12, fontweight='bold', pad=10)
    ax8.axis('off')
    
    # 9. Intensity Only (Grayscale)
    ax9 = fig.add_subplot(gs[2, 2])
    intensity_only = base_img * total_intensity[:, :, None]
    intensity_only = aces_tonemap(intensity_only)
    intensity_only = np.power(intensity_only, 1.0/2.2)
    ax9.imshow(np.clip(intensity_only, 0, 1))
    ax9.set_title('Intensity Only (No Color Tint)', fontsize=12, fontweight='bold', pad=10)
    ax9.axis('off')
    
    plt.tight_layout()
    save_fig(fig, f'05_lighting_frame_{frame_idx:03d}.png', dpi=150)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    """Run all visualizations with enhanced aesthetics"""
    print("="*60)
    print("SCENE RELIGHTING PIPELINE VISUALIZATION")
    print("Enhanced with Seaborn Aesthetics & GIF Animations")
    print("="*60)
    
    # Get sky files
    sky_files = glob.glob(os.path.join(SKY_FOLDER, "*.*"))
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.hdr', '.exr']
    sky_files = [f for f in sky_files if os.path.splitext(f)[1].lower() in valid_exts]
    sky_files = sorted(sky_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split()[-1]))
    
    if len(sky_files) == 0:
        print(f"❌ No sky images found in {SKY_FOLDER}")
        return
    
    print(f"\n✓ Found {len(sky_files)} sky images")
    print(f"✓ Visualizations will be saved to '{VIZ_FOLDER}'")
    print(f"✓ GIF animations will be saved to '{GIF_FOLDER}'")
    
    # 1. Sun Detection (first frame)
    visualize_sun_detection(sky_files[0], frame_idx=0)
    
    # 2. Sun Trajectory (all frames with enhanced styling and GIFs)
    # Process all available frames, sample 8 random frames for grid
    visualize_sun_trajectory(SKY_FOLDER, max_frames=None, num_sample_frames=8)
    
    # 3. Scene Geometry (static)
    visualize_scene_geometry()
    
    # 4. Shadow Computation (first frame)
    visualize_shadow_computation(sky_files[0], frame_idx=0)
    
    # 5. Lighting (first frame)
    visualize_lighting(sky_files[0], frame_idx=0)
    
    # Optional: Generate for middle and last frame too
    if len(sky_files) > 2:
        mid_idx = len(sky_files) // 2
        print(f"\nGenerating additional visualizations for frame {mid_idx}...")
        visualize_sun_detection(sky_files[mid_idx], frame_idx=mid_idx)
        visualize_shadow_computation(sky_files[mid_idx], frame_idx=mid_idx)
        visualize_lighting(sky_files[mid_idx], frame_idx=mid_idx)
        
        last_idx = len(sky_files) - 1
        print(f"\nGenerating additional visualizations for frame {last_idx}...")
        visualize_sun_detection(sky_files[last_idx], frame_idx=last_idx)
        visualize_shadow_computation(sky_files[last_idx], frame_idx=last_idx)
        visualize_lighting(sky_files[last_idx], frame_idx=last_idx)
    
    print("\n" + "="*60)
    print(f"✅ ALL VISUALIZATIONS COMPLETE!")
    print(f"📁 Static images: '{VIZ_FOLDER}'")
    print(f"🎬 Animated GIFs: '{GIF_FOLDER}'")
    print("="*60)
    print("\nGenerated Files:")
    print("  • Sun detection with coordinate conversion")
    print("  • Enhanced sun trajectory (static)")
    print("  • Random frames grid visualization")
    print("  • Sun trajectory 2D animation (GIF)")
    print("  • Sun trajectory 3D rotating animation (GIF)")
    print("  • Scene geometry with 3D planes")
    print("  • Shadow computation ray tracing")
    print("  • Lighting breakdown and analysis")
    print("="*60)

if __name__ == "__main__":
    main()
