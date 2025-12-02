"""
Figure 8: Sky 1 Results - Morning-to-noon Sequence Visualization

This script generates a comprehensive visualization showing:
(a) Sample sky frames with sun elevation progression
(b) Corresponding relit building images with dynamic shadows
(c) Close-up shadow regions highlighting soft edge quality
(d) Temporal consistency analysis with frame-to-frame differences
(e) Final composited results with sky backgrounds

Based on the pipeline.py processing for Sky 1 sequence.
"""

import cv2
import numpy as np
import os
import glob
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from tqdm import tqdm

# Add src directory to path for imports
sys.path.append('../src')
from geometry_classes import get_scene_planes

# Set aesthetic style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.0)

# Configuration matching pipeline.py
VP = (427, 393)
BACK_TL = (362, 322)
BACK_BR = (509, 404)
IMG_PATH = '../images/img.jpeg'
CODED_MAP_PATH = '../coded_id_map.npy'
ALPHA_MASK_PATH = '../images/mask.jpeg'
SKY_FOLDER = '../sky1'
OUTPUT_FOLDER = '../final_output'

def extract_sun_with_debug(sky_img):
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
    
    # Convert to spherical coordinates and then to 3D direction
    phi = (sun_u / sky_w) * 2 * np.pi
    theta = (sun_v / sky_h) * np.pi
    
    sun_x = np.sin(theta) * np.cos(phi)
    sun_y = np.cos(theta)
    sun_z = np.sin(theta) * np.sin(phi)
    
    sun_dir = np.array([sun_x, sun_y, sun_z], dtype=np.float32)
    sun_dir = sun_dir / (np.linalg.norm(sun_dir) + 1e-8)
    sun_dir[1] = -sun_dir[1]  # Invert Y for rendering coordinate system
    
    # Calculate elevation angle
    elevation = np.degrees(np.arcsin(np.abs(sun_dir[1])))
    
    return sun_dir, elevation, (int(sun_u), int(sun_v))

def load_processed_frames():
    """Load processed frames from video file"""
    video_path = os.path.join(OUTPUT_FOLDER, 'sky1.mp4')
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found. Please run pipeline.py first.")
        return []
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames

def compute_frame_differences(frames):
    """Compute frame-to-frame differences for temporal consistency analysis"""
    if len(frames) < 2:
        return []
    
    differences = []
    psnr_values = []
    
    for i in range(1, len(frames)):
        # Convert to float for computation
        prev_frame = frames[i-1].astype(np.float32) / 255.0
        curr_frame = frames[i].astype(np.float32) / 255.0
        
        # Compute difference
        diff = np.abs(curr_frame - prev_frame)
        differences.append((diff * 255).astype(np.uint8))
        
        # Compute PSNR
        mse = np.mean((prev_frame - curr_frame) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        psnr_values.append(psnr)
    
    return differences, psnr_values

def create_sky1_results_figure():
    """Create the comprehensive Sky 1 results figure"""
    print("="*60)
    print("GENERATING SKY 1 RESULTS FIGURE")
    print("="*60)
    
    # Load sky sequence
    sky_files = glob.glob(os.path.join(SKY_FOLDER, "*.jpeg"))
    sky_files = sorted(sky_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    if len(sky_files) == 0:
        print("Error: No sky files found!")
        return
    
    # Skip last 10 frames as in the combined trajectory script
    sky_files = sky_files[:-10]
    
    # Load processed frames
    processed_frames = load_processed_frames()
    if len(processed_frames) == 0:
        return
    
    # Select key frames for visualization (start, middle, end)
    n_frames = len(sky_files)
    key_indices = [0, n_frames//3, 2*n_frames//3, n_frames-1]
    
    # Analyze sun progression
    sun_elevations = []
    sky_samples = []
    
    print("Analyzing sun progression...")
    for i, sky_path in enumerate(tqdm(sky_files)):
        if i not in key_indices:
            continue
            
        ldr_sky = cv2.imread(sky_path, cv2.IMREAD_UNCHANGED)
        if ldr_sky is None:
            continue
            
        ldr_sky = cv2.cvtColor(ldr_sky, cv2.COLOR_BGR2RGB).astype(np.float32)
        if ldr_sky.max() > 1.0:
            ldr_sky = ldr_sky / 255.0
        linear_sky = np.power(np.clip(ldr_sky, 0, 1), 2.2)
        
        sun_dir, elevation, sun_pos = extract_sun_with_debug(linear_sky)
        sun_elevations.append(elevation)
        sky_samples.append(linear_sky)
    
    # Compute temporal consistency
    differences, psnr_values = compute_frame_differences(processed_frames)
    avg_psnr = np.mean(psnr_values) if psnr_values else 0
    
    # Create the figure
    print("Creating comprehensive figure...")
    fig = plt.figure(figsize=(20, 16), facecolor='white')
    
    # Define grid layout: 3 rows x 5 columns
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.2,
                          height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1, 1])
    
    # Row 1: (a) Sample sky frames showing sun elevation progression
    sky_axes = []
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        if i < len(sky_samples):
            ax.imshow(np.power(sky_samples[i], 1/2.2))  # Convert back to sRGB for display
            ax.set_title(f'Frame {key_indices[i]+1}\nElevation: {sun_elevations[i]:.1f}°', 
                        fontsize=10, fontweight='bold')
        ax.axis('off')
        sky_axes.append(ax)
    
    # Add (a) label
    sky_axes[0].text(-0.15, 1.05, '(a)', transform=sky_axes[0].transAxes, 
                    fontsize=14, fontweight='bold', va='top', ha='right')
    
    # Row 1, Col 5: Sun elevation plot
    ax_sun = fig.add_subplot(gs[0, 4])
    frame_numbers = [key_indices[i]+1 for i in range(len(sun_elevations))]
    ax_sun.plot(frame_numbers, sun_elevations, 'o-', color='orange', linewidth=2, markersize=8)
    ax_sun.set_xlabel('Frame Number', fontweight='bold')
    ax_sun.set_ylabel('Sun Elevation (°)', fontweight='bold')
    ax_sun.set_title('Sun Elevation\nProgression', fontsize=10, fontweight='bold')
    ax_sun.grid(True, alpha=0.3)
    
    # Row 2: (b) Corresponding relit building images
    relit_axes = []
    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        if i < len(key_indices) and key_indices[i] < len(processed_frames):
            ax.imshow(processed_frames[key_indices[i]])
            ax.set_title(f'Relit Frame {key_indices[i]+1}', fontsize=10, fontweight='bold')
        ax.axis('off')
        relit_axes.append(ax)
    
    # Add (b) label
    relit_axes[0].text(-0.15, 1.05, '(b)', transform=relit_axes[0].transAxes, 
                      fontsize=14, fontweight='bold', va='top', ha='right')
    
    # Row 2, Col 5: (c) Close-up shadow regions
    ax_shadow = fig.add_subplot(gs[1, 4])
    if len(processed_frames) > 0:
        # Extract a shadow region (bottom portion of the building)
        sample_frame = processed_frames[len(processed_frames)//2]
        h, w = sample_frame.shape[:2]
        shadow_crop = sample_frame[int(0.7*h):h, int(0.2*w):int(0.8*w)]
        ax_shadow.imshow(shadow_crop)
        ax_shadow.set_title('Shadow Detail\n(Soft Edges)', fontsize=10, fontweight='bold')
    ax_shadow.axis('off')
    
    # Add (c) label
    ax_shadow.text(-0.15, 1.05, '(c)', transform=ax_shadow.transAxes, 
                  fontsize=14, fontweight='bold', va='top', ha='right')
    
    # Row 3: (d) Temporal consistency analysis
    if differences:
        # Show frame differences
        diff_axes = []
        for i in range(min(3, len(differences))):
            ax = fig.add_subplot(gs[2, i])
            diff_idx = i * (len(differences) // 3) if len(differences) >= 3 else i
            if diff_idx < len(differences):
                ax.imshow(differences[diff_idx])
                ax.set_title(f'Diff {diff_idx+1}-{diff_idx+2}', fontsize=10, fontweight='bold')
            ax.axis('off')
            diff_axes.append(ax)
        
        # Add (d) label
        if diff_axes:
            diff_axes[0].text(-0.15, 1.05, '(d)', transform=diff_axes[0].transAxes, 
                             fontsize=14, fontweight='bold', va='top', ha='right')
    
    # Row 3, Col 4: PSNR plot
    ax_psnr = fig.add_subplot(gs[2, 3])
    if psnr_values:
        ax_psnr.plot(range(1, len(psnr_values)+1), psnr_values, '-', color='blue', linewidth=2)
        ax_psnr.axhline(y=avg_psnr, color='red', linestyle='--', 
                       label=f'Avg: {avg_psnr:.1f} dB')
        ax_psnr.set_xlabel('Frame Transition', fontweight='bold')
        ax_psnr.set_ylabel('PSNR (dB)', fontweight='bold')
        ax_psnr.set_title('Temporal\nConsistency', fontsize=10, fontweight='bold')
        ax_psnr.legend(fontsize=8)
        ax_psnr.grid(True, alpha=0.3)
    
    # Row 3, Col 5: (e) Final composite with color temperature info
    ax_final = fig.add_subplot(gs[2, 4])
    if len(processed_frames) > 0:
        # Show a representative final result
        final_frame = processed_frames[len(processed_frames)//3]
        ax_final.imshow(final_frame)
        ax_final.set_title('Final Composite\nwith Sky Background', fontsize=10, fontweight='bold')
    ax_final.axis('off')
    
    # Add (e) label
    ax_final.text(-0.15, 1.05, '(e)', transform=ax_final.transAxes, 
                 fontsize=14, fontweight='bold', va='top', ha='right')
    
    # Add main title and subtitle
    fig.suptitle('Sky 1 Results - Morning-to-noon Sequence', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Add descriptive subtitle
    subtitle = f'Sun elevation progression: {sun_elevations[0]:.1f}° → {sun_elevations[-1]:.1f}° | '
    subtitle += f'Temporal consistency: {avg_psnr:.1f} dB PSNR | '
    subtitle += f'Total frames: {len(processed_frames)}'
    
    fig.text(0.5, 0.94, subtitle, ha='center', va='top', fontsize=12, 
            style='italic', color='gray')
    
    # Save the figure
    output_path = 'sky1_results.png'
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"\n✅ Sky 1 results figure saved to: {output_path}")
    print(f"   - Sun elevation range: {sun_elevations[0]:.1f}° → {sun_elevations[-1]:.1f}°")
    print(f"   - Average temporal consistency: {avg_psnr:.1f} dB PSNR")
    print(f"   - Total frames processed: {len(processed_frames)}")
    print("="*60)

def main():
    """Main execution function"""
    # Change to the paper_figures directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    create_sky1_results_figure()

if __name__ == "__main__":
    main()