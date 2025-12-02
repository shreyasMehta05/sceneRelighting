"""
Visualization Gallery for Pipeline Outputs

This script generates a comprehensive gallery showing:
(a) 3D sun trajectory with temporal evolution across frames
(b) Interactive geometry annotation interface (vanishing point and walls)
(c) Ray tracing analysis with shadow coverage
(d) Shadow trajectory across multiple frames
(e) Temporal consistency metrics across sky sequences

Combines existing visualization outputs into a publication-ready figure.
"""

import cv2
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns

# Set aesthetic style
sns.set_style("white")
sns.set_context("paper", font_scale=0.9)

def load_image(path):
    """Load an image and convert to RGB"""
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def create_visualization_gallery():
    """Create comprehensive visualization gallery"""
    print("="*60)
    print("GENERATING VISUALIZATION GALLERY")
    print("="*60)
    
    # Create figure with custom grid layout
    fig = plt.figure(figsize=(20, 12), facecolor='white')
    
    # Define grid: 2 rows x 3 columns
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.2,
                          height_ratios=[1, 1], width_ratios=[1, 1, 1])
    
    # Subplot labels
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    # ===================================================================
    # (a) 3D Sun Trajectory Animation - Use enhanced trajectory from viz1
    # ===================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    trajectory_path = '../viz1/02_sun_trajectory_enhanced.png'
    img_a = load_image(trajectory_path)
    if img_a is not None:
        ax_a.imshow(img_a)
        ax_a.set_title('3D Sun Trajectory\n(Temporal Solar Path Evolution)', 
                      fontsize=11, fontweight='bold', pad=10)
    else:
        ax_a.text(0.5, 0.5, 'Trajectory visualization\nnot available', 
                 ha='center', va='center', fontsize=10)
    ax_a.axis('off')
    ax_a.text(-0.1, 1.05, labels[0], transform=ax_a.transAxes, 
             fontsize=14, fontweight='bold', va='top', ha='right')
    
    # ===================================================================
    # (b) Geometry Annotation Interface - Use scene geometry visualization
    # ===================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    geometry_path = '../viz1/03_scene_geometry.png'
    img_b = load_image(geometry_path)
    if img_b is not None:
        ax_b.imshow(img_b)
        ax_b.set_title('Geometry Annotation Interface\n(Vanishing Point & Wall Specification)', 
                      fontsize=11, fontweight='bold', pad=10)
    else:
        ax_b.text(0.5, 0.5, 'Geometry annotation\nnot available', 
                 ha='center', va='center', fontsize=10)
    ax_b.axis('off')
    ax_b.text(-0.1, 1.05, labels[1], transform=ax_b.transAxes, 
             fontsize=14, fontweight='bold', va='top', ha='right')
    
    # ===================================================================
    # (c) Ray Tracing Analysis - Use shadow computation visualization
    # ===================================================================
    ax_c = fig.add_subplot(gs[0, 2])
    shadow_path = '../viz1/04_shadow_computation_frame_079.png'
    img_c = load_image(shadow_path)
    if img_c is not None:
        ax_c.imshow(img_c)
        ax_c.set_title('Ray Tracing Analysis\n(Hit Statistics & Shadow Coverage)', 
                      fontsize=11, fontweight='bold', pad=10)
    else:
        ax_c.text(0.5, 0.5, 'Ray tracing analysis\nnot available', 
                 ha='center', va='center', fontsize=10)
    ax_c.axis('off')
    ax_c.text(-0.1, 1.05, labels[2], transform=ax_c.transAxes, 
             fontsize=14, fontweight='bold', va='top', ha='right')
    
    # ===================================================================
    # (d) Shadow Trajectory Across Frames - Create composite from user's images
    # ===================================================================
    ax_d = fig.add_subplot(gs[1, 0])
    
    # Load the shadow trajectory images from the attachment
    shadow_frames = [
        '../shadow_analysis/shadow_map.png',
        '../shadow_analysis/shadow_breakdown.png',
        '../shadow_analysis/shadow_analysis.png'
    ]
    
    shadow_img = load_image(shadow_frames[2] if os.path.exists(shadow_frames[2]) else shadow_frames[0])
    if shadow_img is not None:
        ax_d.imshow(shadow_img)
        ax_d.set_title('Shadow Trajectory Across Frames\n(Temporal Shadow Evolution)', 
                      fontsize=11, fontweight='bold', pad=10)
    else:
        ax_d.text(0.5, 0.5, 'Shadow trajectory\nnot available', 
                 ha='center', va='center', fontsize=10)
    ax_d.axis('off')
    ax_d.text(-0.1, 1.05, labels[3], transform=ax_d.transAxes, 
             fontsize=14, fontweight='bold', va='top', ha='right')
    
    # ===================================================================
    # (e) Lighting Parameter Evolution - Use lighting frame visualization
    # ===================================================================
    ax_e = fig.add_subplot(gs[1, 1])
    lighting_path = '../viz1/05_lighting_frame_079.png'
    img_e = load_image(lighting_path)
    if img_e is not None:
        ax_e.imshow(img_e)
        ax_e.set_title('Lighting Parameter Evolution\n(Intensity & Color Temperature)', 
                      fontsize=11, fontweight='bold', pad=10)
    else:
        ax_e.text(0.5, 0.5, 'Lighting parameters\nnot available', 
                 ha='center', va='center', fontsize=10)
    ax_e.axis('off')
    ax_e.text(-0.1, 1.05, labels[4], transform=ax_e.transAxes, 
             fontsize=14, fontweight='bold', va='top', ha='right')
    
    # ===================================================================
    # (f) Temporal Consistency Metrics - Use sun trajectory frames grid
    # ===================================================================
    ax_f = fig.add_subplot(gs[1, 2])
    frames_grid_path = '../viz1/02_sun_trajectory_frames_grid.png'
    img_f = load_image(frames_grid_path)
    if img_f is not None:
        ax_f.imshow(img_f)
        ax_f.set_title('Temporal Consistency Metrics\n(Sky Sequence Analysis)', 
                      fontsize=11, fontweight='bold', pad=10)
    else:
        ax_f.text(0.5, 0.5, 'Temporal metrics\nnot available', 
                 ha='center', va='center', fontsize=10)
    ax_f.axis('off')
    ax_f.text(-0.1, 1.05, labels[5], transform=ax_f.transAxes, 
             fontsize=14, fontweight='bold', va='top', ha='right')
    
    # Add main title
    fig.suptitle('Pipeline Visualization Gallery: Complete System Overview', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add subtitle
    subtitle = 'Comprehensive visualization of sun tracking, geometry reconstruction, shadow computation, and temporal analysis'
    fig.text(0.5, 0.94, subtitle, ha='center', va='top', fontsize=11, 
            style='italic', color='gray')
    
    # Save the figure
    output_path = 'visualization_gallery.png'
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"\n✅ Visualization gallery saved to: {output_path}")
    print("="*60)

def main():
    """Main execution function"""
    # Change to the paper_figures directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    create_visualization_gallery()

if __name__ == "__main__":
    main()
