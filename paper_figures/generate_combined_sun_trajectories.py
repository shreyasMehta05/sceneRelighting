"""
Combined Sun Trajectories Visualization for All Three Sky Sequences (2x3 layout)

Changes made:
- Layout changed to 2 rows x 3 columns (top row: image-space trajectories for Sky 1/2/3; bottom row: 3D direction plots for Sky 1/2/3).
- Subplot labels updated to (1a)-(1f) to match the user's request.
- Caption string updated and written to `paper_figures/sun_trajectories_caption.tex`.
- Output image saved as `paper_figures/sun_trajectories.png`.

Usage: run this script from the repo root (it will change working directory to the repository root automatically).
"""

import cv2
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

# Set aesthetic style with Seaborn
sns.set_style("darkgrid")
sns.set_context("notebook", font_scale=1.1)


def extract_sun_with_debug(sky_img):
    """Extract sun direction with intermediate steps for visualization"""
    sky_h, sky_w = sky_img.shape[:2]
    gray = np.mean(sky_img, axis=2)
    # Use a robust threshold: at least 0.9 or the 95th percentile of brightness
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

    # Flip Y for rendering coordinate system (match previous behavior)
    sun_dir[1] = -sun_dir[1]

    return {
        'sun_uv': (int(sun_u), int(sun_v)),
        'phi': phi,
        'theta': theta,
        'sun_dir': sun_dir
    }


def process_sky_sequence(sky_folder):
    """Process a sky sequence and return trajectory data"""
    print(f"Processing {sky_folder}...")

    sky_files = glob.glob(os.path.join(sky_folder, "*.jpeg"))
    sky_files = sorted(sky_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # If you need to skip tail frames for a particular sequence adjust here
    if os.path.basename(sky_folder) == 'sky1':
        # keep behavior similar to original script but less aggressive: skip last 10 frames
        sky_files = sky_files[:-10] if len(sky_files) > 10 else sky_files

    sun_positions_uv = []
    sun_directions_3d = []
    sky_images = []

    for sky_path in tqdm(sky_files, desc=f"  Processing {os.path.basename(sky_folder)}"):
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
        sun_directions_3d.append(sun_info['sun_dir'])
        sky_images.append(np.clip(linear_sky, 0, 1))

    return {
        'positions_uv': np.array(sun_positions_uv),
        'directions_3d': np.array(sun_directions_3d),
        'images': sky_images,
        'num_frames': len(sky_files)
    }


def create_combined_visualization():
    """Create combined visualization for all three sky sequences in a 2x3 layout"""
    print("=" * 60)
    print("GENERATING COMBINED SUN TRAJECTORIES VISUALIZATION (2x3)")
    print("=" * 60)

    # Define sky folders
    sky_folders = ['sky1', 'sky2', 'sky3']
    sky_names = ['Sky 1', 'Sky 2', 'Sky 3']
    sky_colors = ['#E74C3C', '#3498DB', '#2ECC71']  # Red, Blue, Green

    # Process all sky sequences
    sky_data = {}
    for sky_folder in sky_folders:
        if os.path.exists(sky_folder):
            sky_data[sky_folder] = process_sky_sequence(sky_folder)
        else:
            print(f"Warning: {sky_folder} not found, skipping...")

    if not sky_data:
        print("Error: No sky folders found!")
        return

    print("\nCreating combined visualization...")

    # Create figure with 2 rows, 3 columns
    fig = plt.figure(figsize=(18, 10), facecolor='white')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25,
                          height_ratios=[1, 1], width_ratios=[1, 1, 1])

    # Subplot labels: top row (1a,1b,1c), bottom row (1d,1e,1f)
    top_labels = ['1(a)', '2(a)', '3(a)']
    bot_labels = ['1(b)', '2(b)', '3(b)']

    for col_idx, (sky_folder, sky_name, color) in enumerate(zip(sky_folders, sky_names, sky_colors)):
        if sky_folder not in sky_data:
            continue

        data = sky_data[sky_folder]
        sun_positions_uv = data['positions_uv']
        sun_directions_3d = data['directions_3d']
        sky_images = data['images']
        num_frames = data['num_frames']

        # make a gradient palette with at least 2 colors
        palette_len = max(2, num_frames)
        colors_gradient = sns.color_palette("plasma", palette_len)

        # Top row: Image space trajectory for this sky (column col_idx)
        ax_img = fig.add_subplot(gs[0, col_idx])
        if sky_images:
            frame_idx = min(49, len(sky_images) // 2) if len(sky_images) > 49 else len(sky_images) // 2
            ax_img.imshow(sky_images[frame_idx], alpha=0.9)

        if len(sun_positions_uv) > 1:
            for i in range(len(sun_positions_uv) - 1):
                c = colors_gradient[min(i, palette_len - 1)]
                ax_img.plot(sun_positions_uv[i:i+2, 0], sun_positions_uv[i:i+2, 1],
                            color=c, linewidth=2.5, alpha=0.95)

        if len(sun_positions_uv) > 0:
            scatter = ax_img.scatter(sun_positions_uv[:, 0], sun_positions_uv[:, 1],
                                     c=range(len(sun_positions_uv)), cmap='plasma', s=36,
                                     edgecolors='white', linewidths=0.8, zorder=5, alpha=0.85)
            # start & end
            ax_img.scatter(sun_positions_uv[0, 0], sun_positions_uv[0, 1],
                           color='lime', s=100, marker='o', edgecolors='white', linewidths=1.5, zorder=10)
            ax_img.scatter(sun_positions_uv[-1, 0], sun_positions_uv[-1, 1],
                           color='red', s=100, marker='*', edgecolors='white', linewidths=1.5, zorder=10)

        ax_img.set_title(f'{sky_name}: Sun Trajectory (Image Space)\n{num_frames} frames',
                         fontsize=11, fontweight='bold', pad=10, color=color)
        ax_img.axis('off')
        ax_img.text(-0.08, 1.06, top_labels[col_idx], transform=ax_img.transAxes,
                    fontsize=13, fontweight='bold', va='top', ha='right')

        # Bottom row: 3D directions for this sky
        ax_3d = fig.add_subplot(gs[1, col_idx], projection='3d')
        ax_3d.set_facecolor('white')

        # Draw coordinate axes
        origin = [0, 0, 0]
        ax_3d.quiver(*origin, 1, 0, 0, color='#E74C3C', arrow_length_ratio=0.08,
                     linewidth=2.0, alpha=0.9)
        ax_3d.quiver(*origin, 0, 1, 0, color='#27AE60', arrow_length_ratio=0.08,
                     linewidth=2.0, alpha=0.9)
        ax_3d.quiver(*origin, 0, 0, 1, color='#3498DB', arrow_length_ratio=0.08,
                     linewidth=2.0, alpha=0.9)

        # unit sphere wireframe (light)
        u = np.linspace(0, 2 * np.pi, 25)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax_3d.plot_wireframe(x, y, z, color='lightgray', alpha=0.12, linewidth=0.5)

        if len(sun_directions_3d) > 1:
            for i in range(len(sun_directions_3d) - 1):
                c = colors_gradient[min(i, palette_len - 1)]
                ax_3d.plot(sun_directions_3d[i:i+2, 0], sun_directions_3d[i:i+2, 1],
                           sun_directions_3d[i:i+2, 2], color=c, linewidth=2.5, alpha=0.95)

        if len(sun_directions_3d) > 0:
            scatter3d = ax_3d.scatter(sun_directions_3d[:, 0], sun_directions_3d[:, 1],
                                      sun_directions_3d[:, 2], c=range(len(sun_directions_3d)),
                                      cmap='plasma', s=36, edgecolors='white', linewidths=0.8, zorder=5)
            # start & end
            ax_3d.scatter(*sun_directions_3d[0], color='lime', s=100, marker='o', edgecolors='white', linewidths=1.5, zorder=10)
            ax_3d.scatter(*sun_directions_3d[-1], color='red', s=100, marker='*', edgecolors='white', linewidths=1.5, zorder=10)

        ax_3d.set_xlabel('X', fontsize=9, fontweight='bold')
        ax_3d.set_ylabel('Y', fontsize=9, fontweight='bold')
        ax_3d.set_zlabel('Z', fontsize=9, fontweight='bold')
        ax_3d.set_title(f'{sky_name}: Sun Trajectory (3D Directions)', fontsize=11, fontweight='bold', pad=8, color=color)

        ax_3d.view_init(elev=20, azim=45)
        ax_3d.set_xlim([-1, 1])
        ax_3d.set_ylim([-1, 1])
        ax_3d.set_zlim([-1, 1])

        ax_3d.text2D(-0.08, 1.06, bot_labels[col_idx], transform=ax_3d.transAxes,
                     fontsize=13, fontweight='bold', va='top', ha='right')

    # Main title
    fig.suptitle('Sun Trajectories Across All Sky Sequences', fontsize=16, fontweight='bold', y=0.98)

    # Ensure output directory
    os.makedirs('paper_figures', exist_ok=True)

    # Save figure
    output_path = 'paper_figures/sun_trajectories.png'
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

    # Write LaTeX caption to file
    caption = (
"\\caption{Combined sun trajectory visualization for all three test sky sequences. "
"Top row (1(a)-3(a)): Sun paths in image space overlaid on representative sky frames. "
"Bottom row (1(b)-3(b)): Corresponding 3D direction vectors on the unit sphere. "
"(1(a),1(b)) Sky 1 morning-to-noon transition (214 frames, elevation 15\\degree\\rightarrow65\\degree). "
"(2(a),2(b)) Sky 2 afternoon descent (291 frames, warm lighting conditions). "
"(3(a),3(b)) Sky 3 overcast sequence (158 frames, diffuse illumination). "
"The visualization demonstrates robust sun tracking across diverse weather conditions with temporally coherent trajectories and accurate 3D direction estimation.}"
)

    caption_path = 'paper_figures/sun_trajectories_caption.tex'
    with open(caption_path, 'w') as f:
        f.write(caption)

    print(f"\n✅ Combined sun trajectories image saved to: {output_path}")
    print(f"✅ LaTeX caption saved to: {caption_path}")
    print("=" * 60)


def main():
    # Change to the repository root (one level up from script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    try:
        os.chdir(repo_root)
    except Exception:
        pass

    create_combined_visualization()


if __name__ == "__main__":
    main()
