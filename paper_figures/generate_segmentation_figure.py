import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_image(path):
    """Load image in RGB format"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_to_match(images):
    """Resize all images to the same dimensions (matching the first image)"""
    target_height, target_width = images[0].shape[:2]
    resized = []
    for img in images:
        if img.shape[:2] != (target_height, target_width):
            img_resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            resized.append(img_resized)
        else:
            resized.append(img)
    return resized

def main():
    # Load the three images
    # Adjust paths as needed based on where your images are stored
    original = load_image('../images/img.jpeg')  # Original input image
    annotated = load_image('../images/annotated.jpeg')  # Image with SAM annotations (green/red dots + mask overlay)
    mask = load_image('../images/mask.jpeg')  # Final binary mask
    
    # Resize all images to match dimensions
    original, annotated, mask = resize_to_match([original, annotated, mask])
    
    # Create figure with 1x3 layout
    fig = plt.figure(figsize=(18, 6), facecolor='#f0f0f0')
    gs = GridSpec(1, 3, figure=fig, wspace=0.15)
    
    # Panel (a) - Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original)
    ax1.set_title('Original Input Image', fontsize=14, fontweight='bold', pad=10)
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, 
             fontsize=16, fontweight='bold', va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.axis('off')
    ax1.set_facecolor('#e0e0e0')
    
    # Panel (b) - User Annotation with SAM
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(annotated)
    ax2.set_title('SAM Interactive Annotation\n(Green: Foreground, Red: Background)', 
                  fontsize=14, fontweight='bold', pad=10)
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, 
             fontsize=16, fontweight='bold', va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.axis('off')
    ax2.set_facecolor('#e0e0e0')
    
    # Panel (c) - Final Mask
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(mask, cmap='gray')
    ax3.set_title('Final Foreground Mask', fontsize=14, fontweight='bold', pad=10)
    ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes, 
             fontsize=16, fontweight='bold', va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.axis('off')
    ax3.set_facecolor('#e0e0e0')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the combined figure
    output_path = 'segmentation_process.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#f0f0f0')
    print(f"✓ Saved segmentation figure to: {output_path}")
    print(f"✓ All images resized to: {original.shape[1]}x{original.shape[0]} pixels")
    
    plt.close()

if __name__ == "__main__":
    main()
