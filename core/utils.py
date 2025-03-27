import matplotlib.pyplot as plt

def visualize_sets(x, y, idx_to_label, selected_classes, title=""):
    """
    Visualize a grid of images along with their labels.
    
    Parameters:
      x: Tensor of images.
      y: Tensor of labels.
      idx_to_label: Dict mapping label indices to label names.
      selected_classes: List of class indices selected in the experiment.
      title (str): Title for the plot.
    """
    # Create a 5x5 grid
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    fig.suptitle(title, fontsize=20)
    for i, (xi, yi) in enumerate(zip(x, y)):
        # Denormalize image for display.
        np_img = (xi * 0.5 + 0.5).permute(1, 2, 0).numpy()
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        ax.imshow(np_img)
        # Use a truncated label name.
        label_text = idx_to_label.get(selected_classes[int(yi.item())], "N/A").split('(')[0][:20]
        ax.set_xlabel(label_text, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()