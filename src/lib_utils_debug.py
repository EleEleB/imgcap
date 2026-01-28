import matplotlib.pyplot as plt

def save_batch_images(tensor, filename):
    """
    tensor: torch.Tensor of shape [B, 3, H, W]
    filename: output image path (e.g. "batch.png")
    """
    tensor = tensor.detach().cpu()

    B, C, H, W = tensor.shape
    fig, axes = plt.subplots(1, B, figsize=(B * 2, 2))

    if B == 1:
        axes = [axes]

    for i in range(B):
        img = tensor[i].permute(1, 2, 0)  # (H, W, C)
        img = img.clamp(0, 1)            # if normalized to [0,1]
        axes[i].imshow(img)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)