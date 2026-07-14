"""
Feature map visualization — extract intermediate conv layer activations.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def get_feature_maps(model, x):
    """Extract feature maps from intermediate conv layers using hooks."""
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if name in ('conv1', 'conv2', 'conv3'):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    model.eval()
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return activations


def plot_one_layer(fmaps, layer_name, save_dir, max_channels=16):
    """Plot feature maps for a single conv layer."""
    n_channels = min(fmaps.shape[1], max_channels)
    cols = 8
    rows = (n_channels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 1.8 * rows))
    fig.suptitle(f'{layer_name} — {fmaps.shape[1]} channels (showing {n_channels})',
                 fontsize=13, y=1.02)

    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_channels):
        fmap = fmaps[0, i].numpy()
        axes[i].imshow(fmap, cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Ch{i}', fontsize=7)

    for i in range(n_channels, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'feature_maps_{layer_name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from model import lenet5
    from dataset import get_dataloaders

    device = torch.device('cpu')
    model = lenet5(10).to(device)
    model.load_state_dict(torch.load(
        '../experiments/best_model.pth', map_location='cpu', weights_only=True
    ))

    loaders = get_dataloaders(batch_size=1, data_root='../data')
    test_loader = loaders['test_loader']

    # Get one sample of a moderately complex digit
    found = None
    for images, labels in test_loader:
        if labels[0].item() == 3:
            found = images
            break
    if found is None:
        found = next(iter(test_loader))[0]

    x = found.to(device)
    acts = get_feature_maps(model, x)
    print(f"Extracted layers: {list(acts.keys())}")
    for name, fmaps in acts.items():
        print(f"  {name}: shape={fmaps.shape}")

    save_dir = '../experiments/figures'
    for name in ['conv1', 'conv2', 'conv3']:
        if name in acts:
            n_ch = acts[name].shape[1]
            plot_one_layer(acts[name], name, save_dir,
                           max_channels=16 if n_ch > 16 else n_ch)
    print("Done!")
