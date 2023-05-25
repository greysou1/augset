import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

from segment_anything import sam_model_registry
import numpy as np

model_checkpoint = '../sam_vit_h_4b8939.pth'
sam = sam_model_registry["vit_h"](checkpoint=model_checkpoint).to(device='cuda')

images = np.zeros((2, 256, 256, 3), dtype='uint8')

batched_input = []
for i in range(images.shape[0]):
    batched_input.append(
        {
            'image': torch.as_tensor(images[i], device=sam.device).permute(2, 0, 1).contiguous(),
            'original_size': images[i].shape[:2],
            'boxes': torch.as_tensor([20, 20, 20, 20], device=sam.device)
        }
    )

batched_output = sam(batched_input, multimask_output=False)
print(batched_output)