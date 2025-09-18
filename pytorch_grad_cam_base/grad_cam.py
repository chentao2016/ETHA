import numpy as np
from pytorch_grad_cam_base.base_cam import BaseCAM
import pdb


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        if grads.ndim != 4:
            w = int(np.sqrt(grads.shape[0]-1))
            grads = grads[1:].transpose(1, 2, 0)
            grads = grads.reshape((grads.shape[0], grads.shape[1], w, w))

        return np.mean(grads, axis=(2, 3))
