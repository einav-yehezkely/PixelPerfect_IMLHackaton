import torch
import torch.nn as nn
import torchvision.models as models



class ModelArchitecture(nn.Module):
    """
    Student model architecture.

    Students should define their model here.

    Required behavior:
        input:  torch.Tensor of shape [batch_size, 3, height, width]
        output: torch.Tensor of shape [batch_size, 20]
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        # TODO: write your model architecture here
        # Example:
        #   define layers
        #   define feature extractor
        #   define classifier
        #   define any other modules needed

        super().__init__()

        self.model = models.resnet18(weights=None)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # raise NotImplementedError("TODO: implement ModelArchitecture.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: batch of images

        Returns:
            logits for 20 classes
        """

        # TODO: write the forward pass here
        # The returned tensor should have shape [batch_size, 20]
        return self.model(x)

        # raise NotImplementedError("TODO: implement ModelArchitecture.forward")