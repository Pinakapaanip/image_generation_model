import torch
import torch.nn as nn
from torchvision.utils import save_image

LATENT_DIM = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


G = Generator().to(DEVICE)

G.load_state_dict(
    torch.load("generator_8.pth", map_location=DEVICE)
)

G.eval()

with torch.no_grad():

    noise = torch.randn(
        25,
        LATENT_DIM,
        1,
        1,
        device=DEVICE
    )

    images = G(noise)

    save_image(
        images,
        "generated_8s.png",
        nrow=5,
        normalize=True
    )

print("Generated!")