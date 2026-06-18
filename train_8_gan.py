import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# ==========================
# SETTINGS
# ==========================

IMAGE_SIZE = 64
BATCH_SIZE = 16
LATENT_DIM = 100
EPOCHS = 500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

# ==========================
# DATASET
# ==========================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder(
    root="Dataset",
    transform=transform
)

print("Classes:", dataset.classes)
print("Images found:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ==========================
# GENERATOR
# ==========================

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            nn.ConvTranspose2d(
                LATENT_DIM,
                512,
                4,
                1,
                0,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                512,
                256,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                256,
                128,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                128,
                64,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                64,
                1,
                4,
                2,
                1,
                bias=False
            ),

            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# ==========================
# DISCRIMINATOR
# ==========================

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            nn.Conv2d(
                1,
                64,
                4,
                2,
                1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                64,
                128,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                128,
                256,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                256,
                512,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                512,
                1,
                4,
                1,
                0,
                bias=False
            ),

            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1)

# ==========================
# CREATE MODELS
# ==========================

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

criterion = nn.BCELoss()

optimizer_G = optim.Adam(
    G.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

optimizer_D = optim.Adam(
    D.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

# ==========================
# TRAINING LOOP
# ==========================

print("\nTraining Started...\n")

for epoch in range(EPOCHS):

    for real_images, _ in loader:

        real_images = real_images.to(DEVICE)

        batch_size = real_images.size(0)

        real_labels = torch.ones(
            batch_size,
            device=DEVICE
        )

        fake_labels = torch.zeros(
            batch_size,
            device=DEVICE
        )

        # =====================
        # TRAIN DISCRIMINATOR
        # =====================

        noise = torch.randn(
            batch_size,
            LATENT_DIM,
            1,
            1,
            device=DEVICE
        )

        fake_images = G(noise)

        real_output = D(real_images)
        fake_output = D(fake_images.detach())

        loss_real = criterion(
            real_output,
            real_labels
        )

        loss_fake = criterion(
            fake_output,
            fake_labels
        )

        loss_D = loss_real + loss_fake

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # =====================
        # TRAIN GENERATOR
        # =====================

        output = D(fake_images)

        loss_G = criterion(
            output,
            real_labels
        )

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"D Loss: {loss_D.item():.4f} "
        f"G Loss: {loss_G.item():.4f}"
    )

    if (epoch + 1) % 50 == 0:

        with torch.no_grad():

            sample_noise = torch.randn(
                16,
                LATENT_DIM,
                1,
                1,
                device=DEVICE
            )

            samples = G(sample_noise)

            save_image(
                samples,
                f"samples_epoch_{epoch+1}.png",
                nrow=4,
                normalize=True
            )

# ==========================
# SAVE MODEL
# ==========================

torch.save(
    G.state_dict(),
    "generator_8.pth"
)

print("\nTraining Finished!")
print("Saved model: generator_8.pth")