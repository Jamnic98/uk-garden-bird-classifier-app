import torch.nn as nn

# --- Improved CNN from scratch ---
class CNN(nn.Module):
    def __init__(self, in_layer=3, out_layer=11):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.MaxPool2d(2)      # halves width/height
            )

        self.features = nn.Sequential(
            block(in_layer, 32),     # 32 filters
            block(32, 64),           # 64 filters
            block(64, 128),          # 128 filters
            block(128, 256)          # 256 filters
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, out_layer)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
