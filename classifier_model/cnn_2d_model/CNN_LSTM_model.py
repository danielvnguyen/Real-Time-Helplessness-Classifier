# 2D CNN + LSTM model using grayscale

import torch
import torch.nn as nn

# adapted from: https://github.com/HHTseng/video-classification
# adapted from: https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
# adapted from: https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/

class HelplessnessClassifier(nn.Module):
    """
    2D CNN + LSTM for Grayscale frames:
      - (B, T, 1, H, W) input
      - CNN processes each frame => feature vector
      - LSTM processes the time dimension
      - Final FC for classification
    """

    def __init__(self, num_classes=3):
        super(HelplessnessClassifier, self).__init__()

        # 1) 2D CNN blocks with in_channels=1 for grayscale
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # H/2, W/2
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # H/4, W/4
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # H/8, W/8
        )

        # Dummy pass to find feature dim
        dummy = torch.zeros(1, 1, 112, 112)  # single frame, 1 channel, e.g. 112x112
        out = self._forward_cnn(dummy)
        feature_dim = out.shape[1]  # (1, feature_dim)
        print(f"[DEBUG] CNN feature_dim = {feature_dim}")

        # LSTM: transforms sequence of CNN features => final hidden
        self.lstm_hidden = 128
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=self.lstm_hidden,
                            num_layers=1, batch_first=True)

        # Classifier
        self.fc = nn.Linear(self.lstm_hidden, num_classes)

    def _forward_cnn(self, x):
        """ x: (N, 1, H, W), returns flattened features shape (N, feature_dim) """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        """
        x shape: (B, T, 1, H, W)
          1) Merge B,T => (B*T, 1, H, W)
          2) CNN => (B*T, feature_dim)
          3) Reshape => (B, T, feature_dim)
          4) LSTM => (B, T, hidden)
          5) final hidden => FC => (B, num_classes)
        """
        B, T, C, H, W = x.shape  # C=1 for grayscale
        x = x.view(B * T, C, H, W)
        feats = self._forward_cnn(x)  # => (B*T, feature_dim)

        feats = feats.view(B, T, -1)  # => (B, T, feature_dim)
        lstm_out, _ = self.lstm(feats)  # => (B, T, lstm_hidden)

        last_out = lstm_out[:, -1, :]  # => (B, lstm_hidden)
        logits = self.fc(last_out)  # => (B, num_classes)
        return logits


if __name__ == "__main__":
    model = HelplessnessClassifier(num_classes=3)
    test_input = torch.zeros(2, 10, 1, 112, 112)  # (batch=2, T=10, 1 channel, 112x112)
    out = model(test_input)
    print("Output shape:", out.shape)  # => (2, 3)