from torch import nn


class FaceRecognitionNet(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.base(x)
        return self.classifier(x)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    model.eval()
    print("Model loaded successfully.")
    return model

def create_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
    for param in model.parameters():
        param.requires_grad = False

    return FaceRecognitionNet(model, len(class_names)).to(device)
