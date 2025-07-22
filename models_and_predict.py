import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from PIL import Image

class FakeFacesDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # Blocu 1: 256 -> 128
        self.conv1a = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Blocu 2: 128 -> 64
        self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Blocu 3: 64 -> 32
        self.conv3a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Blocu 4: 32 -> 16
        self.conv4a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4b = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Blocu 5: 16 -> 8
        self.conv5a = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5b = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # (batch, 512, 1, 1)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Bloc 1
        x = F.leaky_relu(self.conv1a(x))
        x = F.leaky_relu(self.bn1(self.conv1b(x)))
        x = F.max_pool2d(x, 2)

        # Bloc 2
        x = F.leaky_relu(self.conv2a(x))
        x = F.leaky_relu(self.bn2(self.conv2b(x)))
        x = F.max_pool2d(x, 2)

        # Bloc 3
        x = F.leaky_relu(self.conv3a(x))
        x = F.leaky_relu(self.bn3(self.conv3b(x)))
        x = F.max_pool2d(x, 2)

        # Bloc 4
        x = F.leaky_relu(self.conv4a(x))
        x = F.leaky_relu(self.bn4(self.conv4b(x)))
        x = F.max_pool2d(x, 2)

        # Bloc 5
        x = F.leaky_relu(self.conv5a(x))
        x = F.leaky_relu(self.bn5(self.conv5b(x)))
        x = F.max_pool2d(x, 2)

        # GAP + fully connected
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)

        return x

class ObjectPersonDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bnorm4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bnorm5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bnorm6 = nn.BatchNorm2d(1024)
        self.conv7 = nn.Conv2d(1024, 2048, 3, padding=1)
        self.bnorm7 = nn.BatchNorm2d(2048)
        self.conv8 = nn.Conv2d(2048, 1024, 3, padding=1)
        self.bnorm8 = nn.BatchNorm2d(1024)
        self.fc1 = nn.Linear(1*1*1024, 1024)  
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 1)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.bnorm1(self.conv1(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.bnorm2(self.conv2(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.bnorm3(self.conv3(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.bnorm4(self.conv4(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.bnorm5(self.conv5(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.bnorm6(self.conv6(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.bnorm7(self.conv7(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.bnorm8(self.conv8(x))), 2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def makeTheNetConvNeXt():
    net = timm.create_model("convnext_tiny", pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    in_features = net.head.in_features
    net.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(in_features, 1)
    )
    for param in net.head.parameters():
        param.requires_grad = True
    return net

device = torch.device("cpu")
MODEL_OO = "models/ClasificareObiectOameniV2cuMaiMulteDate.pt"
MODEL_FACE = "models/fakeFaces1.pt"
MODEL_OBJ = "models/ObiecteFakeVSObiecteRealeConvNextTiny.pt"

model_oo = ObjectPersonDetector()
model_oo.load_state_dict(torch.load(MODEL_OO, map_location=device))
model_oo.eval()

model_face = FakeFacesDetector()
model_face.load_state_dict(torch.load(MODEL_FACE, map_location=device))
model_face.eval()

model_obj = makeTheNetConvNeXt()
model_obj.load_state_dict(torch.load(MODEL_OBJ, map_location=device))
model_obj.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict_all(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        out_oo = model_oo(img_tensor)
        score_oo = torch.sigmoid(out_oo).item()
        is_person = score_oo >= 0.5

    if is_person:
        with torch.no_grad():
            out_face = model_face(img_tensor)
            score_face = torch.sigmoid(out_face).item()
        verdict = "real" if score_face >= 0.6 else "fake"
        return "persoana", verdict, score_face
    else:
        with torch.no_grad():
            out_obj = model_obj(img_tensor)
            score_obj = torch.sigmoid(out_obj).item()
        verdict = "real" if score_obj >= 0.5 else "fake"
        return "obiect", verdict, score_obj
