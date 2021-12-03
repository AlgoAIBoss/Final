import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from sklearn import preprocessing

from base64 import b64encode
from PIL import Image
import requests
le = preprocessing.LabelEncoder()

with open("Animal/inferece/name of the animals.txt") as f:
    classes = f.read().split('\n')

le.fit(classes)


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # Use pre-trained model
        self.model = models.densenet161(pretrained=True)

        # Freeze all layers (No training)
        for param in self.parameters():
            param.requires_grad = False

        # Change final FC layer to num_classes output. This is trainable by default
        self.model.classifier = nn.Linear(2208, 90)

    def forward(self, x):
        x = self.model(x)
        return x


device = torch.device('cpu')

model = NeuralNet().to(device)


def predict(data):
    state = torch.load('Animal/inferece/model0.pth', map_location=device)

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    with torch.no_grad():
        image = Image.open(data).convert('RGB')
        img_tensor = tfms(image).to(device).unsqueeze(0)
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1)
        prediction = le.inverse_transform(prediction.cpu())

    return prediction[0]


def reader(photo):
    data = photo.read()
    encoded = b64encode(data).decode()
    mime = 'image/jpeg;'
    picture = "data:%sbase64,%s" % (mime, encoded)

    return picture
