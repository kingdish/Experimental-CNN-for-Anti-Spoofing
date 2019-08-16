from models import mynet
import torch
import os, cv2
from PIL import Image
from torchvision import datasets, transforms

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_MODEL = os.path.join(MAIN_DIR, "pretrained", "MyNetv2_WebFace_112_SURF.pth")
IMG_SIZE = 112

infer_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(80),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465, 0.5), (0.2023, 0.1994, 0.2010, 0.2)),
])


def get_img(rgb_img, ir_img):
    color_img = cv2.imread(rgb_img)
    ir_img = cv2.imread(ir_img, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.resize(color_img, (IMG_SIZE, IMG_SIZE))
    ir_img = cv2.resize(ir_img, (IMG_SIZE, IMG_SIZE))
    img = cv2.merge([color_img, ir_img])
    img = Image.fromarray(img)  # TODO: use either cv2 or PIL, not both
    return infer_transform(img).unsqueeze(0)


def main(rgb_img, ir_img):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    img = get_img(rgb_img, ir_img)
    net = mynet.MyNetv2Webface(PRETRAINED_MODEL)
    net = net.eval()
    net.to(device)


    with torch.no_grad():
        output = net(img)
        _, predicted = output.max(1)
        print(predicted.item())


if __name__ == "__main__":
    main("dataset/Testing/0000/000000-color.jpg", "dataset/Testing/0000/000000-ir.jpg")