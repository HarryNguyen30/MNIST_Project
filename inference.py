from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F


ArrayLike = Union[np.ndarray, List[List[float]], List[float]]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def preprocess_image(
    image: Union[str, Path, Image.Image, ArrayLike],
    invert: bool = False,
) -> np.ndarray:
    """
    Convert an input image to MNIST-like format: grayscale 28x28 float32 in [0,1].
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        arr = np.asarray(image)
        if arr.ndim == 1:
            arr = arr.reshape(28, 28)
        img = Image.fromarray(arr.astype(np.uint8))

    img = img.convert("L").resize((28, 28), Image.Resampling.BILINEAR)
    if invert:
        img = ImageOps.invert(img)
    x = np.asarray(img, dtype=np.float32) / 255.0
    return x


def _topk_from_scores(scores: np.ndarray, k: int = 3) -> List[Dict[str, float]]:
    idx = np.argsort(scores)[::-1][:k]
    return [{"digit": int(i), "score": float(scores[i])} for i in idx]


@dataclass
class LeastSquaresClassifier:
    """
    Least-squares MNIST classifier.
    Expects weight matrix W of shape (785, 10):
    - 784 image features
    - 1 bias feature (constant 1 appended)
    """

    W: np.ndarray

    @classmethod
    def from_npy(cls, weight_path: Union[str, Path]) -> "LeastSquaresClassifier":
        W = np.load(weight_path)
        if W.shape != (785, 10):
            raise ValueError(f"Expected W shape (785, 10), got {W.shape}")
        return cls(W=W)

    def predict(self, image: Union[str, Path, Image.Image, ArrayLike]) -> Dict:
        x = preprocess_image(image).reshape(-1)  # (784,)
        x_bias = np.hstack([x, [1.0]])  # (785,)
        raw_scores = x_bias @ self.W  # (10,)
        pred = int(np.argmax(raw_scores))
        return {
            "model": "least_squares",
            "predicted_digit": pred,
            "top3": _topk_from_scores(raw_scores, k=3),
        }


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes: int = 10, grayscale: bool = True):
        super().__init__()
        self.inplanes = 64
        in_dim = 1 if grayscale else 3

        self.conv1 = nn.Conv2d(
            in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2.0 / n) ** 0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


def resnet34(num_classes: int = 10) -> ResNet:
    # Keep architecture aligned with your training notebook.
    return ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes, grayscale=True)


@dataclass
class ResNetClassifier:
    model: ResNet
    device: torch.device

    @classmethod
    def from_state_dict(
        cls,
        state_dict_path: Union[str, Path],
        device: Union[str, torch.device, None] = None,
    ) -> "ResNetClassifier":
        chosen_device = torch.device(device) if device is not None else get_device()
        model = resnet34(num_classes=10)
        state = torch.load(state_dict_path, map_location=chosen_device)
        model.load_state_dict(state)
        model.to(chosen_device)
        model.eval()
        return cls(model=model, device=chosen_device)

    def predict(self, image: Union[str, Path, Image.Image, ArrayLike]) -> Dict:
        x = preprocess_image(image).astype(np.float32)  # (28, 28) in [0,1]
        tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, probas = self.model(tensor)
            probs = probas[0].detach().cpu().numpy()
        pred = int(np.argmax(probs))
        return {
            "model": "resnet",
            "predicted_digit": pred,
            "top3": _topk_from_scores(probs, k=3),
        }


class MnistInferenceService:
    def __init__(
        self,
        least_squares_weight_path: Union[str, Path, None] = None,
        resnet_state_dict_path: Union[str, Path, None] = None,
        device: Union[str, torch.device, None] = None,
    ):
        self.least_squares = (
            LeastSquaresClassifier.from_npy(least_squares_weight_path)
            if least_squares_weight_path
            else None
        )
        self.resnet = (
            ResNetClassifier.from_state_dict(resnet_state_dict_path, device=device)
            if resnet_state_dict_path
            else None
        )

    def predict(
        self,
        image: Union[str, Path, Image.Image, ArrayLike],
        model_name: str = "resnet",
    ) -> Dict:
        key = model_name.strip().lower()
        if key in {"least_squares", "ls", "linear"}:
            if self.least_squares is None:
                raise ValueError("Least-squares model is not loaded.")
            return self.least_squares.predict(image)
        if key in {"resnet", "cnn"}:
            if self.resnet is None:
                raise ValueError("ResNet model is not loaded.")
            return self.resnet.predict(image)
        raise ValueError(f"Unknown model_name: {model_name}")


if __name__ == "__main__":
    # Example:
    # python inference.py --image path/to/digit.png --model resnet
    import argparse
    import json

    parser = argparse.ArgumentParser(description="MNIST inference for least-squares and ResNet")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="resnet", choices=["resnet", "least_squares"])
    parser.add_argument("--resnet-path", default="models/resnet_mnist_state_dict.pt")
    parser.add_argument("--ls-path", default="models/least_squares_W.npy")
    parser.add_argument("--invert", action="store_true", help="Invert colors if needed")
    args = parser.parse_args()

    # Optional invert override
    image_obj = Image.open(args.image)
    if args.invert:
        image_obj = ImageOps.invert(image_obj.convert("L"))

    service = MnistInferenceService(
        least_squares_weight_path=args.ls_path if Path(args.ls_path).exists() else None,
        resnet_state_dict_path=args.resnet_path if Path(args.resnet_path).exists() else None,
        device=None,
    )
    result = service.predict(image_obj, model_name=args.model)
    print(json.dumps(result, indent=2))
