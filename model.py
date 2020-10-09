import torch
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

def get_model(num_classes):
    """
    Loads and returns a resnet model
    :param num_classes: num of classes in final layer
    :return model: model loaded in memory
    """
    model = resnet18(pretrained=True)

    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model