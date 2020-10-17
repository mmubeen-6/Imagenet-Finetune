import os 
import yaml
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from dataset import ImageDataset
from model import get_model
from utils import load_config_file, calculate_accuracy

def load_dataset(config):
    """
    Loads and returns the training and validation dataset
    :param config: configs for training/testing
    :return train_loader: dataloader for training data
    :return test_loader: dataloader for testing data
    :return num_of_classes: total classes in training set
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

    test_dataset = ImageDataset(root_dir=config["TEST_DATA_PATH"], model_save_path=config["MODEL_SAVE_PATH"],  testing=True, transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["TEST_BATCH_SIZE"], shuffle=False, num_workers=config["TEST_WORKERS"], pin_memory=True)
    
    return test_loader, test_dataset.get_total_classes()

def test_model(model, test_loader, num_classes):
    """
    Test model on validation set
    :param model: model to test
    :param test_loader: dataloader for test dataset
    :param epoch_num: current epoch
    :param criterion: loss function used for training
    :param writer: tensorboard summary writer object to write validation loss/accuracy
    :return accuracy: current accuracy of the model on validation dataset
    :return test_loss: current loss of the model on validation dataset
    """
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    top_1_correct, top_3_correct = 0, 0
    accuracy_per_class = np.zeros((num_classes,), dtype=np.int64)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    start_time = time.time()
    with torch.set_grad_enabled(False):
        for idx, sample in enumerate(test_loader):
            data, targets = sample
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            _, predictions = torch.max(outputs, 1)

            # Confusion Matrix along with saving wrongly classified images
            for t, p in zip(targets.view(-1), predictions.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            
            accuracies = calculate_accuracy(outputs, targets, topk=(1, 3))
            top_1_correct += accuracies[0]
            top_3_correct += accuracies[1]

            print("Testing Progress |{}{}|  {:3.1f} % complete  \r".format(
                int(60*(idx/len(test_loader)))*"#", int(60*(1-idx/len(test_loader)))*"-", 100*idx/len(test_loader)), end="")

    accuracy_top_1 = 100. * float(top_1_correct) / len(test_loader.dataset)
    accuracy_top_3 = 100. * float(top_3_correct) / len(test_loader.dataset)
    accuracy_per_class = (confusion_matrix.diag() / confusion_matrix.sum(1)).cpu().detach().numpy()
    accuracy_per_class = np.nan_to_num(accuracy_per_class) * 100.
    confusion_matrix = confusion_matrix.cpu().detach().numpy()

    end_time = time.time()
    print("Testing took {:.2f} secs\033[K".format(end_time - start_time))
    print("Testing Summary: Accuracy-1: {:.2f}% Accuracy-3: {:.2f}% \033[K".format(accuracy_top_1, accuracy_top_3))
    
    return accuracy_top_1, accuracy_top_3, accuracy_per_class, confusion_matrix

def save_results(config, accuracy_per_class, confusion_matrix):
    """
    Save confusion matrix and per class accuracy on disk
    :param config: config file
    :param accuracy_per_class: per class accuracy to write
    :param confusion_matrix: confusion matrix to write
    :return None:
    """
    # saving confusion matrix 
    np.savetxt(os.path.join(config["MODEL_OUTPUT_RESULTS_PATH"], "confusion_matrix.csv"), confusion_matrix.astype(np.int32), delimiter=",", fmt="%d")

    # saving accuracy list
    with open(os.path.join(config["MODEL_SAVE_PATH"], "class_labels.txt")) as f_:
        content = f_.readlines()
        class_labels = [x.strip() for x in content]

    with open(os.path.join(config["MODEL_OUTPUT_RESULTS_PATH"], "accuracy.csv"), "w") as f_:
        f_.write("class_number,accuracy\n")
        for i in range(accuracy_per_class.shape[0]):
            f_.write("{},{:.2f}\n".format(class_labels[i], accuracy_per_class[i]))
    
    return

if __name__ == "__main__":
    print("Loading Config")
    config_file = "./default-config,yaml"
    config = load_config_file(config_file)
    
    # dataloaders
    print("Getting Dataloader")
    test_loader, total_classes = load_dataset(config)

    # creating model
    print("Loading Model")
    model = get_model(num_classes=total_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # loading model
    if not os.path.isfile(config["MODEL_TEST_PATH"]):
        print("Model checkpoint weights does no exist, exiting....")
        import sys; sys.exit()
    else:
        ckpt = torch.load(config["MODEL_TEST_PATH"])
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "accuracy" in ckpt:
                print("The loaded model has Validation accuracy of: {:.2f} %\n".format(ckpt["accuracy"]))
        else:
            model.load_state_dict(ckpt)

    print("Testing the model")
    print("-" * 50)
    accuracy_top_1, accuracy_top_3, accuracy_per_class, confusion_matrix = test_model(model, test_loader, total_classes)
    print("-" * 50)

    save_results(config, accuracy_per_class, confusion_matrix)