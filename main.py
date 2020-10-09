import os 
import yaml
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter

from dataset import ImageDataset
from model import get_model

torch.backends.cudnn.benchmark = True

def calc_time_left(batch_index, total_batches, time_consumed):
    """
    Calculates remaining epoch time based on already epoch consumed time
    :param batch_index: total train batches forwarded
    :param total_batches: total train batches in training
    :param time_consumed: total time consumed yet in epoch
    :return: time remaining in training.
    """
    average_time_consumed = time_consumed / batch_index
    time_left = average_time_consumed * (total_batches - batch_index)

    return "{}m {}s".format(int(time_left // 60), int(time_left % 60))

def load_config_file(config_file_path):
    """
    Loads the config file for input parameter loading
    :param config_file_path: path of config yaml file
    :return: configs for training/testing
    """
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    return config

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
    data_transforms = {
        'train':
        transforms.Compose([
            transforms.Resize(256),  # smaller side resized
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            normalize
        ]),
        'validation':
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ]),
    }

    train_dataset = ImageDataset(root_dir=config["TRAIN_DATA_PATH"], model_save_path=config["MODEL_SAVE_PATH"], testing=False, transform=data_transforms["train"])
    test_dataset = ImageDataset(root_dir=config["VAL_DATA_PATH"], model_save_path=config["MODEL_SAVE_PATH"],  testing=True, transform=data_transforms["validation"])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["TRAIN_BATCH_SIZE"], shuffle=True, num_workers=config["TRAIN_WORKERS"], pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["TEST_BATCH_SIZE"], shuffle=False, num_workers=config["TEST_WORKERS"], pin_memory=True)
    return train_loader, test_loader, train_dataset.get_total_classes()


def validate(model, test_loader, epoch_num, criterion, writer):
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
    test_loss = 0.
    correct = 0

    with torch.set_grad_enabled(False):
        for idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            writer.add_scalar('Loss/Val', loss, (idx + ((epoch_num-1) * len(test_loader))))
            print("Validation Progress |{}{}|  {:3.1f} % complete  \r".format(
                int(60*(idx/len(test_loader)))*"#", int(60*(1-idx/len(test_loader)))*"-", 100*idx/len(test_loader)), end='')

        test_loss /= len(test_loader)
        accuracy = 100 * float(correct) / len(test_loader.dataset)

    print("Val Summary: Loss: {:.2f}, Accuracy: {:.2f}% \033[K".format(test_loss, accuracy))
    
    # Logging paramters into tensorboard
    writer.add_scalar('Accuracy/Validation', accuracy, epoch_num)
    return accuracy, test_loss

def train(config):
    """
    Train model on train dataset
    :param config: loaded config from yaml file
    :return: None
    """
    # tensorboard initialization
    writer = SummaryWriter("{}/tensorboard".format(config["MODEL_SAVE_PATH"]), flush_secs=1)

    # dataloaders
    train_loader, test_loader, total_classes = load_dataset(config)

    # getting model
    model = get_model(num_classes=total_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimzers
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])
    step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config["STEP_SIZE"], gamma=0.1)

    # Multi-GPU
    multi_gpu = True if (config["MULTI_GPU"] and torch.cuda.device_count() > 1) else False
    if multi_gpu:
        print("Using {} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    else:
        print("Using 1 GPU!")

    test_loss = 0
    latest_accuracy = 0
    best_accuracy = 0
    #### Training ####
    for epoch in range(1, config["TRAIN_EPOCHS"] + 1):
        epoch_start_time = time.time()
        print('Epoch {}/{}'.format(epoch, config["TRAIN_EPOCHS"]))
        print('-' * 50)

        model.train()
        with torch.set_grad_enabled(True):
            for batch_idx, sample in enumerate(train_loader):
                data, target = sample
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                # Logging paramters into tensorboard
                writer.add_scalar('Loss/training', loss, (batch_idx + ((epoch-1) * len(train_loader))))
                print("Train Progress |{}{}|   {:3.1f} % complete [Loss: {:4.2f}] [ValLoss: {:4.2f}] [PrevValAccuracy: {:4.2f}] [BestValAccuracy: {:4.2f}] [Time Left: {}]\033[K\r".format(
                    int(50*(batch_idx/len(train_loader)))*"#", int(50*(1-batch_idx/len(train_loader)))*"-", 100*batch_idx/len(train_loader), loss, float(test_loss), float(latest_accuracy), float(best_accuracy), calc_time_left(batch_idx + 1, len(train_loader), (time.time() - epoch_start_time))), end='')

        elapsed_time = (time.time() - epoch_start_time) / 60
        print("Train Summary: [Loss: {:4.2f}] [ValLoss: {:4.2f}] [BestValAccuracy: {:4.2f}%]  [Epoch Time:{:.2f} min]\033[K".format(
            loss, float(test_loss), float(best_accuracy), elapsed_time))
        
        latest_accuracy, test_loss = validate(model, test_loader, epoch, criterion, writer)

        if latest_accuracy >= best_accuracy:
            print("Saving Model")
            best_accuracy = latest_accuracy
            torch.save({'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if multi_gpu else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'schedular_state_dict': step_lr_scheduler.state_dict(),
                        'accuracy': latest_accuracy,
                        'loss': test_loss
                        }, os.path.join(config["MODEL_SAVE_PATH"], str(epoch) + '.pth'))
        
        torch.save({'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if multi_gpu else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'schedular_state_dict': step_lr_scheduler.state_dict(),
                    'accuracy': latest_accuracy,
                    'loss': test_loss
                    }, os.path.join(config["MODEL_SAVE_PATH"], 'latest.pth'))
                    
        
        step_lr_scheduler.step()
        print("\n")

    writer.close()
    torch.save(model.module.state_dict() if multi_gpu else model.state_dict(), os.path.join(config["MODEL_SAVE_PATH"], 'final.weights'))
    print('Trained model path: ', config["MODEL_SAVE_PATH"])

if __name__ == "__main__":
    config_file = "./default-config,yaml"
    config = load_config_file(config_file)
    
    train(config)