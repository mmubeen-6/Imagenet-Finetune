import time
import yaml

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
    with open(config_file_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    return config

def calculate_accuracy(output, target, topk=(1,3)):
    """
    Computes the correctly predicted classes @k for the specified values of k
    :param output: output predictions from model
    :param target: target predictions
    :param topk: top-k accuracies to find
    :return: top-k accuracies of model
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return res