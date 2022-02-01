'''
Created on Jan 25, 2022
@author: Xingchen Li
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter


# Weight initialization, default Xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train() # Set the model to the training state
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # The learning rate decays exponentially each epoch: learning rate = gamma * learning rate
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # Record how many batches are carried out
    dev_best_loss = float('inf')
    dev_best_f1 = 0
    last_improve = 0  # Record the batch number of loss decreases in the last verification set
    flag = False  # Record whether the effect has not been improved for a long time
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for _, (trains, labels) in enumerate(train_iter):
            # trains(batch_size, setence, seq_len)
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step() # Attenuation of learning rate
            if total_batch % 100 == 0:
                # Output the effect on the training set and verification set per number of rounds
                true = labels.data.cpu()
                # torch.max() Returns the maximum value of each row on a given dimension of the input tensor, along with the positional index of each maximum value.
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                train_precision, train_recall, train_f1, _ = metrics.precision_recall_fscore_support(true, predic, average="micro")
                dev_acc, dev_precision, dev_recall, dev_f1, dev_loss = evaluate(config, model, dev_iter)
                # Verify losses and save the model while verifying that F1 has improved
                if dev_loss < dev_best_loss or dev_f1 > dev_best_f1:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("train/loss", loss.item(), total_batch)
                writer.add_scalar("train/acc", train_acc, total_batch)
                writer.add_scalar("train/pricision", train_precision, total_batch)
                writer.add_scalar("train/recall", train_recall, total_batch)
                writer.add_scalar("train/f1", train_f1, total_batch)
                writer.add_scalar("train/lr", scheduler.get_last_lr(), total_batch)
                writer.add_scalar("dev/loss", dev_loss, total_batch)
                writer.add_scalar("dev/acc", dev_acc, total_batch)
                writer.add_scalar("dev/pricision", dev_precision, total_batch)
                writer.add_scalar("dev/recall", dev_recall, total_batch)
                writer.add_scalar("dev/f1", dev_f1, total_batch)

                model.train() # Set the model to training mode after verification
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # The loss of the verification set exceeds 1000 and the training is complete
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    dev_precision, dev_recall, dev_f1, _ = metrics.precision_recall_fscore_support(labels_all, predict_all, average="micro")
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, dev_precision, dev_recall, dev_f1, loss_total / len(data_iter)
