import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import torch.nn as nn
from meso import *

class Config(object):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = 'vgg16' 
        self.classes = ('fake', 'real')
        self.metric_path = f'./{self.model_name}_testmetric.pkl'

def data_loading(config):
    if config.model_name == 'vgg16':
        transform_train = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    elif config.model_name == 'meso':
        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
            ])
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
            ])

    training_dataset = datasets.ImageFolder('data1/train', transform=transform_train)
    testing_dataset = datasets.ImageFolder('data1/val', transform=transform)
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=20, shuffle=True)
    testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size = 20, shuffle=False)
    return training_loader, testing_loader

def building_vgg16(config):
    print(f"Model name: {config.model_name}.")
    model = models.vgg16(pretrained=True)
    # Fix pre-trained model's parameters
    # for param in model.features.parameters():
    #     param.requires_grad = False
    # Change the number of the output neurons from 1000 to len(classes) 2
    n_inputs = model.classifier[6].in_features
    last_layer = nn.Linear(n_inputs, len(config.classes))
    model.classifier[6] = last_layer

    model.to(config.device)
    print(f"Device: {config.device}.")
    print(model)
    return model

def building_meso(config):
    model = Meso4()
    model.to(config.device)
    print(f"Device: {config.device}.")
    print(model)
    return model

def training(config, model, criterion, optimizer, training_loader, testing_loader, epochs):
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []

    for ep in range(epochs):

        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0

        model.train()
        for inputs, labels in training_loader:
            inputs,labels = inputs.to(config.device),labels.to(config.device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item()
            train_acc += torch.sum(preds == labels.data)

        model.eval()
        for inputs, labels in testing_loader:
            inputs,labels = inputs.to(config.device),labels.to(config.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            test_loss += loss.item()
            test_acc += torch.sum(preds == labels.data)

        train_epoch_loss = train_loss/len(training_loader.dataset)
        train_epoch_acc = train_acc.float()/ len(training_loader.dataset)
        train_loss_history.append(train_epoch_loss)
        train_acc_history.append(train_epoch_acc)

        test_epoch_loss = test_loss/len(testing_loader.dataset)
        test_epoch_acc = test_acc.float()/ len(testing_loader.dataset)
        test_loss_history.append(test_epoch_loss)
        test_acc_history.append(test_epoch_acc)

        print(f'Epoch {ep+1}/{epochs}:')
        print(f'Train loss: {train_epoch_loss:.4f}, Train accuracy: {train_epoch_acc.item():.4f}.')
        print(f'Test loss: {test_epoch_loss:.4f}, Test accuracy: {test_epoch_acc.item():.4f}.')

    return model, train_loss_history,train_acc_history,test_loss_history,test_acc_history

def train_score(config, model, training_loader):
    model.eval()
    n_correct = 0.
    n_total = 0.
    p_train = np.array([])
    y_train_true = np.array([])
    for inputs, targets in training_loader:
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_train_true = np.concatenate((y_train_true, targets.cpu().numpy()))
        p_train = np.concatenate((p_train, preds.cpu().numpy()))

    train_acc = (1*(y_train_true == p_train)).sum() / y_train_true.shape[0]
    train_precision = precision_score(y_train_true, p_train)
    train_recall = recall_score(y_train_true, p_train)
    train_fscore = f1_score(y_train_true, p_train)
    
    metrics = {}
    metrics['accuracy'] = train_acc
    metrics['precision'] = train_precision
    metrics['recall'] = train_recall
    metrics['fscore'] = train_fscore
    metrics = pd.DataFrame(
        metrics,
        columns = ['accuracy', 'precision', 'recall', 'fscore'],
        index=[0]
    )
    # metrics.to_pickle(f'train_{config.metric_path}')
    print("Training Metrics: ")
    print(metrics)
    return metrics

def test_score(config, model, testing_loader):
    model.eval()
    n_correct = 0.
    n_total = 0.
    p_test = np.array([])
    y_test_true = np.array([])
    for inputs, targets in testing_loader:
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_test_true = np.concatenate((y_test_true, targets.cpu().numpy()))
        p_test = np.concatenate((p_test, preds.cpu().numpy()))

    test_acc = (1*(y_test_true == p_test)).sum() / y_test_true.shape[0]
    test_precision = precision_score(y_test_true, p_test)
    test_recall = recall_score(y_test_true, p_test)
    test_fscore = f1_score(y_test_true, p_test)
    
    metrics = {}
    metrics['accuracy'] = test_acc
    metrics['precision'] = test_precision
    metrics['recall'] = test_recall
    metrics['fscore'] = test_fscore
    metrics = pd.DataFrame(
        metrics,
        columns = ['accuracy', 'precision', 'recall', 'fscore'],
        index=[0]
    )
    # metrics.to_pickle(f'test_{config.metric_path}')
    print("Testing Metrics: ")
    print(metrics)
    return metrics
    

if __name__ == "__main__":
    config = Config()
    print("Start loading...")
    training_loader, testing_loader = data_loading(config)

    if config.model_name == "vgg16": 
        model = building_vgg16(config)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)  
        print("Start training...")
        bestmodel, train_loss_history,train_acc_history,test_loss_history,test_acc_history = training(config,
model, criterion, optimizer, training_loader, testing_loader, epochs=10) 
    elif config.model_name == "meso":
        model = building_meso(config)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08) 
        print("Start training...")
        bestmodel, train_loss_history,train_acc_history,test_loss_history,test_acc_history = training(config,
model, criterion, optimizer, training_loader, testing_loader, epochs=10) 

    train_score(config, bestmodel, training_loader)
    test_score(config, bestmodel, testing_loader)