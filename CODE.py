import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import random

class Perceptron(nn.Module):
    def __init__(self, input_dim, hidden_layers=None, output_dim=1):
        super(Perceptron, self).__init__()
        if hidden_layers is None:
            hidden_layers = []
        layers = []
        previous_dim = input_dim

        # 创建隐藏层
        for hidden_dim in hidden_layers:
            layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
            previous_dim = hidden_dim

        # 创建输出层
        layers.append(nn.Linear(previous_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


    def compute_loss_and_accuracy(outputs, targets, criterion):
        loss = criterion(outputs, targets)
        predicted = torch.round(torch.sigmoid(outputs))
        correct = (predicted == targets).sum().item()
        accuracy = correct / len(targets)
        return loss, accuracy


    def train_model(model, optimizer, tensors, num_epochs=1000, save_best_model=True, save_dir='saved_models'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        criterion = nn.BCEWithLogitsLoss()
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
        best_val_accuracy = 0

        for epoch in range(num_epochs):
            model.train()
            outputs_train = model(tensors['X_train'])
            loss_train, train_accuracy = Perceptron.compute_loss_and_accuracy(outputs_train, tensors['y_train'], criterion)
            train_losses.append(loss_train.item())
            train_accuracies.append(train_accuracy)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                outputs_val = model(tensors['X_val'])
                loss_val, val_accuracy = Perceptron.compute_loss_and_accuracy(outputs_val, tensors['y_val'], criterion)
                val_losses.append(loss_val.item())
                val_accuracies.append(val_accuracy)

            if save_best_model and val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

        if save_best_model:
            model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))

        model.eval()
        with torch.no_grad():
            outputs_test = model(tensors['X_test'])
            _, test_accuracy = Perceptron.compute_loss_and_accuracy(outputs_test, tensors['y_test'], criterion)

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'test_accuracy': test_accuracy
        }


    def plot_training_results(results_dict):
        for neurons, results in results_dict.items():
            plt.figure(figsize=(15, 6))
            
            # 获取对应的训练和验证的误差、准确率
            train_losses = results['train_losses']
            val_losses = results['val_losses']
            train_accuracies = results['train_accuracies']
            val_accuracies = results['val_accuracies']
            
            # 绘制训练和验证的误差
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss', linestyle='--')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.title(f'Training and Validation Loss for {neurons} neurons')
            
            # 绘制训练和验证的准确率
            plt.subplot(1, 2, 2)
            plt.plot(train_accuracies, label='Train Accuracy')
            plt.plot(val_accuracies, label='Validation Accuracy', linestyle='--')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title(f'Training and Validation Accuracy for {neurons} neurons')

            plt.tight_layout()
            plt.show()

        
    def check_overfitting(results_dict,MLP = True):
        if MLP:
            for neurons, results in results_dict.items():
                # 最后一个epoch的训练误差和验证误差
                final_train_loss = results['train_losses'][-1]
                final_val_loss = results['val_losses'][-1]
                loss_difference = final_val_loss - final_train_loss
                
                # 最后一个epoch的训练准确率和验证准确率
                final_train_accuracy = results['train_accuracies'][-1]
                final_val_accuracy = results['val_accuracies'][-1]
                accuracy_difference = final_train_accuracy - final_val_accuracy
                
                print(f"For {neurons} neurons:")
                print(f"Loss difference (Validation - Training): {loss_difference:.4f}")
                print(f"Accuracy difference (Training - Validation): {accuracy_difference:.4f}\n")
        else:
                final_train_loss = results['train_losses'][-1]
                final_val_loss = results['val_losses'][-1]
                loss_difference = final_val_loss - final_train_loss

                final_train_accuracy = results['train_accuracies'][-1]
                final_val_accuracy = results['val_accuracies'][-1]
                accuracy_difference = final_train_accuracy - final_val_accuracy
                print(f"Loss difference (Validation - Training): {loss_difference:.4f}")
                print(f"Accuracy difference (Training - Validation): {accuracy_difference:.4f}\n")


    def set_seed(seed=6):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        
    def data_to_tensors(X_train, y_train, X_val, y_val, X_test, y_test):
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.where(torch.FloatTensor(y_train.values) <= 0, 0, 1).float().unsqueeze(1)
        
        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.where(torch.FloatTensor(y_val.values) <= 0, 0, 1).float().unsqueeze(1)
        
        X_test_tensor = torch.FloatTensor(X_test.values)
        y_test_tensor = torch.where(torch.FloatTensor(y_test.values) <= 0, 0, 1).float().unsqueeze(1)

        tensors = {
            'X_train': X_train_tensor,
            'y_train': y_train_tensor,
            'X_val': X_val_tensor,
            'y_val': y_val_tensor,
            'X_test': X_test_tensor,
            'y_test': y_test_tensor
        }
        
        return tensors