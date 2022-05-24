import warnings
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from utils import *

warnings.filterwarnings("ignore")


class NeuralNet(nn.Module):
    def __init__(self, dropout=0.1, filterSize=16, node=128, Bias=False):
        super(NeuralNet, self).__init__()
        self.som = nn.Softmax()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=filterSize,
                                             kernel_size=(1, 5, 8), stride=(1, 5, 8), padding=0),
                                   nn.MaxPool3d(kernel_size=(1, 7, 1),
                                                stride=(1, 7, 1), padding=0),
                                   nn.ReLU())
        self.conv2_1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=filterSize,
                                               kernel_size=(3, 5, 3), stride=(3, 5, 3), padding=0),
                                     nn.MaxPool3d(kernel_size=(4, 7, 1),
                                                  stride=(4, 7, 1), padding=0),
                                     nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=filterSize,
                                               kernel_size=(6, 5, 3), stride=(6, 5, 3), padding=0),
                                     nn.MaxPool3d(kernel_size=(2, 7, 1),
                                                  stride=(2, 7, 1), padding=0),
                                     nn.ReLU())
        self.conv2_3 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=filterSize,
                                               kernel_size=(12, 5, 3), stride=(12, 5, 3), padding=0),
                                     nn.MaxPool3d(kernel_size=(1, 7, 1),
                                                  stride=(1, 7, 1), padding=0),
                                     nn.ReLU())
        self.conv3_1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=filterSize,
                                               kernel_size=(10, 5, 8), stride=(10, 5, 8), padding=0),
                                     nn.MaxPool3d(kernel_size=(25, 1, 1),
                                                  stride=(25, 1, 1), padding=0),
                                     nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=filterSize,
                                               kernel_size=(25, 5, 8), stride=(25, 5, 8), padding=0),
                                     nn.MaxPool3d(kernel_size=(10, 1, 1),
                                                  stride=(10, 1, 1), padding=0),
                                     nn.ReLU())
        self.conv3_3 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=filterSize,
                                               kernel_size=(50, 5, 8), stride=(50, 5, 8), padding=0),
                                     nn.MaxPool3d(kernel_size=(5, 1, 1),
                                                  stride=(5, 1, 1), padding=0),
                                     nn.ReLU())
        self.conv3_4 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=filterSize,
                                               kernel_size=(125, 5, 8), stride=(125, 5, 8), padding=0),
                                     nn.MaxPool3d(kernel_size=(2, 1, 1),
                                                  stride=(2, 1, 1), padding=0),
                                     nn.ReLU())
        self.conv3_5 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=filterSize,
                                               kernel_size=(250, 5, 8), stride=(250, 5, 8), padding=0),
                                     nn.MaxPool3d(kernel_size=(1, 1, 1),
                                                  stride=(1, 1, 1), padding=0),
                                     nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(in_features=filterSize * 58, out_features=node, bias=Bias),
                                nn.Sigmoid(),
                                nn.Linear(in_features=node, out_features=node, bias=Bias),
                                nn.Sigmoid(),
                                nn.Linear(in_features=node, out_features=1, bias=Bias))
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x1, x2, x3):
        x1 = self.conv1(x1)
        x2_1 = self.conv2_1(x2)
        x2_2 = self.conv2_2(x2)
        x2_3 = self.conv2_3(x2)
        x3_1 = self.conv3_1(x3)
        x3_2 = self.conv3_2(x3)
        x3_3 = self.conv3_3(x3)
        x3_4 = self.conv3_4(x3)
        x3_5 = self.conv3_5(x3)
        x1 = x1.view(x1.numel())
        x2_1 = x2_1.view(x2_1.numel())
        x2_2 = x2_2.view(x2_2.numel())
        x2_3 = x2_3.view(x2_3.numel())
        x3_1 = x3_1.view(x3_1.numel())
        x3_2 = x3_2.view(x3_2.numel())
        x3_3 = x3_3.view(x3_3.numel())
        x3_4 = x3_4.view(x3_4.numel())
        x3_5 = x3_5.view(x3_5.numel())
        x = torch.cat([x1, x2_1, x2_2, x2_3, x3_1, x3_2,x3_3, x3_4, x3_5])

        x = self.dropout1(x)
        x = self.som(x)
        x = self.fc(x)
        return x


def training_function(n, model, best_Test_MAE, opt, learning_rate, dropout, filterNum, trainLoader, testLoader):
    num_epochs = n
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.L1Loss()
    MAETest_list = []
    MAETrain_list = []

    for epoch in range(0, num_epochs):
        print('epochs {}/{}'.format(epoch + 1, num_epochs))

        iteration_list = []
        train_pred_list = []
        train_labels_list = []

        test_iteration_list = []
        test_pred_list = []
        test_labels_list = []

        # set the model in training mode
        model.train()

        train_running_loss = 0.0
        test_running_loss = 0.0
        count = 0
        # loop over the training set
        for i, (daily_train, monthly_train, yearly_train, train_labels) in enumerate(trainLoader):
            daily_train = daily_train.to(device)
            monthly_train = monthly_train.to(device)
            yearly_train = yearly_train.to(device)
            train_labels = train_labels.to(device)
            # perform a forward pass and calculate the training loss
            pred = model(yearly_train, monthly_train, daily_train)
            loss = criterion(pred, train_labels)
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_running_loss += loss.item()

            count += 1
            if i % 1 == 0:
                train_pred_list.append(pred)
                iteration_list.append(i)
                train_labels_list.append(train_labels)

        test_count = 0
        model.eval()
        with torch.no_grad():

            for (daily_test, monthly_test, yearly_test, test_labels) in testLoader:
                daily_test = daily_test.to(device)
                monthly_test = monthly_test.to(device)
                yearly_test = yearly_test.to(device)
                test_labels = test_labels.to(device)
                test_predicted = model(yearly_test, monthly_test, daily_test)
                #                 print(test_predicted)

                test_loss = criterion(test_predicted, test_labels)
                test_running_loss += test_loss.item()
                test_count += 1

                test_pred_list.append(test_predicted)
                test_iteration_list.append(test_count)
                test_labels_list.append(test_labels)

        MAETest = test_running_loss / len(testLoader)
        MAETest_list.append(MAETest)
        MAETrain = train_running_loss / len(trainLoader)
        MAETrain_list.append(MAETrain)
        RMSE = RMSELoss(torch.stack(test_pred_list).reshape([len(testLoader)]),
                        torch.stack(test_labels_list).reshape([len(testLoader)]))
        MAPE = MAPELoss(torch.stack(test_pred_list).reshape([len(testLoader)]),
                        torch.stack(test_labels_list).reshape([len(testLoader)]))

        # save model
        # if epoch == 0 or best_Test_MAE > MAETest:
        #     best_Test_MAE = MAETest
        #     path = "checkpoint/" + PredictedY + ".pth"
        #     saveModel(model, opt, MAETrain, MAETest, RMSE, MAPE, learning_rate, dropout, filterNum, path)

        print(f'MAE for training set is {MAETrain}')
        print(f'RMSE Loss for validation set is {RMSE}')
        print(f'MAE for validation set is {MAETest}')
        print(f'MAPE Loss for validation set is {MAPE}')

        # print to visualize
        # plt.plot(iteration_list, train_pred_list, 'r')
        # plt.plot(iteration_list, train_labels_list, 'b')
        # plt.title('Predicted and target prices of ' + PredictedY + ' - training')
        # plt.show()
        # plt.plot(test_iteration_list, test_pred_list, 'r')
        # plt.plot(test_iteration_list, test_labels_list, 'b')
        # plt.title('Predicted and target prices of ' + PredictedY + ' - testing')
        # plt.show()
    print("finished training")
    return MAETest, MAETrain, RMSE, MAPE, model, opt, MAETrain_list, MAETest_list, test_pred_list, test_labels_list


def TrainModel(dropout, filterNum, node, learning_rate, epoch, trainLoader, testLoader):
    # initiate model and opt
    model = NeuralNet(dropout, filterNum, node)
    opt = torch.optim.Adam(params=model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0, amsgrad=True)
    Test_MAE, Train_MAE, RMSE, MAPE, model, opt, MAETrain_list, MAETest_list, _, _ = training_function(epoch, model,
                                                                                                       10000, opt,
                                                                                                       learning_rate,
                                                                                                       dropout,
                                                                                                       filterNum,
                                                                                                       trainLoader,
                                                                                                       testLoader)
    # save model
    # path = "checkpoint/" + PredictedY + ".pth"
    # saveModel(model, opt, Train_MAE, Test_MAE, RMSE, MAPE, learning_rate, dropout, filterNum, path)
    # with open('ListResults/' + PredictedY + '.txt', 'w') as f:
    #     for i in MAETrain_list:
    #         f.write('%s\n' % i)
    #     for k in MAETest_list:
    #         f.write('%s\n' % k)
    return model


def TestModel(model, opt, dataloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.L1Loss()
    test_pred_list = []
    test_iteration_list = []
    test_labels_list = []

    model.eval()
    with torch.no_grad():
        test_running_loss = 0.0
        test_count = 0
        for (daily_test, monthly_test, yearly_test, test_labels) in dataloader:
            daily_test = daily_test.to(device)
            monthly_test = monthly_test.to(device)
            yearly_test = yearly_test.to(device)
            test_labels = test_labels.to(device)
            test_predicted = model(yearly_test, monthly_test, daily_test)

            loss = criterion(test_predicted, test_labels)
            test_running_loss += loss.item()
            test_count += 1
            test_pred_list.append(test_predicted)
            test_iteration_list.append(test_count)
            test_labels_list.append(test_labels)

    MAETest = test_running_loss / len(dataloader)

    RMSE = RMSELoss(torch.stack(test_pred_list).reshape([len(dataloader)]),
                    torch.stack(test_labels_list).reshape([len(dataloader)]))
    MAPE = MAPELoss(torch.stack(test_pred_list).reshape([len(dataloader)]),
                    torch.stack(test_labels_list).reshape([len(dataloader)]))
    print(f'RMSE Loss for testing set is {RMSE}')
    print(f'test_loss {MAETest}')
    print(f'MAPE Loss for testing set is {MAPE}')

    # visualize testing predictions
    plt.plot(test_iteration_list, test_pred_list, 'r')
    plt.plot(test_iteration_list, test_labels_list, 'b')
    plt.title('Predicted and target prices of ' + PredictedY + ' - testing')
    plt.show()


def main(pretrainedFlag,PredictedY):
    # load preprocessed data
    train_dataset = torch.load('data/train_loader_' + PredictedY + '.pth')
    test_dataset = torch.load('data/test_loader_' + PredictedY + '.pth')

    if not pretrainedFlag:
        # train model
        # define trainning parameters
        dropout = 0.3
        filterNum = 16
        node = 128
        learning_rate = 0.001
        epoch = 2
        model = TrainModel(dropout, filterNum, node, learning_rate, epoch, train_dataset, test_dataset)
        # visualization skipped

    else:
        # test pretrained model
        if PredictedY == "Oats":
            # initiate model first
            model = NeuralNet(0.3, 16, 128)
            opt = torch.optim.Adam(params=model.parameters(), lr=0.001)
            path = "pretrainedModel/" + PredictedY + "_final.pth"
        elif PredictedY == "US Soybeans":
            model = NeuralNet(0.1, 10, 128)
            opt = torch.optim.Adam(params=model.parameters(), lr=0.001)
            path = "pretrainedModel/" + PredictedY + "_final.pth"
        elif PredictedY == "US Corn":
            model = NeuralNet(0.3, 32, 128, True)
            opt = torch.optim.Adam(params=model.parameters(), lr=0.001)
            path = "pretrainedModel/" + PredictedY + "_final.pth"
        elif PredictedY == "Rough Rice":
            model = NeuralNet(0.1, 16, 128)
            opt = torch.optim.Adam(params=model.parameters(), lr=0.001)
            path = "pretrainedModel/" + PredictedY + "_final.pth"
        elif PredictedY == "US Wheat":
            model = NeuralNet(0.1, 20, 128, True)
            opt = torch.optim.Adam(params=model.parameters(), lr=0.001)
            path = "pretrainedModel/" + PredictedY + "_final.pth"
        else:
            print("Please input a valid PredictedY.")
        model, opt = readFile(path, model, opt)

    # test model
    TestModel(model, opt, test_dataset)


# set global variable: choose from ['Oats','Rough Rice','US Corn','US Soybeans', 'US Wheat']
PredictedY = "Oats"
pretrainedFlag = True
main(pretrainedFlag,PredictedY)
