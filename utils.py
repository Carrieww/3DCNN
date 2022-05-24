import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor


def saveModel(model, optimizer, MAETrain, MAETest, RMSE, MAPE, lr, dropout, filterSize, path):
    state = {'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'MAETrain': MAETrain,
             'MAETest': MAETest,
             'RMSE': RMSE,
             'MAPE': MAPE,
             'lr': lr,
             'dropout': dropout,
             'filterSize': filterSize
             }
    torch.save(state, path)


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))


def MAPELoss(yhat, y):
    return torch.mean(((y - yhat) / y).abs()) * 100


def readFile(path, model, opt):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    # MAETrain = checkpoint['MAETrain']
    # MAETest = checkpoint['MAETest']
    # RMSE = checkpoint['RMSE']
    # MAPE = checkpoint['MAPE']
    # lr = checkpoint['lr']
    model.state_dict()
    return model, opt

# used when raw data are used
def LoadData(PredictedY):
    # Create dataset from several tensors with matching first dimension
    # Samples will be drawn from the first dimension (rows)

    X_train_daily_input = torch.load('data/M1_X_train_daily_input.pt')
    X_train_monthly_input = torch.load('data/M1_X_train_monthly_input.pt')
    X_train_yearly_input = torch.load('data/M1_X_train_yearly_input.pt')
    X_test_daily_input = torch.load('data/M1_X_test_daily_input.pt')
    X_test_monthly_input = torch.load('data/M1_X_test_monthly_input.pt')
    X_test_yearly_input = torch.load('data/M1_X_test_yearly_input.pt')
    y_train_input = torch.load('data/M1_Y_train_'+PredictedY+'.pt')
    y_test_input = torch.load('data/M1_Y_test_'+PredictedY+'.pt')

    train_dataset = TensorDataset(Tensor(X_train_daily_input),
                                  Tensor(X_train_monthly_input),
                                  Tensor(X_train_yearly_input),
                                  Tensor(y_train_input))
    test_dataset = TensorDataset(Tensor(X_test_daily_input),
                                 Tensor(X_test_monthly_input),
                                 Tensor(X_test_yearly_input),
                                 Tensor(y_test_input))

    # Create a data loader from the dataset
    # Type of sampling and batch size are specified at this step
    train_loader = DataLoader(train_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Quick test
    # next(iter(loader))
    return train_loader, test_loader
