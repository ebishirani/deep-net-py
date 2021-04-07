import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class LinearReg(nn.Module):
    def __init__(self, inSize, outSize):
        super().__init__()
        self.linear = nn.Linear(inSize, outSize)

    def forward(self, x):
        pred = self.linear(x)
        return pred
#End of class LinearReg

#This function will be used to train outr model
# @param data : data is a list that contains train data:[x, y]
# @param numOfEpochs is a integer value that specifies the number of epochs
def trainer(data, numOfEpochs) -> LinearReg:
    #create an instance of model
    model = LinearReg(1, 1)
    #Create a loss function
    lossFunc = torch.nn.MSELoss()
    #create an optimizer
    optim = torch.optim.SGD(params = model.parameters(), lr = 0.001)

    #try to train created model
    #Declare a list to hold the losses in train operation
    lossValues = []
    for i in range(numOfEpochs):
        pred = model.forward(data[0])
        
        loss = lossFunc(pred, data[1])
        lossValues.append(loss)
        #zerose all gradients from previuos iteration
        optim.zero_grad()
        #Claculate gradeints
        loss.backward()
        #update parameters
        optim.step()
    
    #Plot the loss values against the epochs
    plt.subplot(131)
    plt.plot(range(numOfEpochs), lossValues)
    plt.ylabel('loss')
    plt.xlabel('epoch')

    return model
#End of trainer function
def main():
    #genarate data to train a linear regression model
    x = torch.randn(100, 1) * 10
    y = x + torch.randn(100, 1) * 3
    plt.subplot(132)
    plt.plot(x.numpy(), y.numpy(), 'o')
    plt.ylabel(ylabel = 'Y')
    plt.xlabel(xlabel = 'X')
    # plt.show()

    #Encapsulate generated data
    data = []
    data.append(x)
    data.append(y)
    #build and train a model
    model = trainer(data, 100)

    [w, b] = model.parameters()
    w1 = w[0][0].item()
    b1 = b[0].item()

    x1 = np.array([-30, 30])
    y1 = w1 * x1 + b1

    plt.subplot(133)
    plt.plot(x1, y1, 'r')
    plt.scatter(x, y)
    plt.show()

main()