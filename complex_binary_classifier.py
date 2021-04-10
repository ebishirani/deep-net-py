import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import torch.nn as nn
#**************************************************************
class ComplexClassifier(nn.Module):
    def __init__(self, inputSize, h1, outputSize):
        super().__init__()
        self.mModel = nn.Sequential(
            nn.Linear(inputSize, h1),
            nn.ReLU(),
            nn.Linear(h1, outputSize)
            )

    def forward(self, x):
        result = torch.sigmoid(self.mModel(x))
        return result

    def predict(self, sample):
        pred = self.forward(sample)
        if pred >= 0.5:
            return 1
        else:
            return 0
#**************************************************************
#Declare a trainer function for our model
#@ param trainData: Is a list that contain train data. faitst element contains
#  samples and second one contains true lables. [x, y]
def trainer(trainData, numOfEpochs : np.int32 = 1000) -> ComplexClassifier:
    #Determine a seed to produce better init prams for model
    torch.manual_seed(2)
    #declare a new model
    model = ComplexClassifier(2, 4, 1)
    #For binary classification, the best choice for loss function is BCELoss
    #that stands for binary class entropy loss
    lossFunc = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr = 0.1) 
    #Declare a list to hold loss values in trainin operation
    lossValues = []
    for i in range(numOfEpochs):
        pred = model(trainData[0])
        #calculate loss value
        loss = lossFunc(pred, trainData[1])
        lossValues.append(loss.item())
        #reset previous gradients
        optim.zero_grad()
        #calculate this epoch graidients
        loss.backward()
        optim.step()
    plt.subplot(122)
    plt.plot(range(numOfEpochs), lossValues)
    plt.ylabel('loss')
    plt.xlabel('epochs')

    return model

#**************************************************************
def main():
    # try to genarate a dataset with 2 circular clusters.Both f them has 0.4 as
    # it's standard deviation.
    numOfSamplesPerCluster = 500
    seed = 123
    x, y = datasets.make_circles(
        n_samples = numOfSamplesPerCluster,
        noise = 0.1,
        factor = 0.2,
        random_state = seed)
    #convert data type of x array from floar64 to float32
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    #convert x and y to tensors
    samples = torch.from_numpy(x)
    labels = torch.from_numpy(y.reshape(500, 1))
    #declare and train complex classifiet
    complexClassifier = trainer([samples, labels], 250)  
    # try to test trained model
    point1 = torch.tensor([1.0, -1.0])
    point2 = torch.tensor([0.25, 0.35])
    plt.subplot(121)
    plt.plot(point1.numpy()[0], point1.numpy()[1], 'ro')
    plt.plot(point2.numpy()[0], point2.numpy()[1], 'ko')

    #plot genarated samples
    plt.subplot(121)
    plt.scatter(x[y == 0, 0], x[y == 0, 1])
    plt.scatter(x[y == 1, 0], x[y == 1, 1])
    plt.show()

    print(complexClassifier.predict(point1))
    print(complexClassifier.predict(point2))



main() 