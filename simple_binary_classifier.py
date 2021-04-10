import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import torch.nn as nn
#Declare a class to generate proper data for algorithm
# class DataGenarator:
#     def __init__(
#         self, 
#         seed : np.int32 = 123, 
#         dataAmount : np.int32 = 100,
#         numOfClusters : np.int32 = 2
#         ):
#         self.mSeed = seed
#         self.mDataAmount = dataAmount
#         self.mNumOfClusters = numOfClusters
#         self.
#**************************************************************
class SimpleClassifier(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.mNet = nn.Linear(inputSize, outputSize)

    def forward(self, x):
        result = torch.sigmoid(self.mNet(x))
        return result

    def predict(self, sample):
        pred = self.forward(sample)
        if pred >= 0.5:
            return 1
        else:
            return 0

        
 #**************************************************************
def getModelParams(model):
    [w, b] = model.parameters()
    w1 = w[0][0].item()
    w2 = w[0][1].item()
    b1 = b[0].item()
    return w1, w2, b1
#**************************************************************
#w1, w2 and b1 are model parameters
#xStart: First point of the active range
#xEnd: Last point of the active range
# In this function we try to plot this equation: 0 = w1x1 + w2x2 + b1
def plotFittedModel(label, w1, w2, b1, xStart, xEnd, color):
    x1 = np.array([xStart, xEnd])
    x2 = (w1 * x1 + b1) / (-w2)
    plt.subplot(121)
    plt.label = label
    plt.plot(x1, x2, color)
#**************************************************************
#Declare a trainer function for our model
#@ param trainData: Is a list that contain train data. faitst element contains
#  samples and second one contains true lables. [x, y]
def trainer(trainData, numOfEpochs : np.int32 = 1000) -> SimpleClassifier:
    #Determine a seed to produce better init prams for model
    torch.manual_seed(2)
    #declare a new model
    model = SimpleClassifier(2, 1)
    #get model parameters
    w1, w2, b1 = getModelParams(model)
    plotFittedModel('Initial model', w1, w2, b1, -2.0, 2.0, 'r')
    #For binary classification, the best choice for loss function is BCELoss
    #that stands for binary class entropy loss
    lossFunc = nn.BCELoss()
    optim = torch.optim.SGD(model.parameters(), lr = 0.01) 
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
    # try to genarate a dataset with 2 clusters. One of them with (-0.5, 0.5)
    # center and the other with (0.5, -0.5) center.Both f them has 0.4 as
    # it's standard deviation.
    numOfSamplesPerCluster = 100
    seed = 123
    clusterCenters = [[-0.5, 0.5], [0.5, -0.5]]
    x, y = datasets.make_blobs(
        n_samples = numOfSamplesPerCluster, 
        centers = clusterCenters,
        cluster_std = 0.4,
        random_state = seed)
    #convert data type of x array from floar64 to float32
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    #convert x and y to tensors
    samples = torch.from_numpy(x)
    labels = torch.from_numpy(y.reshape(100, 1))
    #declare and train simple classifiet
    simleClassifier = trainer([samples, labels], 1000)
    #plot created model
    #get model parameters
    w1, w2, b1 = getModelParams(simleClassifier)
    plotFittedModel('trained model', w1, w2, b1, -2.0, 2.0, 'g')    
    # try to test trained model
    point1 = torch.tensor([1.0, -1.0])
    point2 = torch.tensor([-1.0, 1.0])
    plt.subplot(121)
    plt.plot(point1.numpy()[0], point1.numpy()[1], 'ro')
    plt.plot(point2.numpy()[0], point2.numpy()[1], 'ko')

    #plot genarated samples
    plt.subplot(121)
    plt.scatter(x[y == 0, 0], x[y == 0, 1])
    plt.scatter(x[y == 1, 0], x[y == 1, 1])
    plt.show()

    print(simleClassifier.predict(point1))
    print(simleClassifier.predict(point2))



main() 