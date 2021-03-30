import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.convSection = nn.Sequential(
            nn.Conv2d(3, 6, 5),#(N, 3, 32, 32) ---> (N, 6, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2),#(N, 6, 28, 28) ---> (N, 6, 14, 14)
            nn.ReLU(),
            nn.Conv2d(6, 16, 5),#(N, 6, 14, 14) ---> (N, 16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2)# (N, 16, 10, 10) ----> (N, 16, 5, 5)
        )
        self.fcSection = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
    #Forward function
    def forward(self, x):
        x = self.convSection(x)
        y = x.view(-1, self.num_flat_features(x))
        y = self.fcSection(y)
        return y


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension

        num_features = 1
        for s in size:
            num_features *= s
        return num_features
#This class load train and test data from CFAR10 dataset
class Data():
    def __init__(self, dataSetPath, batchSize):
        transform = transforms.Compose(
            [transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #load train data
        self.trainSet = torchvision.datasets.CIFAR10(
            root = dataSetPath, 
            train = True,
            download = True, 
            transform = transform)
        self.trainLoader = torch.utils.data.DataLoader(
            self.trainSet, 
            batch_size = batchSize,
            shuffle = True, 
            num_workers = 2)
        #load test data
        self.testSet = torchvision.datasets.CIFAR10(
            root = dataSetPath, 
            train = False,
            download = True, 
            transform = transform)
        self.testLoader = torch.utils.data.DataLoader(
            self.testSet, 
            batch_size = batchSize,
            shuffle = False, 
            num_workers = 2)

#This function is used to train the model
def trainer(dataLoader : Data, numOfEpochs = 50, mustSaveModel = True) -> LeNet:
    #Declare a network.
    net = LeNet()
    #Specify loss function
    lossFunc = nn.CrossEntropyLoss()
    #Select an optimizer to update parameters of model
    optimizer = torch.optim.Adam(net.parameters())

    #try to train model
    for i in range(numOfEpochs):
        for data in dataLoader.trainLoader:
            imageData, labels = data

            optimizer.zero_grad()

            predLabels = net(imageData)
            loss = lossFunc(predLabels, labels)
            loss.backward()
            optimizer.step()

    if True == mustSaveModel:        
        currentDir = os.path.dirname(os.path.abspath(__file__))
        torch.save(net.state_dict(), currentDir + '/LeNet.pth')
    
    return net

def evaluateModel(loader, net):    
    total = 0
    correct = 0
    for data in loader:
        imageData, labels = data
        out = net(imageData)
        maxValues, predClass = torch.max(out, 1)
        total += labels.shape[0]
        correct += (predClass == labels).sum().item()
        accuracy = (100 * correct) / total
        return accuracy

def main():
    dataSetPath = '/media/sam/DeletedC/work_dir/deep_learning_projects/CIFAR10/data/cifar-10-batches-py'
    #load dataset
    dataLoader = Data(dataSetPath , batchSize = 256)
    #creat and train a model
    net = trainer(dataLoader)
    #evaluate created model
    testAcc = evaluateModel(dataLoader.testLoader, net)
    print('Accuracy of model on test set is : ', testAcc)

main()
