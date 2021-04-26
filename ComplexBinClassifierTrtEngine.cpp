#include "ComplexBinClassifierTrtEngine.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>

#include <NvInfer.h>

#include "logging.h"
#include "common.h"

int ComplexBinClassifierTrtEngine::doInferance(Point p)
{
    std::cout << "Hello ebi" << std::endl;
    return 0;
}
//***************************************************************************
int ComplexBinClassifierTrtEngine::loadEngine(std::string enginePath)
{
    return 0;
}
//***************************************************************************
ComplexBinClassifierTrtEngine::ComplexBinClassifierTrtEngine(
    DataType dataType, 
    Dims dataDims,
    uint32_t maxBatchSize,
    std::size_t workspaceSize):
    mDataType(dataType),
    mDataDim(dataDims),
    mMaxBatchSize(maxBatchSize),
    mWorkspaceSize(workspaceSize)
    {}
//***************************************************************************
ComplexBinClassifierTrtEngine::ComplexBinClassifierTrtEngine():
    ComplexBinClassifierTrtEngine(DataType::kFLOAT, Dims2(1, 2), 1, 1 << 20)
{}
//***************************************************************************
ComplexBinClassifierTrtEngine::~ComplexBinClassifierTrtEngine()
{
    if (nullptr != this->mEngine)
    {
        this->mEngine->destroy();
    }
}
//***************************************************************************
//Builder module is passed to this function because of logger object is
//a privare member of TrtEngineBuilder class.
int ComplexBinClassifierTrtEngine::buildEngine(
    const std::string modelWaightsPath, 
    IBuilder *builder, 
    ICudaEngine **engine)
{
    //Assume this function can not create engine
    *engine = nullptr;
    //Create a networkDefinition module
    auto network = std::unique_ptr<INetworkDefinition, InferDeleter>
                                             (builder->createNetworkV2(0U));
    auto config = std::unique_ptr<IBuilderConfig, InferDeleter>
                                             (builder->createBuilderConfig());
    // Create input tensor of shape {1, 1} with name kInputBlobName
    nvinfer1::ITensor *inputData = network->addInput(
                            this->kInputBlobName, 
                            this->mDataType, 
                            this->mDataDim);
    // assert(inputData);
    if (nullptr == inputData)
    {
        return -4;
    }
    //Load waits of previously traind model
    int result = 0;
    std::map<std::string, Weights> weightMap = this->loadWeights(
                                                        modelWaightsPath, 
                                                        result);
    if (result < 0)
    {
        return result;
    }
    
    //Create network layers base on previously trained model structure
    IFullyConnectedLayer *fc = network->addFullyConnected(
                                                *inputData, 4, 
                                                weightMap["mModel.0.weight"],
                                                weightMap["mModel.0.bias"]);
    // assert(fc);
    if (nullptr == fc)
    {
        return -4;
    }
    // Add activation layer using the ReLU algorithm.
    IActivationLayer* activation = network->addActivation(
                                            *fc->getOutput(0),
                                            ActivationType::kRELU);
    // assert(activation);
    if (nullptr == activation)
    {
        return -4;
    }
    fc = network->addFullyConnected(
                            *activation->getOutput(0), 1, 
                            weightMap["mModel.2.weight"],
                            weightMap["mModel.2.bias"]);
    // assert(fc);
    if (nullptr == fc)
    {
        return -4;
    }
    // Add activation layer using the ReLU algorithm.
    activation = network->addActivation(
                                *fc->getOutput(0),
                                ActivationType::kSIGMOID);
    // assert(activation);
    if (nullptr == activation)
    {
        return -4;
    }
    //REgister output layer
    activation->getOutput(0)->setName(kOutputBlobName);
    network->markOutput(*activation->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(mMaxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);

    *engine = builder->buildEngineWithConfig(*network, *config);
    //After building engine it is possible to release created network and
    //loaded waights
    network = nullptr;
    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }
    //Check is engine created successfully?
    if (nullptr == *engine)
    {
        return -4;
    }
    
    return 0;
}
//***************************************************************************
int ComplexBinClassifierTrtEngine::buildAndSaveEngine(
                                    std::string modelWaightsPath, 
                                    std::string destinationPath)
{
    //Create a builder module to build engine.
    auto builder = createInferBuilder(this->mLogger);
    
    int result = this->buildEngine(
        modelWaightsPath,
        builder,
        &this->mEngine);
    //Check is any error occured in engine creation
    if (0 != result)
    {
       return result;
    }
    //Serialize created engine
    // IHostMemory* modelStream = engine->serialize();
    auto modelStream = std::unique_ptr<IHostMemory, InferDeleter>
                                                (this->mEngine->serialize());
    //Check is serialization done successfully?
    if (nullptr == modelStream)
    {
        return -5;
    }
    // assert(modelStream != nullptr);

    //save serialized model on specified file
    std::ofstream outFileStream(destinationPath);
    if (!outFileStream || false == outFileStream.is_open())
    {
        std::cerr << "could not open plan output file" << std::endl;
        return -6;
    }
    outFileStream.write(reinterpret_cast<const char*>
                                 (modelStream->data()), modelStream->size());
    modelStream = nullptr;
    return 0;                                         
}
//***************************************************************************
//!
//! \brief Loads weights from weights file
//!
//! \details TensorRT weight files have a simple space delimited format
//!          [type] [size] <data x size in hex>
//!
std::map<std::string, nvinfer1::Weights> 
    ComplexBinClassifierTrtEngine::loadWeights(
        const std::string& file, 
        int &result)
{
    std::cout << "Loading weights: " << file << std::endl;
    //Assume that function can load waights successfully.
    result = 0;
    //Build a map to hold loaded waights
    std::map<std::string, nvinfer1::Weights> weightMap;
    
    // Open weights file
    std::ifstream input(file, std::ios::binary);
    if (false == input.is_open())
    {
        result = -1;
        return weightMap;
    }
    
    // Read number of weight blobs
    int32_t count;
    input >> count;
    if (count <= 0)
    {
        result = -2;
        return weightMap;
    }
    
    // assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        nvinfer1::Weights wt{DataType::kFLOAT, nullptr, 0};
        int type;
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> type >> size;
        wt.type = static_cast<DataType>(type);

        // Load blob
        if (wt.type == DataType::kFLOAT)
        {
            uint32_t* val = new uint32_t[size];
            for (uint32_t x = 0; x < size; ++x)
            {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }
        else if (wt.type == DataType::kHALF)
        {
            uint16_t* val = new uint16_t[size];
            for (uint32_t x = 0; x < size; ++x)
            {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}