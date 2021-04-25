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
ComplexBinClassifierTrtEngine::ComplexBinClassifierTrtEngine()
{

}
//***************************************************************************
int ComplexBinClassifierTrtEngine::buildAndSaveEngine(
                                    std::string modelWaightsPath, 
                                    std::string destinationPath)
{
    //Define an instance of ILogger to use in tensorrt API methods.
    Logger gLogger;
    //Create a builder module to build engine.
    auto builder = std::unique_ptr<nvinfer1::IBuilder, InferDeleter>
                                             (createInferBuilder(gLogger));
    //Create a networkDefinition module
    auto network = std::unique_ptr<INetworkDefinition, InferDeleter>
                                             (builder->createNetwork());
    auto config = std::unique_ptr<IBuilderConfig, InferDeleter>
                                             (builder->createBuilderConfig());
    // Create input tensor of shape {1, 1} with name kInputBlobName
    ITensor *inputData = network->addInput(
                            this->kInputBlobName, 
                            this->mDataType, 
                            this->mDataDim);
    assert(inputData);

    //Load waits of previously traind model
    std::map<std::string, Weights> weightMap = this->loadWeights(
                                                        modelWaightsPath);
    //Create network layers base on previously trained model structure
    IFullyConnectedLayer *fc = network->addFullyConnected(
                                                *inputData, 4, 
                                                weightMap["mModel.0.weight"],
                                                weightMap["mModel.0.bias"]);
    assert(fc);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* activation = network->addActivation(
                                            *fc->getOutput(0),
                                            ActivationType::kRELU);
    assert(activation);

    fc = network->addFullyConnected(
                            *activation->getOutput(0), 1, 
                            weightMap["mModel.2.weight"],
                            weightMap["mModel.2.bias"]);
    assert(fc);
    // Add activation layer using the ReLU algorithm.
    activation = network->addActivation(
                                *fc->getOutput(0),
                                ActivationType::kSIGMOID);
    assert(activation);
    //REgister output layer
    activation->getOutput(0)->setName(kOutputBlobName);
    network->markOutput(*activation->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(mMaxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    auto engine = std::unique_ptr<ICudaEngine, InferDeleter>
                         (builder->buildEngineWithConfig(*network, *config));

    //After building engine it is possible to release created network and
    //loaded waights
    network = nullptr;
    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }
    
    assert(engine != nullptr);
    //Serialize created engine
    // IHostMemory* modelStream = engine->serialize();
    auto modelStream = std::unique_ptr<IHostMemory, InferDeleter>
                                                   (engine->serialize());
    //After creating model stream it is possible to free engine and config
    //modules
    engine = nullptr;
    config = nullptr;

    assert(modelStream != nullptr);

    //save serialized model on specified file
    std::ofstream outFileStream(destinationPath);
    if (!outFileStream)
    {
        std::cerr << "could not open plan output file" << std::endl;
        return -3;
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
    ComplexBinClassifierTrtEngine::loadWeights(const std::string& file)
{
    std::cout << "Loading weights: " << file << std::endl;

    // Open weights file
    std::ifstream input(file, std::ios::binary);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    std::map<std::string, nvinfer1::Weights> weightMap;
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