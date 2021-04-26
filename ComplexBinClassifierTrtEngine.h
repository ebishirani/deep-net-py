#ifndef COMPLEX_BIN_CLASSIFIER_TRT_ENGINE
#define COMPLEX_BIN_CLASSIFIER_TRT_ENGINE

#include <iostream>
#include <map>

#include <NvInferRuntime.h>
#include <NvInfer.h>

#include "common.h"
#include "logging.h"

using namespace nvinfer1;

class Point
{
public:
    int x = 0;
    int y = 0;    
};
/**
 * @brief 
 * 
 */
class ComplexBinClassifierTrtEngine
{
//Operation section******************************************************
public:
    /**
     * @brief Construct a new ComplexBinClassifierTrtEngine object
     */
    ComplexBinClassifierTrtEngine();
    /**
     * @brief Construct a new ComplexBinClassifierTrtEngine object
     * 
     * @param dataType: Valid data type that build engine can handle
     * @param dataDims: Diamention of data that can feed to model
     * @param maxBatchSize: Maximum valid batch size
     * @param workspaceSize: Maximum allowd memory that can be used as 
     *     workspace
     */
    ComplexBinClassifierTrtEngine(
        DataType dataType, 
        Dims dataDims,
        uint32_t maxBatchSize,
        std::size_t workspaceSize);

    ComplexBinClassifierTrtEngine(ComplexBinClassifierTrtEngine &other) = delete;
    /**
     * @brief Destroy the ComplexBinClassifierTrtEngine object
     */
    ~ComplexBinClassifierTrtEngine();
    /**
     * @brief This function gets waights of traind model and build equivalent
     *     tensorrt engine, then save it in specified path.
     * 
     * @param modelWaightsPath : Path of waights of trained model.
     * @param destinationPath : Destination path to save genetated engine.To 
     *     cancel save operation, user must put "" as destination path
     * @return retutn value specifies the operation result as fallow:
     *     0 : Operation is done successfully
     *     -1: Specified waights path is wrong.
     *     -2: Specified waights is not compatible with this model
     *     -3: Destination file can't be created.
     *     -4: Error in generating network components
     *     -5: Engine was build successfully but can not be serialized
     *     -6: Destination file can't be created.
     */
    int buildAndSaveEngine(std::string modelWaightsPath, 
                           std::string destinationPath);
    /**
     * @brief This function gets previuslly serialize engine path and tru to
     *     load it
     * 
     * @param enginePath : Path of serialized engine
     * @return retutn value specifies the operation result as fallow:
     *     0 : Operation is done successfully
     *     -1: There is no engine file in specified path
     */
    int loadEngine(std::string enginePath);
    /**
     * @brief This function gets a point and specifies its class.
     * 
     * @param inputData : Pointer to a tensor that contains input data
     * @return retutn value specifies the operation result as fallow:
     *     0 : Specified point belongs to class 0.
     *     1 : Specified point belongs to class 1.
     *     -1: There is'nt any loaded engine to petform infrence
     */
    int doInferance(Point p);
protected:
    /**
     * @brief This function gets needed data and build a tesorrt engine
     *     based on pretraind model.
     * 
     * @param [in] modelWaightsPath: Path to .wts file that contain pretraind 
     *     model waights.
     * @param [in] builder: Pointer to a builder module that must be used to
     *     build engine.
     * @param [out] engine: Pointer to created engine must be placed in this
     *     parameter.
     * @return An integer value that specifies the result of this function
     *     operation.Possible values is as fallow:
     *     0 : This function successfully creats an engine.
     *     -1: Specified waights path is wrong.
     *     -2: Specified waights is not compatible with this model
     *     -3: Destination file can't be created.
     *     -4: Error in generating network components
     */
    int buildEngine(
        const std::string modelWaightsPath,
        IBuilder *builder, 
        ICudaEngine **engine);
private:
    /**
     * @brief 
     * 
     * @param filePath : A string that specifies the waights file path 
     * @param result : An integer value that specifies result of this
     *     function operathin.Possible values is as fallow:
     *     0 : Operation terminated successfully
     *     -1: The specified file is not exist.
     *     -2: Specified file is a wrong file.
     * @return This function reurns a map that contains waights of model
     */
    std::map<std::string, nvinfer1::Weights> 
                loadWeights(const std::string& filePath, int &result);
//Property section*******************************************************
public:
protected:
    /**
     * @brief This variable specifies the type of parameters of the model.
     */
    DataType mDataType = DataType::kFLOAT;
    /**
     * @brief Diamentions of input data of model
     */
    nvinfer1::Dims mDataDim;
    /**
     * @brief 
     */
    const char* kInputBlobName = "data";

private:
    /**
     * @brief 
     */
    const char* kOutputBlobName = "prob";
    /**
     * @brief 
     */
    unsigned int mMaxBatchSize = 10;
    /**
     * @brief 
     */
    ICudaEngine *mEngine = nullptr;
    /**
     * @brief 
     */
    Logger mLogger;

    std::size_t mWorkspaceSize = 1 << 20;
};

#endif