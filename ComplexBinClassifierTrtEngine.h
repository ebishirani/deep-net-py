#ifndef COMPLEX_BIN_CLASSIFIER_TRT_ENGINE
#define COMPLEX_BIN_CLASSIFIER_TRT_ENGINE

#include <iostream>

#include <NvInferRuntime.h>

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
     * 
     */
    ComplexBinClassifierTrtEngine();
    /**
     * @brief This function gets waights of traind model and build equivalent
     *     tensorrt engine, then save it in specified path.
     * 
     * @param modelWaightsPath : Path of waights of trained model.
     * @param destinationPath : Destination path to save genetated engine.To 
     *     cancel save operation, user must put "" as destination path
     * @return retutn value specifies the operation result as fallow:
     *     0 : Operation is done successfully
     *     -1: Specified waights path is wrong
     *     -2: Specified waights is not compatible with this model
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
     * @param p : The point that must be classify.
     * @return retutn value specifies the operation result as fallow:
     *     0 : Specified point belongs to class 0.
     *     1 : Specified point belongs to class 1.
     *     -1: There is'nt any loaded engine to petform infrence
     */
    int doInferance(Point p);
protected:
//                      There is no ptrotected member in this class
private:
    ICudaEngine* mEngine = nullptr;

//Property section*******************************************************
public:
protected:
private:
};

#endif