#include <iostream>

#include "ComplexBinClassifierTrtEngine.h"

int main()
{
    ComplexBinClassifierTrtEngine engine;
    Point p;
    p.x = 10;
    p.y = 20;
    engine.doInferance(p);
    std::cout << "Hello world" << std::endl;   
    return 0;
}