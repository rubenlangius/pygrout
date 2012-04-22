
#include "vrptw.h"
#include "RouteMinimization.h"

#include <iostream>

// test

int main()
{
    vrptw::Problem p;
    vrptw::load("../solomons/c101.txt", p);
    vrptw::RouteMinimization rm(p);
    rm.execute();
    std::cout << rm.getSolution();
    return 0;
}
