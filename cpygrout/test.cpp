
#include "vrptw.h"
#include <iostream>

// test

int main()
{
    vrptw::Problem p;
    vrptw::load("../solomons/c101.txt", p);
    std::cout << vrptw::route_minimization(p);
    return 0;
}
