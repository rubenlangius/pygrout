
#include "vrptw.h"
#include "RouteMinimization.h"

#include <iostream>
#include <iterator>

// test

int main()
{
    vrptw::Problem p;
    p.load("../solomons/c101.txt");
    vrptw::RouteMinimization rm(p);
    rm.execute();
    std::cout << rm.getSolution() << "Left in EP: ";
    rm.ejectionPoolExport(std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    return 0;
}
