
#include "vrptw.h"

// test

int main()
{
    vrptw::Problem p;
    vrptw::load("../solomons/c101.txt", p);
    return 0;
}
