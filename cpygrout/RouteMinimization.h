#include "vrptw.h"

namespace vrptw
{

class RouteMinimization
{
    Solution s;
    Problem *p;
public:
    RouteMinimization(Problem *p_) : p(p_) {}
    RouteMinimization(Problem &p_) : p(&p_) {}
    void execute()
    {
        all_customers_as_routes(*p, s);
    }
    Solution getSolution() { return s; }
};

}
