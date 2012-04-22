#include "vrptw.h"
#include "rng.h"

namespace vrptw
{

class RouteMinimization
{
    Solution s;
    Problem *p;
    std::vector<int> ejectionPool;
public:
    RouteMinimization(Problem *p_) : p(p_) {}
    RouteMinimization(Problem &p_) : p(&p_) {}
    bool removeRoute()
    {
        int toRemove = RANDINT(s.routes.size());
        // TODO: init EP, loop etc.
        return false;
    }
    void execute()
    {
        all_customers_as_routes(*p, s);
        bool success;
        do
        {
            success = removeRoute();
        } while (success);
    }
    Solution getSolution() { return s; }
};

}
