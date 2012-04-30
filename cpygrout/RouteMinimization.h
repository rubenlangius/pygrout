#include "vrptw.h"
#include "rng.h"

namespace vrptw
{

class RouteMinimization
{
    Solution s;
    Problem *p;
    std::vector<int> ejectionPool;
    void initEjectionPool(int toRemove)
    {
        int n = s.routes[toRemove].size();
        ejectionPool.resize(n);
        for(int i=0; i<n; ++i)
            ejectionPool[i] = s.routes[toRemove][i].customer->id;
    }
public:
    RouteMinimization(Problem *p_) : p(p_) {}
    RouteMinimization(Problem &p_) : p(&p_) {}

    bool insert(int v_in)
    {
        std::vector<std::pair<int,int> > Nb_in;
        int maxDemand = p->capacity - p->customers[v_in].demand;
        for(IRoute r = s.routes.begin(); r!=s.routes.end(); ++r)
        {
            if(r->demand > maxDemand)
                continue;
            for(int i=0; i<r->size(); ++i)
            {
                //TODO: check insertion, if possible: add to Nb_in
            }
        }
        if (Nb_in.empty())
            return false;
        return true;
    }

    bool removeRoute()
    {
        int toRemove = RANDINT(s.routes.size());
        initEjectionPool(toRemove);
        s.routes.erase(s.routes.begin()+toRemove);
        int v_in = ejectionPool.back();
        if (insert(v_in))
        {
            ejectionPool.pop_back();
        }
        else
        {
        }
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
