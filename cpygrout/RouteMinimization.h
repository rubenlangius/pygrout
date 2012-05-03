#include "vrptw.h"
#include "rng.h"

namespace vrptw
{

class RouteMinimization
{
    Solution s;
    Problem *p;
    std::vector<int> ejectionPool;

    struct Insertion
    {
        Route *r;
        float arrival;
        int pos;
    };

    void initEjectionPool(int toRemove)
    {
        int n = s.routes[toRemove].services.size() - 1;
        ejectionPool.resize(n);
        for(int i=0; i<n; ++i)
            ejectionPool[i] = s.routes[toRemove].services[i].customer->id;
    }
    void insert_customer(const Insertion &ins, int v_in)
    {
        Service& next = ins.r->services[ins.pos];
        float latest = p->latest_arrival(next.customer->id, next.latest, v_in);
        ins.r->services.insert(
                ins.r->services.begin() + ins.pos,
                Service(&p->customers[v_in], ins.arrival, latest));
        // TODO: update starts ahead, and latest's before v_in
    }
public:
    RouteMinimization(Problem *p_) : p(p_) {}
    RouteMinimization(Problem &p_) : p(&p_) {}

    bool insert(int v_in)
    {
        std::vector<Insertion> Nb_in;
        int maxDemand = p->capacity - p->customers[v_in].demand;
        float v_in_due = p->customers[v_in].due_date;
        for(IRoute r = s.routes.begin(); r!=s.routes.end(); ++r)
        {
            if(r->demand > maxDemand)
                continue;
            int c_from = 0;
            float time_at = 0;
            for(int i=0; i<r->services.size(); ++i)
            {
                const Customer * customer_to = r->services[i].customer;
                float possible_arrival = p->arrival_at_next(c_from, time_at, v_in);
                if (possible_arrival <= v_in_due)
                {
                    float next_arrival = p->arrival_at_next(v_in, possible_arrival, customer_to->id);
                    if (next_arrival <= customer_to->due_date)
                    {
                        Insertion insertion = { &(*r), possible_arrival, i };
                        Nb_in.push_back(insertion);
                    }
                }
                c_from = customer_to->id;
                time_at = r->services[i].start;
            }
        }
        if (Nb_in.empty())
            return false;
        int randomInsertion = RANDINT(Nb_in.size());
        insert_customer(Nb_in[randomInsertion], v_in);
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
