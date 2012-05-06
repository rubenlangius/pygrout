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
        Customer *customer = &p->customers[v_in];
        float latest = p->latest_arrival(next.customer->id, next.latest, v_in);
        IService fwd = ins.r->services.insert(
                ins.r->services.begin() + ins.pos,
                Service(customer, ins.arrival, latest));
	IService end_(ins.r->services.end());
        // demand
        ins.r->demand += customer->demand;
        // time windows
        int c_from = v_in;
        float last_arrival = ins.arrival;
        for (++fwd; fwd != end_; ++fwd)
        {
            float arrival = p->arrival_at_next(c_from, last_arrival, fwd->customer->id);
            if (arrival == fwd->start)
                break;
            last_arrival = fwd->start = arrival;
            c_from = fwd->customer->id;            
        }
        int c_to = v_in;
        for(IrService back(fwd), rend(ins.r->services.rend()); back != rend; ++back)
        {            
            float new_latest = p->latest_arrival(back->customer->id, latest, c_to);
            if (new_latest == back->latest)
                break;
            back->latest = latest = new_latest;
            c_to = back->customer->id;
        }
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
        while (!ejectionPool.empty())
        {
            int v_in = ejectionPool.back();
            if (insert(v_in))
            {
                ejectionPool.pop_back();
            }
            else
            {
                return false;
            }
        }
        return true;
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
    template<class Iterator>
    void ejectionPoolExport(Iterator out)
    {
        std::copy(ejectionPool.begin(), ejectionPool.end(), out);
    }
};

}
