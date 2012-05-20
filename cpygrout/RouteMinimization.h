#include "vrptw.h"
#include "rng.h"

#include <float.h>

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

    struct SqueezeInsertion : public Insertion
    {
        float demand_penalty;
        float timewin_penalty;
        float total_penalty;
        SqueezeInsertion() : total_penalty(FLT_MAX) {}
        SqueezeInsertion(Route *r_, float arrival_, int pos_, float demand_penalty_, float timewin_penalty_) :
            demand_penalty(demand_penalty_), timewin_penalty(timewin_penalty_)
        {
            r = r_;
            arrival = arrival_;
            pos = pos_;
        }
    };

    class SqueezeInserter
    {
        // const float alpha_decr = 0.99f;
        float alpha;
        Problem *p;
    public:
        SqueezeInsertion check_before(IRoute r, IService is, int v_in)
        {
            int c_from = 0;
            float arrival_c_from = 0;
            if (is == r->services.begin())
            {
                IService prev(is);
                --prev;
                c_from = prev->customer->id;
                arrival_c_from = prev->start;
            }
            float arrival_v_in = p->arrival_at_next(c_from, arrival_c_from, v_in);
            float demand_penalty = std::max(0, p->capacity - r->demand - p->customers[v_in].demand);
            float timewin_penalty = std::max(0.0f, p->customers[v_in].due_date - arrival_v_in);
            SqueezeInsertion result(&(*r), arrival_v_in, is - r->services.begin(), demand_penalty, timewin_penalty);
            float arrival_prev = arrival_v_in;
            c_from = v_in;
            IService end(r->services.end());
            for(IService is2 = is; is2 != end; ++is2)
            {
                int c_to = is2->customer->id;
                float arrival = p->arrival_at_next(c_from, arrival_prev, c_to);
                if (arrival < is2->latest)
                    break;
                if (arrival > is2->customer->due_date)
                    result.timewin_penalty += is2->customer->due_date - arrival;
                c_from = c_to;
                arrival_prev = arrival;
            }
            result.total_penalty = alpha * result.timewin_penalty + demand_penalty;
            return result;
        }
        SqueezeInserter(Problem *p_) : p(p_), alpha(1.0f) {}
    } squeezeInserter;

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
    RouteMinimization(Problem *p_) : p(p_),  squeezeInserter(p_)  {}
    RouteMinimization(Problem &p_) : p(&p_), squeezeInserter(&p_) {}

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

    bool squeeze(int v_in)
    {
        SqueezeInsertion best;
        int customer_demand = p->customers[v_in].demand;
        for (IRoute r = s.routes.begin(); r!=s.routes.end(); ++r)
        {
            float demand_penalty = r->demand;
            int c_from = 0;
            float arrival = 0;
            IService r_end = r->services.end();
            for (IService is=r->services.begin(); is != r_end; ++is)
            {
                SqueezeInsertion ins = squeezeInserter.check_before(r, is, v_in);
                if (ins.total_penalty < best.total_penalty)
                    best = ins;
            }
        }
        // TODO: insert best
        return false;
    }

    bool removeRoute()
    {
        Solution backup(s);
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
            else if (squeeze(v_in))
            {
                ejectionPool.pop_back();
            }
            else
            {
                // TODO: eject-insert
                s = backup;
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
