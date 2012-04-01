
#include <string>
#include <vector>
#include <math.h>

#include <fstream>
using std::ifstream;
using std::istream;

namespace vrptw {

struct Customer
{
    int x, y;
    int demand;
    int ready_time;
    int due_date;
    int service_time;
    istream& load(istream& in)
    {
        int id;
        in >> id >> x >> y >> demand >> ready_time >> due_date >> service_time;
        return in;
    }
};

inline istream& operator>>(istream &in, Customer &c)
{
    return c.load(in);
}

struct Problem
{
    std::string name;
    int capacity;
    std::vector<Customer> customers;
    float distance(int customer1, int customer2)
    {
        return hypot(
             customers[customer1].x-customers[customer2].x,
             customers[customer1].y-customers[customer2].y
        );
    }
};

inline void load(std::string filename, Problem &p)
{
    std::ifstream f(filename.c_str());
    std::string w;
    int vehicles;
    f >> p.name;
    f >> w >> w >> w; // VEHICLE NUMBER CAPACITY
    f >> vehicles;
    f >> p.capacity;
    for(int i=0; i<12; ++i) f >> w;
    p.customers.resize(4 * vehicles + 1);
    for(int i=0; i <= 4*vehicles; ++i)
        p.customers[i].load(f);
}

struct Service
{
    int customer_id;
    float start;
    float latest;
    Service(int customer_id, float start, float latest) :
        customer_id(customer_id), start(start), latest(latest) {}
};

struct Solution
{
    std::vector<std::vector<Service> > routes;
};

void all_customers_as_routes(Problem &p, Solution &s);

Solution route_minimization(Problem &p)
{
    Solution s;
    all_customers_as_routes(p, s);
}

void all_customers_as_routes(Problem &p, Solution &s)
{
    int n = p.customers.size();
    s.routes.resize(n);
    for(int i=0; i<n; ++i)
    {
        s.routes[i].clear();
        s.routes[i].push_back(Service(i+1, p.distance(0, i+1), p.customers[i+1].due_date));
    }
}

}
