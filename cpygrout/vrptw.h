#ifndef VRPTW_H
#define VRPTW_H

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
    int id;
    istream& load(istream& in)
    {
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
    for(int i=0; i<12; ++i) f >> w; // skip CUSTOMER ... SERVICE TIME
    p.customers.resize(4 * vehicles + 1); // for S and H it's always 4*vehicles
    for(int i=0; i <= 4*vehicles; ++i)
        p.customers[i].load(f);
}

struct Service
{
    Customer *customer;
    float start;
    float latest;
    Service(Customer *customer, float start, float latest) :
        customer(customer), start(start), latest(latest) {}
};

typedef std::vector<Service> serviceVector;
struct Route : public serviceVector
{
    int demand;
    void clear()
    {
        serviceVector::clear();
        demand = 0;
    }
    void push_back(const Service &s)
    {
        serviceVector::push_back(s);
        demand += s.customer->demand;
    }
};

struct Solution
{
    std::vector<Route> routes;
};
typedef std::vector<Route>::iterator IRoute;


void all_customers_as_routes(Problem &p, Solution &s);

Solution route_minimization(Problem &p)
{
    Solution s;
    all_customers_as_routes(p, s);
    return s;
}

void all_customers_as_routes(Problem &p, Solution &s)
{
    int n = p.customers.size()-1;
    s.routes.resize(n);
    for(int i=0; i<n; ++i)
    {
        s.routes[i].clear();
        s.routes[i].push_back(Service(&p.customers[i+1], p.distance(0, i+1), p.customers[i+1].due_date));
    }
}

}

inline std::ostream& operator<<(std::ostream &out, const vrptw::Solution& s)
{
    for(int i=0; i<s.routes.size(); ++i)
    {
        for(int j=0; j<s.routes[i].size(); ++j)
        {
            out << s.routes[i][j].customer->id << '-';
        }
        out << "0\n";
    }
    return out;
}

#endif
