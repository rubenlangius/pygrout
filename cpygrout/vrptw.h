#ifndef VRPTW_H
#define VRPTW_H

#include <string>
#include <vector>
#include <math.h>

#include <fstream>
using std::ifstream;
using std::istream;

namespace vrptw {

struct Problem;

struct Customer
{
    int x, y;
    int demand;
    float ready_time;
    float due_date;
    int service_time;
    int id;
    Problem *problem;

    istream& load(istream& in)
    {
        in >> id >> x >> y >> demand >> ready_time >> due_date >> service_time;
        return in;
    }
};

struct Problem
{
    std::string name;
    int capacity;
    float horizon;
    std::vector<Customer> customers;
    float distance(int customer1, int customer2)
    {
        return hypot(
             customers[customer1].x-customers[customer2].x,
             customers[customer1].y-customers[customer2].y
        );
    }
    float time(int customer1, int customer2)
    {
        return customers[customer1].service_time + distance(customer1, customer2);
    }
    float arrival_at_next(int c_from, float arrival_at_from, int c_to)
    {
       return std::max(arrival_at_from + time(c_from, c_to), customers[c_to].ready_time);
    }
    float latest_arrival(int c_from, float latest_at_to, int c_to)
    {
       return std::min(latest_at_to - time(c_from, c_to), customers[c_from].due_date);
    }
    void load(std::string filename)
    {
        std::ifstream f(filename.c_str());
        std::string w;
        int vehicles;
        f >> name;
        f >> w >> w >> w; // VEHICLE NUMBER CAPACITY
        f >> vehicles;
        f >> capacity;
        for(int i=0; i<12; ++i) f >> w; // skip CUSTOMER ... SERVICE TIME
        customers.resize(4 * vehicles + 1); // for S and H it's always 4*vehicles
        for(int i=0; i <= 4*vehicles; ++i)
        {
            customers[i].problem = this;
            customers[i].load(f);
        }
        horizon = customers[0].due_date;
    }
};


struct Service
{
    Customer *customer;
    float start;
    float latest;
    Service(Customer *customer, float start, float latest) :
        customer(customer), start(start), latest(latest) {}
};

struct Route
{
    std::vector<Service> services;
    int demand;
    Route() : demand(0) {}
    void init_single(Customer *customer)
    {
        services.clear();
        demand = customer->demand;
        services.push_back(Service(
            customer,
            customer->problem->arrival_at_next(0, 0.0f, customer->id),
            customer->problem->latest_arrival(customer->id, customer->problem->horizon, 0)));
    }
};
typedef std::vector<Service>::iterator IService;

struct Solution
{
    std::vector<Route> routes;
};
typedef std::vector<Route>::iterator IRoute;

void all_customers_as_routes(Problem &p, Solution &s)
{
    int n = p.customers.size()-1;
    s.routes.resize(n);
    for(int i=0; i<n; ++i)
    {
        s.routes[i].init_single(&p.customers[i+1]);
    }
}

}

inline std::ostream& operator<<(std::ostream &out, const vrptw::Solution& s)
{
    for(int i=0; i<s.routes.size(); ++i)
    {
        for(int j=0; j<s.routes[i].services.size(); ++j)
        {
            out << s.routes[i].services[j].customer->id << '-';
        }
        out << "0\n";
    }
    return out;
}

#endif
