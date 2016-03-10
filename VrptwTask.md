# Introduction #

The class VrptwTask is used by VrptwSolution to store constant (read-only) information about the specific case (an instance) of the Vehicle Routing Problem with Time Windows.

# Details #

Fields of this class represent specific features of a single problem.

The data of an instance of the VRPTW problem consists of:
  * the vehicle maximal capacity,
  * location of the depot and the _scheduling horizon_ (depot's time windows),
  * locations, time windows, service times and demands of customers.