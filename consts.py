# tuple indices in customer tuple:
# number, coordinates(X,Y), demand, ready(A), due(B), service time
ID, X, Y, DEM, A, B, SRV = range(7)

# list indices in route list structure:
# route len (num edges), capacity, total distance, edge list
R_LEN, R_CAP, R_DIS, R_EDG = range(4)

# list indices in edge list structure (in route edge list)
# customer "a" id, customer "b" id, arrival at "a", latest at "b"
E_FRO, E_TOW, E_ARF, E_LAT = range(4)
