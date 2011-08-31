#!/usr/bin/env python

from itertools import count, izip, dropwhile
from collections import deque
from operator import itemgetter
from bisect import bisect_left

import sys
import os
import time
import cPickle

import numpy as np

# additional modules
from vrptw import *
from vrptw.consts import *
from compat import *
        
# NEIGBOURHOOD OPERATORS - single step trials
    
operations = dict()
def operation(func):
    """A decorator for single solution operations."""
    operations[func.__name__] = func
    return func

@operation
def op_greedy_single(sol, randint = r.randint):
    """Neighbourhood operator - remove random customer and insert back."""
    # pick a route
    r = randint(0, sol.k-1)
    pos = randint(0, sol.r[r][R_LEN]-2)
    c = remove_customer(sol, r, pos)  
    insert_customer(sol, c)

@operation
def op_greedy_multiple(sol, randint = r.randint):
    """Remove a few customers from a random route and insert them back."""
    r = randint(0, sol.k-1)
    num_removed = randint(1, min(9, sol.r[r][R_LEN]-1))
    removed = []
    for i in xrange(num_removed):
        removed.append(remove_customer(sol, r, randint(0, sol.r[r][R_LEN]-2)))
    for c in removed:
        insert_customer(sol, c)

def pick_short_route(sol, random=r.random):
    """Pick a route with preference for shortest."""
    r_lengths = np.array([1.0/(rt[R_LEN]-1) for rt in sol.r]).cumsum()
    return bisect_left(r_lengths, random()*r_lengths[-1])

def pick_long_route(sol, random=r.random):
    """Return a random route, with preference for the longer."""
    lengths = np.array([rt[R_LEN]-1 for rt in sol.r]).cumsum()
    return bisect_left(lengths, random()*lengths[-1])
             
@operation
def op_fight_shortest(sol, random=r.random, randint=r.randint):
    """Picks and tries to empty a random route with preference for shortest."""
    r = pick_short_route(sol)
    num_removed = min(randint(1, 10), sol.r[r][R_LEN]-1)
    removed = []
    for i in xrange(num_removed):
        removed.append(remove_customer(sol, r, randint(0, sol.r[r][R_LEN]-2)))
    for c in removed:
        insert_customer(sol, c)

    
@operation
def op_tabu_single(sol, randint = r.randint, choice=r.choice):
    """Pick one customer from a random route and move him to a different."""
    r = pick_short_route(sol)
    old_len = sol.r[r][R_LEN]
    old_k = sol.k
    pos = randint(0, old_len-2)
    c = remove_customer(sol, r, pos)
    # d("Route %d, from %d, removed customer %d"%(r,pos,c))  
    for tries in xrange(sol.k-1):
        # max k tries
        r2 = randint(0, sol.k-2)
        # picking all other with equal probability
        if r2 >= r and sol.k == old_k: r2 +=1
        # d("other route %d" % r2)
        if sol.r[r2][R_LEN] < old_len:
            continue
        candidates = sorted(find_allpos_on(sol, c, r2))
        if not candidates:
            continue
        dist, pos = candidates[-1] # choice(candidates)
        # d("found pos %d (%.2f inc)" % (pos, dist))
        insert_at_pos(sol, c, r2, pos)
        return
    # customer c from r failed to move - insert him back
    u.undo()

@operation
def op_tabu_shortest(sol, randint = r.randint):
    r = pick_short_route(sol)
    num_removed = randint(1, sol.r[r][R_LEN]-1)
    removed = []
    for i in xrange(num_removed):
        removed.append(remove_customer(sol, r, randint(0, sol.r[r][R_LEN]-2)))
    for c in removed:
        tried = set()
        found = False
        for tries in xrange(sol.k*2):
            # max k tries
            r2 = pick_long_route(sol)
            if r2 in tried:
                continue
            # print "Long route", r2
            # time.sleep(0.6)
            dist, pos = find_bestpos_on(sol, c, r2)
            if pos:
                insert_at_pos(sol, c, r2, pos)
                found = True
                break
            tried.add(r2)
        if not found:
            u.undo()
            return
    ##print "We displaced %d customers from %d:" % (num_removed, r), removed
    
# major solution functions (metaheuristics)

def build_first(sol):
    """Greedily construct the first solution."""
    sol.reset()
    for c in sol.task.getSortedCustomers():
        insert_customer(sol, c[ID])
    sol.mem['init_order'] = VrptwTask.sort_order
    u.commit()
    sol.loghist()

def build_by_savings(sol, wait_limit = None, mi = 1):
    """Construct a new solution by savings heuristic."""
    
    def check_saving(x, y):
        """Compute and return possible saving for concatenating x and y.
        
        The return value is a tuple (saving, wait_time)"""
        xk, _, arr_xk, _ = sol.r[x][R_EDG][-1]
        _, y0, _, larr_y0 = sol.r[y][R_EDG][0]
        arr_y0 = arr_xk + sol.t(xk, y0)
        wait_y0 = max(0, sol.a(y0) - arr_y0)
        if (sol.r[x][R_CAP]+sol.r[y][R_CAP] > sol.task.capa
            or arr_y0 > larr_y0 
            or (wait_limit and wait_y0 > wait_limit) ):
            return None, None
        return sol.d(xk, 0) + sol.d(0, y0) - mi*sol.d(xk, y0), wait_y0
        
    def list_savings():
        """Return list of possible savings as [(saving, route 1, route 2)]."""
        savings = []
        for i in xrange(sol.k):
            for j in xrange(sol.k):
                if i <> j:
                    s, w = check_saving(i, j)
                    if s is not None:
                        savings.append((s, -w, i, j))                
        return savings
        
    sol.reset()
    for c in xrange(sol.task.N):
        insert_new(sol, c+1)
    while True:
        savings = list_savings()
        if len(savings) == 0:
            break
        sav, wt, r1, r2 = max(savings)
        # print 'saving', sav, 'by join of', r1, r2, 'wait', wt, 'in', sol.task.name
        join_routes(sol, r1, r2)
        # turned off for efficiency (now assumed correct)
        # sol.check()
    u.commit()
    return sol

def build_by_mfsavings(sol, wait_limit = None, mi = 1):
    """Build by maybe faster savings heuristic implementation.
    Should actually provide the same results as 'normal' O(n**3) savings."""
    sol.reset()
    for c in xrange(sol.task.N):
        insert_new(sol, c+1)
    prevs = [0] * (sol.task.N+1)
    nexts = [0] * (sol.task.N+1)
    route = [None]+sol.r
    possible = [ (mi*sol.d(i,j) - sol.d(0, i) - sol.d(j, 0), i, j) 
    for i in xrange(1, sol.task.N+1)
        for j in xrange(1, sol.task.N+1)
            if i <> j ]
    possible.sort()         
    # TODO: check validity and perform savings
    for sav, i, j in possible:
        # third condition: already joint (ends of the same route)
        if nexts[i] or prevs[j] or route[i]==route[j]:
            continue
        #  check arrivals
        xk, _, arr_xk, _ = route[i][R_EDG][-1]
        _, y0, _, larr_y0 = route[j][R_EDG][0]
        arr_y0 = arr_xk + sol.t(xk, y0)
        wait_y0 = max(0, sol.a(y0) - arr_y0)
        if (route[i][R_CAP]+route[j][R_CAP] > sol.task.capa
            or arr_y0 > larr_y0 
            or (wait_limit and wait_y0 > wait_limit) ):
            continue
        # join routes
        print "Joining %d and %d" % (i, j)
        prevs[j] = i
        nexts[i] = j
        # remember last customer, before joining
        last_rj = route[j][R_EDG][-1][E_FRO]
        prevdist = sol.dist
        join_routes_ref(sol, route[i], route[j])
        # careful - this must come after joining
        route[last_rj] = route[i]
        # print "result:\n", route[i]
        # from compat import print_like_Czarnas
        # print_like_Czarnas(sol, sparse=True)
        # TURN ON IF NECCESSARY:
        # sol.check_full()
        
    return sol

def local_search(sol, oper, end=0, verb=False, speed=None):
    """Optimize solution by local search."""
    # local rebinds
    ci=u.commit; undo=u.undo; val=sol.val 
    oldval = val()
    from time import time
    # stats
    updates = 0
    steps = 0
    start = time()
    if end == 0:
        end = time()+3
    while time() < end:
        steps += 1
        oper(sol)
        newval = val()
        if  newval < oldval:
            oldval = newval
            updates += 1
            sol.loghist()
            ci()
        elif val()[0] == oldval[0]:
            # huh, not worse, when it comes to routes
            ci()
        else:
            undo()
    elapsed = time()-start
    if verb:
        print " ".join([ sol.infoline(), 
          "%.1f s, %.2f fps, %d acc (%.2f aps)" % (
          elapsed, steps/elapsed, updates, updates/elapsed) ])
    # fps measurement from outside
    if not speed is None:
        speed.append(steps/elapsed)
    sol.loghist()
    return sol
    

# MISC. SOLUTION FUNCTIONS - postprocessing

def plot_history(sol):
    """Display a matplotlib graph of solution progress"""
    from matplotlib import pyplot as plt
    k, dist, t = zip(*sol.history)
    fig = plt.figure()
    
    fig.suptitle(sol.task.name+" "+sol.infoline())
    
    # subplot of routes vs. time
    kplt = fig.add_subplot(121)
    kline = kplt.plot(t, k, 'g')
    min_k = (sol.task.best_k or 2)-2
    # scaling
    kplt.axis([0, sol.history[-1][2], min_k, max(k)+1])
    # labels etc.
    plt.xlabel('time [s]')
    plt.ylabel('routes (k)')
    if sol.task.best_k:
        kplt.axhline(sol.task.best_k+0.03)
    
    # subplot of distance vs. time
    dplt = fig.add_subplot(122)
    dline = dplt.plot(t, dist, 'g')
    # scaling the plot
    min_d = min(dist+(sol.task.best_dist,))
    max_d = max(dist+(sol.task.best_dist,))
    span_d = max_d - min_d
    dplt.axis([0, sol.history[-1][2], min_d-span_d/20., max_d+span_d/20.])
    # decoration with labels, etc.
    plt.grid(True)
    dplt.set_xlabel('time [s]')
    dplt.set_ylabel('dist')
    dplt.yaxis.set_label_position("right")
    dplt.yaxis.set_ticks_position("right")
    if sol.task.best_dist:
        dplt.axhline(sol.task.best_dist)
    plt.show()

# aggressive route minimization

def find_replace_pos_on(sol, c, r):
    """Return a position (occupied), where the customer could be inserted."""
    # pull out deep things locally
    time = sol.task.time
    cust = sol.task.cust
    dist = sol.task.dist
    c_A = cust[c][A]
    c_B = cust[c][B]
    edges = sol.r[r][R_EDG]
    q_out = sol.r[r][R_CAP] + cust[c][DEM]  - sol.task.capa
    # customers - d - deleted, a - starting, b - final, c - inserted
    a, d, arr_a, _ = edges[0]
    for pos in xrange(1, len(edges)):
        d, b, arr_d, larr_b = edges[pos]
        
        # check for too early positions, and weight constraint
        if c_A > larr_b or cust[d][DEM] < q_out:
            a, d, arr_a, larr_d = d, b, arr_d, larr_b
            continue
        
        # check for too late - end of scan
        if arr_a > c_B:
            break
        
        arr_c = max(c_A, arr_a+time[a][c])
        arr_b = max(cust[b][A], arr_c+time[c][b])
        larr_c = min(c_B, larr_b-time[c][b])
        larr_a = min(cust[a][B], larr_c-time[c][b])
        
        if arr_a <= larr_a and arr_c <= larr_c and arr_b <= larr_b:
            distinc = dist[a][c]+dist[c][b]-(dist[a][d]+dist[d][b])
            yield (distinc, pos-1)
        # for next loop pass:
        a, d, arr_a, larr_d = d, b, arr_d, larr_b

def find_replace_pos(sol, c):
    for r in xrange(len(sol.r)):
        # replacing single customer makes little sense
        if sol.r[r][R_LEN] > 2:
            for distinc, pos in find_replace_pos_on(sol, c, r):
                yield (distinc, r, pos)
            
def short_light_route(sol):
    """Return the index of the shortest of the three lightest routes."""
    from heapq import nsmallest
    if sol.k > 3:
        candidates = nsmallest(3, xrange(sol.k), key=lambda x: sol.r[x][R_CAP])
    else:
        candidates = xrange(sol.k)
    return min( (sol.r[i][R_LEN], i) for i in candidates )[1]
    
def remove_route(sol, r):
    """Remove a route and retur a list of its customers."""
    data = u.pop(sol.r, r)
    cust = map(itemgetter(0), data[R_EDG])[1:]
    u.ada(sol, 'k', -1)
    u.ada(sol, 'dist', -data[R_DIS])    
    return cust

@operation
def op_route_min(sol, route=None, random=r.random, randint=r.randint, data=dict(die=0)):
    """Emulate the route minimization (RM) heuristic by Nagata et al."""
    from collections import deque, defaultdict
    if route is None:
        r = short_light_route(sol)
    else:
        r = route
    # print "I'll try to eliminate route", r+1
    ep = deque(remove_route(sol, r))
    # print "%d customers left to go:"% len(ep), ep
    
    def insert(c, r, pos, ep):
        # print "Customer %d goes to %d at pos %d" % (c, r+1, pos)
        insert_at_pos(sol, c, r, pos)
        #print_like_Czarnas(sol)
        # print "Still left are:", ep
        
    recycled = defaultdict(int)
    def put_to_ep(c, front=True):
        if front:
            ep.appendleft(c)
        else:
            ep.append(c)
            
        recycled[c] += 1
        # print "Next (%d) round for %d" % (c, recycled[c])
        
        if any(recycled[x] > 5 for x in ep):
            # print "Too much recycling in the EP: dead end"
            raise RuntimeError
        
    while len(ep) > 0 and not data['die']:
        c = ep.pop()
        r = randint(0, sol.k-1)
        _, pos = find_bestpos_on(sol, c, r)
        if not pos is None:
            insert(c, r, pos, ep)
            continue
        
        (_, pos), r = find_bestpos(sol, c)
        if not pos is None:
            insert(c, r, pos, ep)
            continue
        
        pos = sorted(find_replace_pos(sol, c))
        if pos:
            #print "Positions there:", pos
            #raw_input()
            _, r, p = pos[randint(0, min(5,len(pos)-1))]
            put_to_ep(remove_customer(sol, r, p), False)
            insert(c, r, p, ep)
            continue
        put_to_ep(c)
            
    if len(ep) > 0:
        print "Time out!"
        raise RuntimeError
    u.commit()
 
# MAIN COMMANDS
commands = set()
def command(func):
    """A command decorator - the decoratee should be a valid command."""
    commands.add(func.__name__)
    return func

# the CLUSTER command - mpi4py parallelism

def mpi_master(sol, comm, size, args):
    from mpi4py import MPI
    essencs = []
    # 'inifinite' values:
    my_k = sol.task.N
    my_dist = sol.task.dist.sum()
    stat = MPI.Status()
    time_to_die = time.time() + args.wall
    started = time.time()

    # initial jobs - creating initial solutions
    jobs = deque([('initial', k) for k in sort_keys.keys()])
    if len(jobs) < size+5:
        jobs.extend([('initial', 'by_random_ord')]*(size+5-len(jobs)))
    print "initial jobs are:", jobs
    for i in xrange(1, size):
        comm.send(jobs.popleft(), dest=i)
    
    # working loop
    workers = size-1    
    while workers > 0:
        resp = comm.recv(source=MPI.ANY_SOURCE, status=stat)
        if time.time() < time_to_die and len(jobs)>0:
            job = jobs.popleft()
            while len(jobs) > 2000 and job[0]=='killroute' and job[2][0] > my_k+1:
                job = jobs.popleft()
            comm.send(job, dest=stat.Get_source())
        else:
            comm.send(('done',), dest = stat.Get_source())
            workers -= 1
        
        if resp[0] == 'initial' or resp[1] == 'ok':
            essence = resp[2]
            if (my_k, my_dist) > essence[:2]:
                sol.set_essence(essence)
                sol.loghist()
                my_k = sol.k
                my_dist = sol.dist
                print "%.1f s, new best:" % (time.time()-started), sol.infoline()
            if essence[0] < my_k + 2:
                for x in xrange(essence[0]):
                    jobs.append(('killroute', x, essence))
        elif resp[0] == 'killroute' and resp[1] == 'failed':
            pass # TODO: no idea what failed... (not sent)
            
        if len(jobs) > 1000000 and time_to_die <> 0:
            print "We've got problems, %.1f s" % (time.time()-started)
            time_to_die = 0
    if len(jobs) == 0:
        print "The jobs went out, %.1f s" % (time.time()-started)
    sol.save('_clus')
    exit()

def mpi_worker(sol, comm, rank, args):
    # maybe start working immediately
    while True:
        orders = comm.recv(source=0)
        # print rank, "recieved orders:", orders
        if orders[0] == 'done':
            break
        elif orders[0] == 'initial':
            VrptwTask.sort_order = orders[1]
            build_first(sol)
            comm.send(('initial','ok', sol.get_essence()), dest=0)
        elif orders[0] == 'killroute':
            sol.set_essence(orders[2])
            try:
                op_route_min(sol, orders[1])
                comm.send(('killroute', 'ok', sol.get_essence()), dest=0)
            except RuntimeError:
                u.undo()
                comm.send(('killroute', 'failed', sol.get_essence()), dest=0)
        elif orders[0] == 'perturb':
            pass
        else:
            print rank, "orders not understood", orders
            
    print "Bye from worker", rank
    exit()

@command
def cluster(args):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    if size < 2:
        print "Sorry, only for > 1 process"
        exit()
    
    sol = VrptwSolution(VrptwTask(args.test))
    
    if rank == 0:
        mpi_master(sol, comm, size, args)
    else:
        mpi_worker(sol, comm, rank, args)
    
# POSTPROCESSING of old solutions

@command
def resume(args):
    """Load serialized solution, try to aliminate one or two routes."""
    # autodestruction timeout mechanism:
    data = dict(die=0)
    def die():
        data['die'] = 1
    from threading import Timer
    t = Timer(args.wall, die)
    t.start()
    sol = load_solution(args.test)
    # print_like_Czarnas(sol)
    tgt = pick_short_route(sol)
    # guarded tries
    try:
        op_route_min(sol, tgt, data=data)
    except:
        t.cancel()
        print "Failed removal of %d from %s, still: %d." % (tgt, sol.task.name, sol.k+1)
        exit(1)
    else:
        t.cancel()
    sol.check_full()
    sol.save('_rsm')
    print_like_Czarnas(sol)   
    print "Removed %d in %s, now: %s" % (tgt, sol.task.name, sol.infoline()), 

@command
def perturb(args):
    """Load a solution and perform some search around."""
    sol = load_solution(args.test)
    local_search(sol, op_tabu_single)
    print_like_Czarnas(sol)
    sol.save('_pert')
    
@command
def grout(args):
    """Postprocess a solution using the proprietary grout program."""
    import grout
    sol = load_solution(args.test)
    grout.DataLoader_load(sol.task.filename)
    dd = grout.DistanceDecreaser()
    dd.inflate(sol.flatten())
    dd.setMaxEpochs(60)
    best = grout.Solution()
    dd.simulatedAnnealing(best)
    sol.inflate(best.flatten())
    sol.save('_grout')
    print best.flatten()

# LOCAL SEARCH related techniques
    
def _optimize(test, op, wall, intvl):
    """An optimization funtion, which does not use argparse namespace."""
    sol = VrptwSolution(VrptwTask(test))
    build_first(sol)
    print_like_Czarnas(sol)
    print "Starting optimization for %d s, update every %s s." % (wall, intvl)
    time_to_die = time.time() + wall
    next_feedback = time.time() + intvl
    while time.time() < time_to_die:
        local_search(sol, operations[op], next_feedback, True)
        print_like_Czarnas(sol)
        next_feedback = time.time()+intvl
    print "Wall time reached for %s." % test.name
    sol.save()
    print(sol.mem)
    print_like_Czarnas(sol)
    return sol
    
@command
def optimize(args):
    """Perform optimization of a VRPTW instance according to the arguments."""
    sol = _optimize(args.test, args.op, args.wall, args.intvl)
    return sol

def _optimize_by_name(arg):
    # open the test filename (VrptwTask had problems with it)
    arg[0] = open(arg[0])
    return _optimize(*arg)
    
@command
def run_all(args):
    """As optimize, but runs all instances."""
    from glob import glob
    runs = args.runs or 1
    all_tasks = [[n, args.op, args.wall, args.intvl] 
                 for n in glob(args.glob) * args.runs]
    if args.multi:
        from multiprocessing import Pool
        p = Pool()    
        p.map(_optimize_by_name, all_tasks)
    else:
        map(_optimize_by_name, all_tasks)

def load_solution(f):
    """Unpickle solution from a stream."""
    solution_data = cPickle.load(f)
    print os.path.dirname(__file__), solution_data['filename']
    filename = os.path.join(os.path.dirname(__file__),
                            solution_data['filename'])
    print "Loading solution from:", filename
    sol = VrptwSolution(VrptwTask(open(filename)))
    sol.k, sol.dist = solution_data['val']
    sol.r = solution_data['routes']
    sol.mem = solution_data['mem']
    try:
        sol.history = solution_data['history']
    except: pass    
    if not sol.check_full():
        return None
    print "Solution loaded:", sol.infoline()
    return sol
    
@command
def load(args):
    """Loads a previously saved solution for analysis."""
    sol = load_solution(args.test)
    print_like_Czarnas(sol)
    print sol.mem
    print sol.get_signature()
    try:
        if len(sol.history):
            plot_history(sol)
        else:
            print "The solution has no history to plot"
    except ImportError:
        print "Plotting history impossible (missing GUI or matplotlib)"

@command
def export(args):
    """Create other formats for saved solution, like .vrp"""
    # TODO: real export; for now just print successors
    sol = load_solution(args.test)
    print "\n".join(str(s) for s in sol.get_successors())
    
# POOLCHAIN metaheuristic and friends_

def worker(sol, pools, operators, config):
    """The actual working process in a poolchain."""
    import Queue as q
    from multiprocessing import Queue
    proc_id, size, intvl, deadline = config
    print "Worker launched, id:", proc_id
    # disperse workers' random nubmer generators
    r.jumpahead(20000*proc_id)
    # disperse workers' feedback a bit (actually: random)
    next_feedback = time.time() + (proc_id+1)*intvl
    num_produced = 0
    # the list for measurement of fps etc.
    myfps = []
    
    while time.time() < deadline:
        # choose solution to work on this round
        try:
            # fish in the the pool
            new_essence = pools[1].get_nowait()
            sol.set_essence(new_essence)
            print "Worker", proc_id, "got job:", sol.infoline()
        except q.Empty:
            # if nothing to take - produce new one or keep current
            if num_produced < 5 or r.random() < 4.0/num_produced:
                order = r.choice(sort_keys.keys())
                VrptwTask.sort_order = order
                build_first(sol)
                print("Worker %d produced new: %s by %s" % 
                      (proc_id, sol.infoline(), order))
            # else: go on with current
        
        # run optimization 
        local_search(sol, operators[1], next_feedback, speed=myfps)
        next_feedback = time.time() + intvl*(size+1)
        
        # throw the solution back to the pool
        pools[2].put(sol.get_essence())
    # endwhile:
    # declare not to do any more output, send 'fps'
    pools[2].put((proc_id, sum(myfps)/len(myfps), 0))
    # print "Worker", proc_id, "should now finish."
    
@command
def poolchain(args):
    """Parallel optimization using a pool of workers and a chain of queues."""
    import Queue as q
    from multiprocessing import cpu_count, Process, Queue
    
    time_to_die = time.time()+args.wall
    # create own solution object (for test data being inherited)
    began = time.time()
    sol = VrptwSolution(VrptwTask(args.test))
    # setup the queues
    poison_pills = Queue()
    input_ = Queue()
    output = Queue()
    queues = [ poison_pills, input_, output ]
    oplist = [ None, operations[args.op], None ]
    
    # create and launch the workers
    num_workers = args.runs or cpu_count()
    workers = [ Process(
        target=worker, args=(sol, queues, oplist, 
          (i, num_workers, args.intvl, time_to_die)))
        for i in xrange(num_workers) ]
    map(Process.start, workers)
    
    # get a solution from the fastest worker (we have to service them...)
    print "Master waits for first solution..."
    essence = output.get()
    input_.put(essence)
    sol.set_essence(essence)
    print "Got first solution:", sol.infoline(), "after", time.time()-began 
    sol.loghist()

    # the discriminators of the solution circulation
    best_seen_k = essence[0]
    best_essncs = [essence]
    
    if best_seen_k == sol.task.best_k:
        print "Best known route count immediately:", time.time()-began
        sol.mem['best_k_found'] = time.time()-began
        if args.strive:
            time_to_die = time.time() + args.wall / 5.0
            print "Wall time reduced to:", time_to_die - time.time()
    
    # manage the pool for a while (now - simply feed them back)
    # ---- START OF MAIN LOOP ----
    while time.time() < time_to_die:
        essence = output.get()
        # drop solutions worse than best_seen_k+1
        if essence[0] <= best_seen_k+1:
            # -- check for route count record
            if best_seen_k > essence[0]:
                best_seen_k = essence[0]
                if best_seen_k == sol.task.best_k:
                    print "Best known route count reached:", time.time()-began
                    sol.mem['best_k_found'] = time.time()-began

                    if args.strive and time_to_die > time.time()+args.wall/5.0:
                        time_to_die = time.time()+args.wall/5.0
                        print "Remaining time reduced to:", args.wall/5.0
            # -- check against pool
            pos = bisect_left(best_essncs, essence)            
            if ( len(best_essncs)<15 
                or (pos < 15 and best_essncs[pos][:2] <> essence[:2]) ):
                # this solution is ok - pay it forward
                input_.put(essence)
                best_essncs.insert(pos, essence)
                if len(best_essncs) > 15:
                    best_essncs.pop()
                if pos == 0:
                    # new global best - remembering as a historical event
                    sol.set_essence(essence)
                    sol.loghist()            
            else:
                # throw in one of the elite solutions
                input_.put(r.choice(best_essncs))
        elif r.random() < 0.5:
            # if solution was bad (route count), maybe throw in old 
            input_.put(r.choice(best_essncs))
    # ---- END OF MAIN LOOP ----
    print "Wall time passed, after:", time.time()-began

    fpss = []
    workers_left = num_workers
    while workers_left > 0:
        k, dist, routes = output.get()
        if routes == 0:
            workers_left -= 1
            print "Worker's",k,"pill-box received", time.time()-began
            fpss.append(dist)
        else:
            if (k, dist) < sol.val():
                sol.set_essence((k, dist, routes))
            print 'got out from output: ', k, dist

    print "Staff is to join: so much are alive:"
    print map(Process.is_alive, workers)

    print input_.qsize(), 'solutions still in queue 1'
    try:
        while True:
            # print "Waiting for a solution"
            k, dist, routes = input_.get(timeout=0.3)
            if (k, dist) < sol.val():
                sol.set_essence((k, dist, routes))
            print 'got out: ', k, dist
    except q.Empty:
        pass
    
    try:
        output.get(timeout=0.1)
    except q.Empty:
        pass
    else:
        print "Possible rubbish in output"
    
    print "Best solution chosen. Saving.", time.time()-began
    sol.save('_pc') # suffix for poolchain
    print_like_Czarnas(sol)
    print "summary:", sol.task.name, "%d %.1f"%sol.val(), "%.1f %.1f"%sol.percentage(), 
    print "wall", args.wall, "workers", num_workers, "op", args.op, 'best_k',
    try:
        print "%.1f" % sol.mem['best_k_found'],
    except KeyError:
        print 'NO',
    print 'fps', "%.1f" % sum(fpss)
                
    #map(Process.join, workers)
    print "\nTotal time elapsed:", time.time()-began

@command
def initials(args):
    """Produce initial solutions in all available ways, and 10x randomly."""
    sol = VrptwSolution(VrptwTask(args.test))
    results = []
    best_order = None
    build_by_savings(sol)
    for k in sort_keys.keys():
        VrptwTask.sort_order = k
        build_first(sol)
        results.append((sol.percentage(), k, sol.k))
        if min(results) == results[-1]:
            best_order = k
    VrptwTask.sort_order = 'by_random_ord'
    for i in xrange(9):
        build_first(sol)
        results.append((sol.percentage(), 'by_random_ord', sol.k))
    rank = 1
    for prec, k, sol_k in sorted(results):
        print "%-20s %.2f %.2f  routes %d  rank %02d %s" % (
              (k+':',)+prec+(sol_k, rank, sol.task.name))
        rank += 1
    # best deterministic order, or the order given
    # will be used for saving
    if args.order is None:
        VrptwTask.sort_order = best_order
    else:
        VrptwTask.sort_order = args.order
    build_first(sol)
    sol.save("_init")

def mksol(name = 'c101'):
    """Produce a solution with a task for the given name (shorthand)."""
    import os.path
    sol = None
    for cand in [name, 'solomons/%s'%name, 'hombergers/%s'%name, 
                 'solomons/%s.txt'%name, 'hombergers/%s.txt'%name]:
        if os.path.exists(cand):
            sol = VrptwSolution(VrptwTask(cand))
    return sol
    
def get_argument_parser():
    """Create and configure an argument parser.
    Used by main function; may be used for programmatic access."""
    try:
        from argparse import ArgumentParser, Action
        parser = ArgumentParser(
            description="Optimizing VRPTW instances with some heuristics")
            
        parser.add_argument(
            "test", type=file, nargs='?', default=os.path.join(
                os.path.dirname(__file__), 'hombergers','rc210_1.txt'),
            help="the test instance: txt format as by M. Solomon")
        parser.add_argument(
            "command", choices=commands, nargs="?", default="poolchain",
            help="main command to execute (when omitted: poolchain)")
            
        parser.add_argument(
            "--op", choices=operations.keys(), nargs="?",
            default="op_fight_shortest", help="neighbourhood operator to use")
        
        parser.add_argument(
            "--runs", "-n", type=int, default=0,
            help="repeat (e.g. optimization) n times, or use n processes")
        parser.add_argument(
            "--glob", "-g", default="hombergers/*.txt",
            help="glob expression for run_all, defaults to all H")
            
        parser.add_argument(
            "--wall", "-w", type=int, default=600,
            help="approximate walltime (real) in seconds")
        parser.add_argument(
            "--intvl", type=int, default=10,
            help="approximate refresh rate (delay between messages)")
        parser.add_argument(
            "--strive", action="store_true",
            help="run for best known route count, and then only short")
                    
        parser.add_argument(
            "--multi", "-p", action="store_true",
            help="use multiprocessing for parallelism e.g. with run_all")
        parser.add_argument(
            "--prof", action="store_true",
            help="profile the code (don't do that), 10x slower")
    
        class OptionAction(Action):
            """A dispatching action for option parser - global configs"""
            def __call__(self, parser, namespace, values,
                         option_string=None):
                if option_string in ['-o', '--output']:
                    VrptwSolution.outdir = values
                elif option_string == '--order':
                    VrptwTask.sort_order = values
                    namespace.order = values
                elif option_string in ['-s', '--seed']:
                    global r_seed
                    r_seed = int(values)
                    r.seed(r_seed)

        parser.add_argument(
            "--seed", "-s", action=OptionAction,
            help="Set a custom RNG seed")    
                
        parser.add_argument(
            "--output", "-o", default="output", action=OptionAction,
            help="output directory for saving solutions")
        
        parser.add_argument(
            "--order", action=OptionAction, choices=sort_keys.keys(),
            help="choose specific order for initial customers")
        
        return parser
    except ImportError:
        print "Install argparse module"
        raise

def main(can_profile = False):
    """Entry point when this module is ran at top-level."""
    args = get_argument_parser().parse_args()
    if can_profile and args.prof:
        import cProfile
        args.test.close()
        cProfile.run('main()', 'profile.bin')
        return
    # execute the selected command
    globals()[args.command](args)

if __name__ == '__main__':
    main(True)
    
