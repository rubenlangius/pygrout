#!/usr/bin/env python

import re 
import sys

def smart_input(prompt, history=None, suggestions=[], info=None):
    from collections import deque
        
    def ensure_file(f):
        import os
        if os.path.exists(f):
            return f
        if not os.path.exists(os.path.dirname(f)):
            os.makedirs(os.path.dirname(f))
        open(f, 'w')
        return f
    
    def inner_loop(default, suggestions, history):
        while True:
            question = "%s (%s): " % (prompt, default)
            ans = raw_input(question)

            if ans == '':
                return default
                
            if ans == ' ':
                if not len(history):                
                    print "No history provided..."
                    continue
                if len(history) == 1 and default == history[0]:
                    print "No other history items."
                history.rotate(-1)
                default = history[0]
                continue

            if ans == '  ':
                if not len(history):                
                    print "No history provided..."
                    continue
                if len(history) == 1 and default == history[0]:
                    print "No other history items."
                history.rotate(1)
                default = history[0]
                continue
                
            if ans == '`':
                if not len(suggestions):
                    print "No suggestions provided..."
                    continue
                if len(suggestions) == 1 and default == suggestions[0]:
                    print "No other suggestions."
                suggestions.rotate(-1)
                default = suggestions[0]
                continue
            
            if ans == '``':
                if not len(suggestions):
                    print "No suggestions provided..."
                    continue
                if len(suggestions) == 1 and default == suggestions[0]:
                    print "No other suggestions."
                suggestions.rotate(1)
                default = suggestions[0]
                continue
            
            if ans[0] == '+':
                return default + ans[1:]
                
            if ans == '~':
                if default == history[0]:
                    print "Removing '%s' from history." % default
                    history.popleft()
                    default = ''
                    continue
                
            # all other cases:
            return ans
            
    suggestions = deque(suggestions)    
    default = suggestions[0] if len(suggestions) else ''
    
    if history is None:
        hist = deque()
    else:
        hist = deque(map(str.strip, open(ensure_file(history))))
        
    # print hist
    
    result = inner_loop(default, suggestions, hist)

    if not history is None:
        if result in hist:
            hist.remove(result)
        hist.appendleft(result)
        open(history,'w').write("\n".join(hist))
        
    return result
    
def run_job(task_portion, wall, auto = False, extra = ''):
    from subprocess import Popen, PIPE
    from os import getcwd

    # some task preparation (if it wasn't a file, but a test name?)

    script = """
    cd %s
    pwd
    date """ % getcwd() + "".join("""
    ./pygrout.py %s --wall %d %s
    date """ % (extra, wall, task) for task in task_portion)
    
    # prepare jobname
    jobname = re.sub('.txt|hombergers/|solomons/', '', 'vrptw_' + task_portion[0])

    command = 'qsub -l nodes=1:nehalem -l walltime=%d -N %s' % (
        (wall+60)*len(task_portion), jobname)
    
    if not auto:
	    print "About to pipe: \n%s\n to the command: \n%s\n\nPress Enter" % (
		script, command)
	    raw_input()
    
    output, errors = Popen(command, shell=True, stdin=PIPE, 
                           stdout=PIPE, stderr=PIPE).communicate(script)
    print "Process returned", repr((output, errors))
    return command, script, jobname, output.strip()

def main():
    import datetime
    # job_name = smart_input('Job name', 'output/.pbs/jobname',['poolchain']) 
    # pbs_opts = smart_input('PBS options', 'output/.pbs/options', 
    #     ['-l nodes=1:nehalem -l walltime=20000'])
    # tasks = smart_input('Tasks [glob pattern]', 'output/.pbs/tasks', 
    #     ['solomons/', 'hombergers/', 'hombergers/*_2??.txt'])
    # pygrout_opts = smart_input('pygrout options', 'output/.pbs/pygroupts', 
    #     ['--strive --wall 600', '--wall '])
    
    if len(sys.argv) < 2:
        print "No arguments (tasks) provided"
        return
    
    tasks = sys.argv[1:]
    wall = int(smart_input('Enter wall time (per task)', suggestions=[2000]))
    total = len(tasks)*(wall+60)
    print "There are %d tasks, which makes %s s (%02d:%02d:%02d) total." % (
        len(tasks), total, total/3600, total%3600/60, total%60)
    print "A single task is %02d:%02d" % (wall/60+1, wall%60)
    per_job = int(smart_input('How many task per job', suggestions=[20]))
    total = per_job*(wall+60)
    print "A single job will run %02d:%02d:%02d" % (total/3600,
        total%3600/60, total%60)
    extra = smart_input('Extra args for pygrout', suggestions=[''])
    auto = raw_input('Confirm single jobs (Y/n)?')=='n'
    jobs = []
    
    for i in xrange(0, len(tasks), per_job):
        jobs.append(run_job(tasks[i:i+per_job], wall, auto, extra))

    log = "\n".join("""
Command: %s
Script: %s
Job name: %s
Job id: %s
""" % tup for tup in jobs)
    open('output/%s.log.txt' % datetime.datetime.now().isoformat(), 'w').write(log)
    
if __name__ == '__main__':
    main()
