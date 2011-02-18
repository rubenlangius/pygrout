#!/usr/bin/env python

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
    
def main():
    from os import getcwd
    from subprocess import Popen, PIPE
    
    job_name = smart_input('Job name', 'output/.pbs/jobname',['poolchain'])
    pbs_opts = smart_input('PBS options', 'output/.pbs/options', 
        ['-l nodes=1:nehalem -l walltime='])
    tasks = smart_input('Tasks [glob pattern]', 'output/.pbs/tasks', 
        ['solomons/????.txt solomons/?????.txt', 'hombergers/*_2??.txt'])
    pygrout_opts = smart_input('pygrout options', 'output/.pbs/pygroupts', 
        ['--strive --wall ', '--wall '])
         
    script = """
    cd %s
    for i in %s; do
    ./pygrout %s $i
    done
    """ % (getcwd(), tasks, pygrout_opts)
    
    command = 'qsub %s -N %s' % (pbs_opts, job_name)
    
    print "About to pipe: \n%s\n to the command: \n%s\n\nPress Enter"
    raw_input()
    
    pipe = Popen(command, shell=True, stdin=PIPE).stdin
    pipe.write(script)
    pipe.close()
    
if __name__ == '__main__':
    main()
