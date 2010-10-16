# undo handlers

def undo_ins(list_, idx):
    list_.pop(idx)

def undo_pop(list_, idx, val):
    list_.insert(idx, val)

def undo_set(list_, idx, val):
    list_[idx] = val
        
# undo elements type:
# edge inserted, edge removed
U_ELEM_IN, U_ELEM_OUT, U_ELEM_MOD, U_CHECKPOINT = range(4)

# undo mapping
handlers = [ undo_ins, undo_pop, undo_set ]

class UndoStack(object):
    """Holds description of a sequence of operations, possibly separated by checkpoints."""
    def __init__(self):
        """Construct empty undo stack."""
        self.commit()

    def ins(self, list_, idx, value):
        """Inserts the value at a specific index in list and returns for chaining."""
        self.actions.append( (U_ELEM_IN, (list_, idx)) ) # value not needed
        list_.insert(idx, value)
        return value

    def pop(self, list_, idx):
        """Removes a list element and returns its value."""
        data = list_.pop(idx)
        self.actions.append( (U_ELEM_OUT, (list_, idx, data)) )
        return data

    def set(self, list_, idx, value):
        """Sets a list element to new value, returns it for possible chaining."""
        self.actions.append( (U_ELEM_MOD, (list_, idx, list_[idx])) )
        list_[idx] = value
        return value

    def checkpoint(self):
        """Marks current state and returns the marker."""
        self.point += 1
        self.actions.append( (U_CHECKPOINT, self.point) )
        return self.point
    
    def commit(self):
        """Forget all undo information."""
        self.actions = []
        self.point = 0

    def undo(self, checkpoint = None):
        """Reverse all operation performed through this stack, or up to a checkpoint."""
        while len(self.actions):
            tag, args = self.actions.pop()
            if tag == U_CHECKPOINT and args == checkpoint:
                break
            else:
                print tag, args
                handlers[tag](*args)
                
            
if __name__=='__main__':
    """Later to appear here (or in a different file): unit tests."""
    u = UndoStack()
    l = []
    u.ins(l, 0, 2)
    print l
    u.undo()
    print l
