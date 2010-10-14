        
# undo elements type:
# edge inserted, edge removed
U_ELEM_IN, U_ELEM_OUT, U_ELEM_MOD, U_CHECKPOINT = range(4)

class UndoStack(object):
    """Holds description of a sequence of operations, possibly separated by checkpoints."""
    def __init__(self):
        """Construct empty undo stack."""
        self.actions = []
        
    def set(self, list_, idx, value):
        """Sets a list element to new value, returns it for possible chaining."""
        self.actions.append( (U_ELEM_MOD, list_, idx, list_[idx]) )
        list_[idx] = value
        return value

    def pop(self, list_, idx):
        """Removes a list element and returns its value."""
        data = list_.pop(idx)
        self.actions.append( (U_ELEM_OUT, list_, idx, data) )
        return data

    def ins(self, list_, idx, value):
        """Inserts the value at a specific index in list and returns for chaining."""
        self.actions.append( (U_ELEM_IN, list_, idx, value) ) # value is not used, drop it?
        list_.insert(idx, value)
        return value

    def commit(self):
        """Forget all undo information."""
        self.actions = []

    def undo(self, checkpoint = None):
        """Reverse all operation performed through this stack, or up to a checkpoint."""
        while len(self.actions):
            op = self.actions.pop()
            print op
            
