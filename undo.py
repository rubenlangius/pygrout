# undo handlers

def undo_ins(list_, idx):
    list_.pop(idx)

def undo_pop(list_, idx, val):
    list_.insert(idx, val)

def undo_set(list_, idx, val):
    list_[idx] = val

def undo_atr(obj, atr, val):
    setattr(obj, atr, val)

def undo_add(list_, idx, val):
    list_[idx] -= val

def undo_ada(obj, atr, val):
    setattr(obj, atr, getattr(obj, atr) - val)

# undo elements type:
#
U_ELEM_IN, U_ELEM_OUT, U_ELEM_MOD, U_ATTRIB, U_ADD, U_ADA, U_CHECKPOINT = range(7)

# undo mapping
handlers = [ undo_ins, undo_pop, undo_set, setattr, undo_add, undo_ada ]

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

    def atr(self, obj, atr, val):
        """Change an object's attribute."""
        data = getattr(obj, atr)
        self.actions.append( (U_ATTRIB, (obj, atr, data)) )
        setattr(obj, atr, val)
        return val

    def add(self, list_, idx, value):
        """Inplace add something to list element."""
        self.actions.append( (U_ADD, (list_, idx, value)) )
        list_[idx] += value

    def ada(self, obj, atr, val):
        """Inplace add to object's attribute."""
        data = getattr(obj, atr)
        self.actions.append( (U_ADA, (obj, atr, val)) )
        setattr(obj, atr, val+data)
        return val+data

    def commit(self):
        """Forget all undo information."""
        self.actions = []
        self.point = 0

    def undo(self, checkpoint = None):
        """Reverse all operation performed through this stack, or up to a checkpoint."""
        assert checkpoint <= self.point, 'Undo to invalid checkpoint'
        while len(self.actions):
            tag, args = self.actions.pop()
            if tag == U_CHECKPOINT:
                if args == checkpoint:
                    self.point = checkpoint-1
                    break
            else:
                # print tag, args
                handlers[tag](*args)

    def undo_last(self):
        """Rollback actions to last checkpoint."""
        assert self.point > 0, 'No actions to undo'
        self.undo(self.point)

class TestUndoStack(object):
    """Unit test class for py.test"""
    def setup_class(self):
        """Create the UndoStack used with every test and an example list."""
        self.u = UndoStack()
        self.l_orig = [7, 'dolorem', 4, None, 5.3]
        self.l = self.l_orig[:]

    def setup_method(self, method):
        """Restore the example list, not needed if tests pass, undo does it."""
        # self.l = self.l_orig[:]

    def test_ins(self):
        """Undoing an insertion."""
        self.u.ins(self.l, 0, 2)
        expected = [2]+self.l_orig
        assert self.l == expected
        self.u.undo()
        assert self.l == self.l_orig

    def test_pop(self):
        out = self.u.pop(self.l, 2)
        assert out == 4
        self.u.undo()
        assert self.l == self.l_orig

    def test_set(self):
        self.u.set(self.l, 1, 'ipsum')
        assert self.l[1] == 'ipsum'
        self.u.undo()
        assert self.l == self.l_orig

    def test_sequence(self):
        self.u.pop(self.l, 3)
        self.u.ins(self.l, 3, 123)
        tag = self.u.checkpoint()
        l_on_check = self.l[:]
        self.u.set(self.l, 0, 0)
        self.u.pop(self.l, 0)
        self.u.undo(tag)
        assert l_on_check == self.l
        self.u.undo()
        assert self.l == self.l_orig

    def test_atr(self):
        self.color = 'red'
        self.u.atr(self, 'color', 'blue')
        assert self.color == 'blue'
        self.u.undo()
        assert self.color == 'red'
