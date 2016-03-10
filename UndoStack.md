# Introduction #

This page describes the ideas and implementations of the `UndoStack` class, which provides a one-direction undo mechanism (i.e. no "redo").

Below is a list of its methods with short examples to illustrate their effect. Finally there is a short explanation about possibilities that are offered by such a tool, and how were they used in `pygrout`.

In short, the `UndoStack` is an intermediary, if we modify a list or and object _through_ it, it remembers how to undo this change.

A later call to `undo()` can restore the target (list or object) to the state before the operation.

By using the method `commit()` we can make the changes permanent, that is not undoable anymore. ,

Finally, there is a way to mark certain states with a checkpoint - calling `checkpoint()` puts a marker onto the stack and then calling `undo_last()` will only restore changes to the previous checkpoint (and remove it).

The `checkpoint()` call returns the checkpoint's number, which can be passed to `undo(checkpt)` as an argument, which causes undoing changes to a specified checkpoint.

# Available methods #

## List modification ##

Three most basic modyfying operations on a list are - insertion, removal and substitution (assignment) of an element.

Thus, we have:

```
u = UndoStack()
l = range(5)

# l.insert(0, 8)
u.ins(l, 0, 8)

# l == [8, 0, 1, 2, 3, 4]
u.undo()
# l == range(5)

# l.pop(3)
u.pop(l, 3)

# l == [0, 1, 2, 4]
u.undo()
# l == range(5)

# l[2] = 7
u.set(l, 2, 7)

# l == [0, 1, 7, 3, 4]
u.undo()
# l == range(5)
```

There is also an add method, which provides two simultaneous operations - subscript and in-place addition, like:

```
# l[4] += 5
u.add(l, 4, 5)
# l == [0, 1, 2, 3, 9]
```

## Object attribute write access ##

Fields of objects can be modified with undo functionality, by using two methods - `atr` and `ada`, which features in-place addition for the attribute.

```
class Liberal(object):
    pass
lib = Liberal()

# lib.x = 4
u.atr(lib, 'x', 4) ##### important: attrib name must be given as a string

# lib.x += 3
u.ada(lib, 'x', 3) #### see: atr() - atribute name in quotes!

# lib.x == 7
```


# Possibilities and usage #

The main advantage of undo used in automatical context (outside human interaction) is to employ heuristics which "try". After the rule of "better asking forgiveness", the target - like e.g. a solution to VRPTW, is modified directly. When the change is cancelled, the steps can be reversed to return to a previous, correct state.

Otherwise, one would have to copy the whole structure and modify the copy, which would be discarded, if the changes had to be dropped. The thereshold from which usage of an `UndoStack` breaks even is when a typical change involves less than about a half of the structure (because the use of this mechanism incurs a factor of 2~3 for every operation).

So, the actual complexity is as follows:
n - size of the structure
k - number of modifications
p - probability of undo, between 0 and 1

for full copying:
T(n, k) = n + k

cost of full copy (n) + cost of the modifications (k)

for undo rollback:
T(n, k) = k (2 + p)

Cost of modifications (k) times (2 + average undo probability)

With these assumptions, undo is better when the average number of modifications is less than n / (1 + p)

k < n / (1 + p)

So, if the probability of undo == 1 (every time), k must be less than half n.

Conversly, if the probability of undo == 0 (never), k must be less than n, to justify the usage of undo. This is a possible catch, because -- depending on the algorithm -- the number of modifications might even exceed n in some situations. However, most important is the average k, and not individual outliers.