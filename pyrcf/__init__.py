"""A Python Robot Control Framework for quickly prototyping control algorithms for different robot
embodiments.

Primarily, this library provides an implementation of a typical control loop (via a
`MinimalCtrlLoop` (extended from `SimpleManagedCtrlLoop`) class), and defines interfaces for the
components in a control loop that can be used directly in these control loop implementations. It
also provides utility and debugging tools that will be useful for developing controllers and
planners for different robots. This package also provides implementations of basic controllers
and planners.

In the long run, this package will also provide implementations of popular motion planners and
controllers from literature and using existing libraries.
"""
