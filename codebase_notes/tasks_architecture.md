# What is up with tasks in AthenaK?


## Motivation for the Task Architecture

AthenaK doesn't have one hardcoded "Do A, then B, then C" type update loop. Each cycle is broken into a handful of named task lists,  each of which is a dependency graph of small operations (flux computations, MPI sends, boundary condition, etc etc). The graph is walked repeatedly, running whatever's "ready" until everything in the graph is done. 

This architecture has two advantages:
1. The graph shape can differ for different problem (Hydro, MHD, MHD+Radiation all have different things to handle)
2. Waiting on one thing like an MPI Boundary recieve for physics A doesn't block physics B from happening. 

## Implementation in AthenaK

The bulk of the work happens in the [`task_list.hpp`](../athenak/src/tasklist/task_list.hpp). `TaskID` is a 64-bit bitmask, each task added to a list claims the next bit as its own ID. A 'Dependency" is also a bitmask, the OR of the bits of everything it must wait for. 

[`Task`](../athenak/src/tasklist/task_list.hpp#L88-110) wraps a function, it's ID and dependency mask. Each Task takes inputs of `(Driver*, int stage)` and nothing else. 

[`Tasklist::AddTask`](../athenak/src/tasklist/task_list.hpp#L178-185): the member-function overload. For some physics object, a task is just a bound member function. 

The Driver in [`driver.cpp`](../athenak/src/driver/driver.cpp#L438-455) calls the function [`TaskList::DoAvailable`](../athenak/src/tasklist/task_list.hpp#L146-159) repeatedly till the whole list is complete. 

Each [`MeshBlockPacp`](../athenak/src/mesh/meshblock_pack.hpp#L86) owns a TaskList. 

## How this impacts scope of variables. 

A `Task`'s signature is locked to `(Driver*, int)`, so there is no way to pass problem data through the tas-calling mechanism itself. All the actual state a task reads/writes has to live as member variables of the physics-module object the task is bound to.

1. A chain of tasks communicates implicitly thorugh shared object state, not function arguments. `Fluxes` writes uflux, `SendFlux/RecvFlux`, exchange it over MPI, `RKUpdate` reads it back, and all of this happens through the same `Hydro` instance's member arrays. Not by passing anything between task calls.
2. For most variables, the lifetime is the same as a module lifetime, not a tas-list-run lifetime. 
3. The framework enforces ordering. If the dependency graph is wrong, then you can and will run into issues with race conditions. 