Angles for inventing a constraint the search has not tried. These are about how
to search, not about any particular problem, so none of them names a technique —
that is your job, given what this problem turns out to reward.

- **Borrow from another domain.** How would a database engine, a compiler
  backend, a network stack, a filesystem or a game engine attack this? Add the
  constraint that names the borrowed idea.

- **Invert a shared assumption.** Name something every branch so far takes for
  granted, and add a constraint forbidding it.

- **Trade a resource.** Add a constraint that spends more memory, more
  preprocessing or more code size to buy less of whatever is scored.

- **Subtract instead of adding.** Take the best branch and add a constraint that
  removes a mechanism rather than introducing one. The fastest code is often the
  code that stopped doing something.

- **Mine a failure.** Read a candidate that scored badly. Add a constraint
  isolating the one idea in it that was worth keeping.

- **Go to an extreme.** Find a quantity every branch treats as moderate and
  constrain it to an extreme — very large, very small, exactly one, none at all.

- **Re-read what is allowed.** Go back to the problem statement and find
  something it permits that no branch has used, then constrain toward it.

- **Follow a secondary metric.** Look at the metrics that are recorded but not
  scored. If one of them moves with the score, constrain toward improving it
  directly.

- **Specialise for the common case.** Add a constraint that handles the typical
  input on a fast path and everything else on a slow one.

- **Do it ahead of time.** Add a constraint that moves work out of the measured
  region — precompute it, table it, or arrange the data so the work disappears.

- **Change the representation.** Add a constraint on the shape of the data rather
  than on the algorithm over it. A different layout often makes the old algorithm
  the wrong one.

- **Question the obvious step.** Name the one thing every branch does because it
  seems necessary, and add a constraint that does without it.
