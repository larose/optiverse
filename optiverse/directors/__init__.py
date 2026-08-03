"""Director implementations.

`AgentDirector` is deliberately not re-exported here, for the same reason
`AgentGenerator` is not: importing it pulls mini-swe-agent and its dependency
tree, and the core is meant to import cleanly without the `agent` extra
installed. Import it directly:

    from optiverse.directors.agent import AgentDirector
"""
