"""Generator implementations.

`AgentGenerator` is deliberately not re-exported here: importing it pulls
mini-swe-agent and its dependency tree, and the core is meant to import cleanly
without the `agent` extra installed. Import it directly:

    from optiverse.generators.agent import AgentGenerator
"""
