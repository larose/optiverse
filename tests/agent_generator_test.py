"""Tests for how the agent generator is configured.

Requires the `agent` extra, which `make init` installs. Nothing here reaches the
network or needs a key: the model object is built, not queried.
"""

import os
import unittest
from unittest import mock

from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

from optiverse.generators.agent import AgentGenerator, AgentLimits


class AgentLimitsTest(unittest.TestCase):
    def test_cost_is_not_capped_by_default(self) -> None:
        """mini-swe-agent reads 0 as "no limit"; cost is a metric, not a bound."""
        self.assertEqual(AgentLimits().cost_limit, 0.0)

    def test_steps_and_wall_time_are_capped(self) -> None:
        limits = AgentLimits()

        self.assertGreater(limits.step_limit, 0)
        self.assertGreater(limits.wall_time_limit_seconds, 0)


class FromEnvTest(unittest.TestCase):
    def test_reads_the_model_variable(self) -> None:
        with mock.patch.dict(os.environ, {"OPTIVERSE_MODEL": "gemini/some-model"}):
            generator = AgentGenerator.from_env()

        self.assertEqual(generator.build_model().config.model_name, "gemini/some-model")

    def test_requires_the_model_variable(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as raised:
                AgentGenerator.from_env()

        self.assertIn("OPTIVERSE_MODEL", str(raised.exception))


class BuildModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = AgentGenerator(model_name="gemini/some-model").build_model()

    def test_parses_the_text_format_the_prompt_asks_for(self) -> None:
        """The tool-calling class would reject the fenced replies we ask for."""
        self.assertIsInstance(self.model, LitellmTextbasedModel)
        self.assertIn("mswea_bash_command", self.model.config.action_regex)

    def test_survives_a_model_litellm_cannot_price(self) -> None:
        """Otherwise an unpriced model raises and costs the whole iteration."""
        self.assertEqual(self.model.config.cost_tracking, "ignore_errors")


if __name__ == "__main__":
    unittest.main()
