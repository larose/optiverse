"""Tests for waiting the delay the provider actually stated.

Requires the `agent` extra. Nothing sleeps for real: the clock is injected.

`_query` is mini-swe-agent's name for the hook being overridden, so exercising it
by that name is the point rather than an oversight.
"""

# pyright: reportPrivateUsage=false

import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

import httpx
import litellm
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

from optiverse.generators._mini_swe_agent import (
    MAXIMUM_RATE_LIMIT_WAIT_SECONDS,
    MAXIMUM_RATE_LIMIT_WAITS,
    RATE_LIMIT_MARGIN_SECONDS,
    RateLimitAwareModel,
    suggested_delay,
)

# Trimmed from a real Gemini 429: the delay is in the body, not a header.
GEMINI_BODY = """litellm.RateLimitError: geminiException - {
  "error": {
    "code": 429,
    "message": "You exceeded your current quota. Please retry in 47.006079238s.",
    "status": "RESOURCE_EXHAUSTED",
    "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                 "retryDelay": "47s"}]
  }
}"""

ASKED_FOR = 47.0 + RATE_LIMIT_MARGIN_SECONDS


def rate_limited(
    message: str = GEMINI_BODY, headers: Optional[Dict[str, str]] = None
) -> litellm.exceptions.RateLimitError:
    return litellm.exceptions.RateLimitError(
        message=message,
        llm_provider="gemini",
        model="gemma-4-31b-it",
        response=httpx.Response(429, headers=headers or {}),
    )


class SuggestedDelayTest(unittest.TestCase):
    def test_reads_the_retry_after_header(self) -> None:
        self.assertEqual(
            suggested_delay(rate_limited(headers={"retry-after": "12"})), 12.0
        )

    def test_reads_the_delay_gemini_puts_in_the_body(self) -> None:
        """litellm's own helper only reads headers, so this form is on us."""
        self.assertEqual(suggested_delay(rate_limited()), 47.0)

    def test_reads_the_prose_form(self) -> None:
        self.assertEqual(suggested_delay(rate_limited("Please retry in 8.5s.")), 8.5)

    def test_says_nothing_when_the_provider_did_not(self) -> None:
        self.assertIsNone(suggested_delay(rate_limited("too many requests")))

    def test_ignores_an_unparseable_header(self) -> None:
        """The HTTP-date form falls through to the body rather than crashing."""
        headers = {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}

        self.assertEqual(suggested_delay(rate_limited(headers=headers)), 47.0)


class RetryLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.waits: List[float] = []
        self.model = RateLimitAwareModel(
            model_name="gemini/gemma-4-31b-it", sleep=self.waits.append
        )

    def query(self, *responses: Any) -> Any:
        """Run one query against a scripted sequence of API outcomes."""
        with mock.patch.object(LitellmTextbasedModel, "_query", side_effect=responses):
            return self.model._query([])

    def test_waits_exactly_what_the_provider_asked(self) -> None:
        self.assertEqual(self.query(rate_limited(), "answered"), "answered")
        self.assertEqual(self.waits, [ASKED_FOR])

    def test_does_not_climb_a_ladder_of_doomed_requests(self) -> None:
        """The behaviour being replaced retried at 4s, 8s and 16s against a
        stated 47 — three requests that could not have succeeded."""
        self.query(rate_limited(), rate_limited(), rate_limited(), "answered")

        self.assertEqual(self.waits, [ASKED_FOR, ASKED_FOR, ASKED_FOR])

    def test_gives_up_once_the_provider_keeps_saying_no(self) -> None:
        refusals = [rate_limited()] * (MAXIMUM_RATE_LIMIT_WAITS + 1)

        with self.assertRaises(litellm.exceptions.RateLimitError):
            self.query(*refusals)

        self.assertEqual(len(self.waits), MAXIMUM_RATE_LIMIT_WAITS)

    def test_does_not_wait_out_an_exhausted_quota(self) -> None:
        """Minutes of waiting means the daily allowance is gone, not a burst."""
        seconds = MAXIMUM_RATE_LIMIT_WAIT_SECONDS + 1

        with self.assertRaises(litellm.exceptions.RateLimitError):
            self.query(rate_limited(f"Please retry in {seconds}s."), "answered")

        self.assertEqual(self.waits, [])

    def test_does_not_retry_blind(self) -> None:
        with self.assertRaises(litellm.exceptions.RateLimitError):
            self.query(rate_limited("too many requests"), "answered")

        self.assertEqual(self.waits, [])

    def test_leaves_other_failures_to_the_caller(self) -> None:
        """Only rate limits are handled here; everything else is mini-swe-agent's
        retry to deal with."""
        with self.assertRaises(litellm.exceptions.APIConnectionError):
            self.query(
                litellm.exceptions.APIConnectionError(
                    message="dropped", llm_provider="gemini", model="gemma-4-31b-it"
                ),
                "answered",
            )

        self.assertEqual(self.waits, [])


class AbortListTest(unittest.TestCase):
    def test_rate_limits_are_not_retried_again_upstream(self) -> None:
        """Otherwise mini-swe-agent's own loop re-tries what this class already
        gave up on, multiplying into dozens of requests."""
        self.assertIn(
            litellm.exceptions.RateLimitError, RateLimitAwareModel.abort_exceptions
        )


if __name__ == "__main__":
    unittest.main()
