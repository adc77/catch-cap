from __future__ import annotations

import abc

from ..types import JudgeVerdict


class BaseJudge(abc.ABC):
    """Interface for judge models."""

    @abc.abstractmethod
    async def evaluate(
        self,
        query: str,
        response_text: str,
        reference_text: str,
    ) -> JudgeVerdict:
        """Evaluate the response relative to reference evidence."""


