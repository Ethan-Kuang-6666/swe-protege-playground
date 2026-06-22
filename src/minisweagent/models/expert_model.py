import litellm
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.retry import retry
from minisweagent.models import GLOBAL_MODEL_STATS
import logging
import os
import time
logger = logging.getLogger("litellm_model")

class ExpertModel(LitellmModel):
    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your expert API key."
            raise e
        
        
    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(messages, **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

