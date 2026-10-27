import math
import random

from vidur.config import PoissonRateShiftRequestIntervalGeneratorConfig
from vidur.request_generator.base_request_interval_generator import (
    BaseRequestIntervalGenerator,
)


class PoissonRateShiftRequestIntervalGenerator(BaseRequestIntervalGenerator):
    """
    Poisson arrivals whose rate switches from `qps` to `qps_after_shift` at `shift_time`.
    """

    def __init__(self, config: PoissonRateShiftRequestIntervalGeneratorConfig):
        super().__init__(config)

        self._current_time = 0.0

    @staticmethod
    def _sample_interval(qps: float) -> float:
        # same truncation as PoissonRequestIntervalGenerator
        interval = -math.log(1.0 - random.random()) / qps
        return min(interval, 3.0 / qps)

    def get_next_inter_request_time(self) -> float:
        shift_time = self.config.shift_time

        if self._current_time < shift_time:
            next_time = self._current_time + self._sample_interval(self.config.qps)
            if next_time > shift_time:
                # memoryless arrivals: restart the clock at the shift with the new rate
                next_time = shift_time + self._sample_interval(
                    self.config.qps_after_shift
                )
        else:
            next_time = self._current_time + self._sample_interval(
                self.config.qps_after_shift
            )

        interval = next_time - self._current_time
        self._current_time = next_time
        return interval
