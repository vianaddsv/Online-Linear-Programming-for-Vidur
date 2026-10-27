from typing import List, Tuple

import numpy as np

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class PowerOfTwoGlobalScheduler(BaseGlobalScheduler):
    """
    Power-of-two choices global scheduler: samples two replicas uniformly at random
    and routes the request to the one with fewer outstanding requests (same measure as LOR).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        seed = self._config.cluster_config.global_scheduler_config.seed
        self._rng = np.random.default_rng(seed)

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        pending_requests_map = {
            replica_scheduler.replica_id: replica_scheduler.num_pending_requests
            for replica_scheduler in self._replica_schedulers.values()
        }
        replica_ids = list(pending_requests_map.keys())

        while self._request_queue:
            request = self._request_queue.pop(0)
            if len(replica_ids) == 1:
                replica_id = replica_ids[0]
            else:
                first, second = (
                    int(r) for r in self._rng.choice(replica_ids, size=2, replace=False)
                )
                replica_id = (
                    first
                    if pending_requests_map[first] <= pending_requests_map[second]
                    else second
                )
            pending_requests_map[replica_id] += 1
            request_mapping.append((replica_id, request))

        return request_mapping
