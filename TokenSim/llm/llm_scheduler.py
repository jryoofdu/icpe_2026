from __future__ import annotations
from abc import ABC, abstractmethod


from TokenSim.llm.llm_comm import SwapMetadata
from TokenSim.llm.llm_request import Request, RequestStatus
from TokenSim.block.block_manager import BlockManager
from TokenSim.config.config import CacheConfig

check_req_id = []


class LLMScheduler(ABC):
    def __init__(self):
        self.running: list[Request] = []
        self.waiting: list[Request] = []
        self.swapped: list[Request] = []

    def add_requests(self, requests: list[Request]):
        self.waiting.extend(requests)

    @abstractmethod
    def schedule(self) -> tuple[list[Request], list[Request], SwapMetadata]:
        pass

    def update(self, requests: list[Request]):
        # update block manager
        for req in requests:
            if req.is_done:
                self.free(req)
        # update scheduler running
        self.running = [req for req in self.running if not req.is_done]
        return self.running

    def schedule_prompt(self):
        running: list[Request] = []
        swapped: list[Request] = []
        for req in self.running:
            swapped.append(req)
        self.running = running
        return swapped, SwapMetadata(0, 0, sum([_.num_physical_token_blocks for _ in swapped]))

    def _swap_in(self, req: Request):
        print(f"Swap in request {req.id}")
        pass

    def _swap_out(self, req: Request):
        print(f"Swap out request {req.id}")
        pass

    def _remote_swap_out(self, req: Request):
        print(f"Remote swap out request {req.id}")
        pass

    def _remote_swap_in(self, requests: list[Request]) -> tuple[list[Request], list[Request]]:
        print(f"Remote swap in requests {[req.id for req in requests]}")
        return requests, []

    def free(self, req: Request):
        req.status = RequestStatus.FINISHED_STOPPED

    def workload(self):
        return 0


class LLMDynamicScheduler(LLMScheduler):
    def __init__(self, max_parallem_sum=200):
        super().__init__()
        self.max_parallem_sum = max_parallem_sum

    def schedule(self) -> tuple[list[Request], list[Request], SwapMetadata | None]:
        running = []
        swapped = []
        if not self.swapped:
            while self.waiting:
                request = self.waiting.pop()
                running.append(request)
                if sum([req.prompt_len for req in running]) == self.max_parallem_sum:
                    break
            if running:
                self.running.extend(running)
                return self.running, swapped, None

            self.running = sorted(self.running, key=lambda r: r.arrival_time)
            while self.running:
                running.append(self.running.pop())

        self.running.extend(running)
        return self.running, swapped, None

    def cpu_workload(self):
        return 0


class LLMStaticScheduler(LLMScheduler):
    def __init__(self, max_parallem_sum=0):
        super().__init__()
        self.max_parallem_sum = max_parallem_sum

    def schedule(self) -> tuple[list[Request], list[Request], SwapMetadata | None]:
        running = []
        if not self.swapped:
            while self.running:
                running.append(self.running.pop())
                if sum([req.prompt_len for req in running]) == self.max_parallem_sum:
                    break
            if running:
                self.running.extend(running)
                return running, self.swapped, None
            while self.waiting:
                running.append(self.waiting.pop())
                if sum([req.prompt_len for req in running]) == self.max_parallem_sum:
                    break
        self.running.extend(running)
        return running, self.swapped, None

    def cpu_workload(self):
        return 0


class LLMPromptScheduler(LLMScheduler):
    def __init__(self, max_parallem_sum=0):
        super().__init__()
        self.max_parallem_sum = max_parallem_sum

    def schedule(self) -> tuple[list[Request], list[Request], SwapMetadata | None]:
        running = []
        if len(self.waiting) >= self.max_parallem_sum:
            while self.waiting:
                running.append(self.waiting.pop())
        else:
            if self.running:
                while self.running:
                    running.append(self.running.pop())
            else:
                while self.waiting:
                    running.append(self.waiting.pop())
        self.running.extend(running)
        return running, self.swapped, None


class LLMPagedAttnScheduler(LLMScheduler):
    def __init__(
        self,
        id: int,
        cache_config: CacheConfig,
        lazy_swap: bool,
        max_parallem_sum=None,
        max_occupy_ratio: float = 1,
    ):
        super().__init__()

        self.id = id
        self.cache_config = cache_config
        self.lazy_swap = lazy_swap
        self.max_parallem_sum = max_parallem_sum

        self.block_manager: BlockManager = BlockManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
        )

        self.max_occupy_ratio = max_occupy_ratio

    def schedule(self) -> tuple[list[Request], list[Request] | None, SwapMetadata | None]:
        """The scheduler FSM for dynamic scheduling, basically being ported
            from the vLLM's Scheduler.schedule() method.

            currently we donnot consider requests whose generation length exceeds
            the max_len limitation.

        Return:
            A list of candidate requests for the next generation step.
        """
        num_blocks_to_swap_in: int = 0
        num_blocks_to_swap_out: int = 0

        scheduled: list[Request] = []
        if not self.swapped:
            while self.waiting:
                if not self._is_occupy_below_usage():
                    break
                if self.max_parallem_sum is not None and len(self.running) >= self.max_parallem_sum:
                    break

                req = self.waiting[0]

                if not self.block_manager.can_allocate(req):
                    # print("cant allocate waiting")
                    break

                req = self.waiting.pop(0)
                self.block_manager.allocate(req)
                req.status = RequestStatus.RUNNING
                self.running.append(req)
                scheduled.append(req)

                # if sum([req.prompt_len for req in scheduled]) >= self.max_parallem_sum:
                #     break

            if scheduled:
                return scheduled, None, None

        # [TODO: xuechao] sort self.running according to some priority policy.
        self.running = sorted(self.running, key=lambda req: req.arrival_time)

        # Reserve new token slots for the running sequence groups.
        running: list[Request] = []
        preempted: list[Request] = []
        if_swap: bool = False

        while self.running:
            req = self.running.pop(0)
            num_blocks_to_reserve = 0
            if self.lazy_swap:
                num_blocks_to_reserve = max(len(running), num_blocks_to_swap_out)
            while not self.block_manager.can_append_slot(num_blocks_to_reserve):
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_req = self.running.pop(-1)
                    self._preempt(victim_req)
                    num_blocks_to_swap_out += len(victim_req._physical_token_blocks)
                    preempted.append(victim_req)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(req)
                    num_blocks_to_swap_out += len(req._physical_token_blocks)
                    preempted.append(req)
                    break
            else:
                # Append new slots to the sequence group.
                self.block_manager.append_slot(req)
                running.append(req)
        self.running = running

        if num_blocks_to_swap_out > 500:
            print("wtf")
            pass
        if self.lazy_swap:
            self.block_manager.try_allocate(num_blocks_to_swap_out)

        # TODO: sort self.swapped according to some priority policy.
        self.swapped = sorted(self.swapped, key=lambda req: req.arrival_time)

        # Swap in the requests in the SWAPPED state if possible.
        while self.swapped and (num_blocks_to_swap_out == 0):
            if not self._is_occupy_below_usage():
                break
            if self.max_parallem_sum is not None and len(self.running) >= self.max_parallem_sum:
                break
            req = self.swapped[0]
            # If the sequence group has been preempted in this step, stop.
            if req in preempted:
                break
            # If the sequence group cannot be swapped in, stop.
            if not self.block_manager.can_swap_in(req, len(running) + 1):
                break
            req = self.swapped.pop(0)
            num_blocks_to_swap_in += len(req._physical_token_blocks)
            self._swap_in(req)
            self.block_manager.append_slot(req)
            self.running.append(req)

        return (
            self.running,
            preempted,
            SwapMetadata(self.id, num_blocks_to_swap_in, num_blocks_to_swap_out),
        )

    def _is_occupy_below_usage(self):
        (_, used_blks, all_blks) = self.block_manager.get_gpu_status()
        return used_blks / all_blks < self.max_occupy_ratio

    def _preempt(self, req: Request) -> None:
        # In lazy swap mode, request status must still be updated, and the swap related
        # meta data should also be recorded (e.g., number of blocks to be swapped). But
        # the swap operations are not necessarily performed immediately by current worker.
        # print(f"Preempt request {req.id}")
        if self.lazy_swap:
            self._remote_swap_out(req)
        else:
            self._swap_out(req)
            self.swapped.append(req)

    def _swap_in(self, req: Request) -> None:
        assert req.status == RequestStatus.SWAPPED
        self.block_manager.swap_in(req)
        req.status = RequestStatus.RUNNING

    def _remote_swap_in_running(self, req: Request) -> None:
        assert req.status == RequestStatus.SWAPPED_REMOTE
        self.block_manager.remote_swap_in_running(req)
        req.status = RequestStatus.RUNNING

    def _remote_swap_in_swapped(self, req: Request) -> None:
        assert req.status == RequestStatus.SWAPPED_REMOTE
        self.block_manager.remote_swap_in_swapped(req)
        req.status = RequestStatus.SWAPPED

    def _swap_out(self, req: Request) -> None:
        if not self.block_manager.can_swap_out(req):
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error."
            )
        assert req.status == RequestStatus.RUNNING
        self.block_manager.swap_out(req)
        req.status = RequestStatus.SWAPPED

    def _remote_swap_out(self, req: Request) -> None:
        assert req.status == RequestStatus.RUNNING
        self.block_manager.remote_swap_out(req)
        req.status = RequestStatus.SWAPPED_REMOTE

    def _remote_swap_in(self, requests: list[Request]):
        to_running: list[Request] = []
        to_swapped: list[Request] = []
        for req in requests:
            if self.block_manager.can_swap_in(req) and self._is_occupy_below_usage():
                self._remote_swap_in_running(req)
                to_running.append(req)
            elif self.block_manager.can_swap_out(req):
                self._remote_swap_in_swapped(req)
                to_swapped.append(req)
            else:
                raise RuntimeError("Aborted due to the lack of CPU swap space.")
        return to_running, to_swapped

    def free(self, req: Request):
        self.block_manager.free(req)
        req.status = RequestStatus.FINISHED_STOPPED

    def workload(self):
        """返回GPU内存使用情况"""
        free_blocks, used_blocks, total_blocks = self.block_manager.get_gpu_status()
        return (used_blocks / total_blocks) * 100 if total_blocks > 0 else 0

    def cpu_workload(self):
        """返回CPU内存使用情况"""
        _GB = 1 << 30
        capacity = (
            self.block_manager.cpu_allocator.get_num_allocated_blocks()
            * self.cache_config.block_size
            * self.cache_config.size_per_token
            / _GB
        )
        # return self.block_manager.cpu_allocator.get_num_allocated_blocks()

        return capacity
