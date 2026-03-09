import time
from collections.abc import Iterator
from typing import Optional
import threading
from threading import Thread, Lock
from queue import SimpleQueue

import logging

class VolumeSampler:
    """
    Generates slices of cell_size
    respecting output_volume_size boundary.
    """

    def __init__(
        self,
        output_volume_size: tuple[int, int, int],
        cell_size: tuple[int, int, int]
    ):
        """
        Store arguments
        """
        self.output_volume_size: tuple[int, int, int] = output_volume_size
        self.cell_size: tuple[int, int, int] = cell_size

    def __iter__(self):
        """
        Cell metadata generator.
        Returns cell coordinates as well as tile id information.
        """
        oz, oy, ox = self.output_volume_size
        cz, cy, cx = self.cell_size

        for z in range(0, oz, cz):
            for y in range(0, oy, cy):
                for x in range(0, ox, cx):
                    curr_cell: geometry.AABB = \
                    (z, min(z + cz, oz),
                     y, min(y + cy, oy),
                     x, min(x + cx, ox))

                    yield curr_cell


class ThreadQueue:
    """
    Cache without shared memory constraint.
    Cache is as large as RAM.
    """
    def __init__(
        self,
        max_size: int,
        logger: Optional[logging.Logger] = None,
        name: Optional[str] = None
        ):
        self.queue = SimpleQueue()
        self.max_size = max_size  # Limits put operation
        self.current_size = 0

        if (logger is None) != (name is None):
            raise ValueError("Logger and name must be provided together or not at all.")
        self.logger = logger
        self.name = name

    def put(self, item):
        self.guard_put()
        self.queue.put(item)
        self.current_size += 1

    def get(self):
        self.guard_get()
        item = self.queue.get()
        self.current_size -= 1
        return item

    def qsize(self):
        return self.current_size

    def guard_put(self):
        """
        Skips thread/waits for queue consumption
        if queue is already at max capacity.
        Issues warning on excessive stall.
        """
        stall_count = 0
        stall_alert = 10
        while self.current_size >= self.max_size:
            stall_count += 1

            if self.logger and stall_count >= stall_alert:
                warning = \
                f"""Put thread in {self.name} has been stalled {stall_count} times.
                Consider increasing the queue size."""
                self.logger.info(warning)

            time.sleep(1)  # Yield control to another thread

    def guard_get(self):
        """
        Skips thread/waits for queue to fill
        if queue is empty.
        Issues warning on excessive stall.
        """
        stall_count = 0
        stall_alert = 10
        while self.qsize() == 0:
            stall_count += 1

            if self.logger and stall_count >= stall_alert:
                warning = \
                f"""Get thread in {self.name} has been stalled {stall_count} times.
                Consider increasing processing power."""
                self.logger.info(warning)

            time.sleep(1)  # Yield control to another thread

class CloudDataloader(Iterator):
    def __init__(
        self,
        dataset,
        sampler,
        num_workers: int,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize the ThreadQueue and worker pool.
        """
        self.dataset = dataset
        self.sampler = iter(sampler)
        self.iterator_lock = Lock()
        log_name = 'CloudDataloader'
        if logger is None:
            log_name = None
        self.output_queue = ThreadQueue(max_size=num_workers,
                                        logger=logger,
                                        name=log_name)
        self.num_workers = num_workers
        self.workers: list[Thread] = []
        self.finished_cnt = 0

        for i in range(self.num_workers):
            t = Thread(target=self.load_cell)
            t.start()
            self.workers.append(t)
        print('Warming up Queue...')
        time.sleep(10)

    def load_cell(self):
        """
        Called across multiple threads.
        Dispenses from sampler and
        places item into queue.
        """
        while True:
            try:
                with self.iterator_lock:
                    cell_aabb, src_ids = next(self.sampler)
                item = self.dataset[(cell_aabb, src_ids)]
                self.output_queue.put(item)
            except StopIteration:
                self.finished_cnt += 1
                break
        
        if self.finished_cnt == self.num_workers:
            self.output_queue.put(None)

    def __next__(self):
        """
        Called by main thread.
        Places items into queue and
        shuts down thread pool after receiving 'None'.
        """
        if self.num_workers == 0:
            with self.iterator_lock:
                return self.dataset[next(self.sampler)]
        
        item = self.output_queue.get()
        
        if item is None: 
            self.shutdown()
            raise StopIteration

        return item

    def shutdown(self):
        """
        Close the thread pool.
        """
        for thread in self.workers:
            if thread != threading.current_thread():
                thread.join()
        self.workers.clear()

        print('Shutdown CloudDataloader.')