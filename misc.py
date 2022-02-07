from multiprocessing import Process, Queue

# Multiprocessor allowing multiple functions to be executed in parallel.
class Multiprocessor():

    def __init__(self):
        self.processes = []
        self.queue = Queue()

    @staticmethod
    def _wrapper(f, queue, args, kwargs):
        res = f(*args, **kwargs)
        queue.put(res)

    # Add a function to be run in parallel
    def run(self, f, *args, **kwargs):
        args_wrapper = [f, self.queue, args, kwargs]
        p = Process(target=self._wrapper, args=args_wrapper)
        self.processes.append(p)
        p.start()

    # Wait for the executions to finish.
    # Returns the results of the input functions.
    def wait(self):
        results = []
        for p in self.processes:
            res = self.queue.get()
            results.append(res)
        for p in self.processes:
            p.join()
        return results