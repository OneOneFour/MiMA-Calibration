import sys


def detect_free_threading() -> bool:
    """Detect if the current Python environment supports free-threading.
    Returns:
        bool: True if free-threading is detected, False otherwise.
    """
    if "free-threading" in sys.version:
        print("Free-threading IS supported in this Python environment.")
        return True
    print("Free-threading IS NOT supported in this Python environment.")
    return False


def process_workers(n_processes):
    if detect_free_threading():
        return {
            "use_thread_pool": True,
            "n_worker": n_processes,
        }
    return {
        "n_process": n_processes,
    }
