from Server.checkpointer import checkpointer


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def delete_thread(thread_id):
    """Permanently remove a conversation's checkpoints from Postgres."""
    checkpointer.delete_thread(thread_id)
