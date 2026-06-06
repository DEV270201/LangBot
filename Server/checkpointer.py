from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from Server.config import DATABASE_URI

pool = ConnectionPool(
    conninfo=DATABASE_URI,
    max_size=10,
    kwargs={
        "autocommit": True,
        "prepare_threshold": 0,
    },
)

checkpointer = PostgresSaver(pool)
checkpointer.setup()
