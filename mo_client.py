def connect_mo(config):
    try:
        from matrixone import Client as MatrixOneClient
    except Exception:
        MatrixOneClient = None

    if MatrixOneClient is not None:
        client = MatrixOneClient()
        client.connect(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            database=config["database"],
        )
        return _MatrixOneAdapter(client)

    try:
        import pymysql
    except Exception as exc:
        raise SystemExit(
            "Missing dependency. Install with: python3 -m pip install pymysql"
        ) from exc

    conn = pymysql.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        charset="utf8mb4",
        autocommit=True,
    )
    adapter = _PyMySQLAdapter(conn)
    adapter.ensure_database(config["database"])
    return adapter


class _MatrixOneAdapter:
    def __init__(self, client):
        self._client = client

    def execute(self, sql):
        return self._client.execute(sql)

    def disconnect(self):
        self._client.disconnect()


class _PyMySQLAdapter:
    def __init__(self, conn):
        self._conn = conn

    def ensure_database(self, database):
        cur = self._conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        cur.execute(f"USE {database}")

    def execute(self, sql):
        cur = self._conn.cursor()
        cur.execute(sql)
        return cur

    def disconnect(self):
        self._conn.close()
