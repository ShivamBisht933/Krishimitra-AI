
import os
import mysql.connector


def get_db_connection():

    password = os.getenv("MYSQL_PASSWORD") or "root"

    if not password:
        raise ValueError("MYSQL_PASSWORD environment variable is not set.")

    connection = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password=password,
        database="krishimitra_db"
    )

    return connection
