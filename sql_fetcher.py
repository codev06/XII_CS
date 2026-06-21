import mysql.connector
from mysql.connector import errorcode

class MySQLDataFetcher:
    def __init__(self, host="localhost", user="root", password="seanc0de!", database="sentiment_db", port=3306):
        self.config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database,
            'port': port,
            'connect_timeout': 3
        }


    def fetch_record(self, table="conversations", column="raw_payloadtext", text_payload=None,record_id=None):
        """Queries raw text from a specified row inside MySQL safely."""
        conn = None
        cursor = None
        try:
            conn = mysql.connector.connect(**self.config)
            cursor = conn.cursor()
            query = f"SELECT {column} FROM {table} WHERE id = %s"
            cursor.execute(query, (record_id,))
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            print(f"Extraction Error: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()

    def insert_record(self, table="conversations", column="raw_payloadtext", text_payload=None):
        """Inserts a new raw text record into the database and returns its auto-incremented ID."""
        conn = None
        cursor = None
        try:
            conn = mysql.connector.connect(**self.config)
            cursor = conn.cursor()
            
            # Ensure the insertion targets the requested 'raw_payloadtext' column
            target_column = "raw_payloadtext" if column == "raw_payload" else column
            
            # Parametric query execution ensures safe input sanitation
            query = f"INSERT INTO {table} ({target_column}) VALUES (%s)"
            cursor.execute(query, (text_payload,))
            conn.commit()
            
            # Retrieve the newly inserted auto-incremented primary key ID
            return cursor.lastrowid
        except Exception as e:
            print(f"Database insertion failure error: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()