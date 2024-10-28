from typing import List, Tuple
from sqlalchemy import desc, asc
from sqlalchemy import text, bindparam


class TrainingDataManager(object):
    def __init__(self, table_type, conn):
        self._table_type = table_type
        self._session = None
        self._conn = conn
        self._cursor = conn.cursor()


    def new_training_data(self) -> List[Tuple[bytes]]:
        return self._session.query(self._table_type.text).filter(self._table_type.trained == 0).all()

    
    def all_training_data(self, limit: int = None, order_by: str = 'id', order='desc') -> List[Tuple[bytes]]:
        self._cursor.execute(f"SELECT id, text FROM {self._table_type.__tablename__} WHERE trained != 1 ORDER BY {order_by} {order} LIMIT {limit}")
        return self._cursor.fetchall()

    def mark_trained(self, message_ids=None):
        if not message_ids:
            return
        if len(message_ids) == 1:
            self._cursor.execute(f"UPDATE {self._table_type.__tablename__} SET trained = 1 WHERE id = ?", (message_ids[0],))
        else:
            self._cursor.execute(f"UPDATE {self._table_type.__tablename__} SET trained = 1 WHERE id IN {tuple(message_ids)}")
        self._conn.commit()

    def close_connection(self):
        self._conn.close()
            
    def OLD_mark_trained(self):
        self._session.execute(text('UPDATE ' + self._table_type.__tablename__ + ' SET TRAINED = 1'))
        self._session.commit()

    def mark_untrained(self):
        self._session.execute(text('UPDATE ' + self._table_type.__tablename__ + ' SET TRAINED = 0'))
        self._session.commit()
        
    def commit(self):
        self._session.commit()

    def store(self, data):
        pass

