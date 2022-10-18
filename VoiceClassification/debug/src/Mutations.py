print('Importing libraries...')
import logging
from sqlalchemy import create_engine

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

logging.info('starting db engine...')
db = create_engine('postgresql://postgres:postgres@localhost:5432/VoiceDataMining')

conn = db.connect()
conn.autocommit = False
cursor = conn.cursor()

sql = """'
CREATE TYPE VALID_FILE_TYPES IF NOT EXISTS AS ENUM ('TRAIN', 'TEST', 'VALIDATION');
'"""

cursor.execute(sql)

sql = """'
CREATE TABLE IF NOT EXISTS FILE_INDEX (
    ID GENERATED PRIMARY KEY,
    FILE_NAME TEXT NOT NULL,
    ID_SPEAKER INTEGER NOT NULL,
    FILE_TYPE VALID_FILE_TYPES NOT NULL
);
'"""

cursor.execute(sql)

conn.commit()
conn.close()
