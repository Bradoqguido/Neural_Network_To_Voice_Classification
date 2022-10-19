print('Importing libraries...')
import logging
from sqlite3 import SQLITE_ALTER_TABLE
from sqlalchemy import create_engine

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

logging.info('starting db engine...')
db = create_engine('postgresql://postgres:postgres@localhost:5432/VoiceDataMining')

conn = db.connect()
conn.autocommit = False
cursor = conn.cursor()

logging.info('Creating postgresql FUNCTION: trigger_set_timestamp...')
sql = """'
CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
'"""
cursor.execute(sql)

logging.info('Creating postgresql ENUM: VALID_FILE_TYPES...')
sql = """'
CREATE TYPE IF NOT EXISTS VALID_FILE_TYPES AS ENUM ('TRAIN', 'TEST', 'VALIDATION');
'"""
cursor.execute(sql)

logging.info('Creating postgresql TABLE: file_index...')
sql = """'
CREATE TABLE IF NOT EXISTS file_index (
    id_file SERIAL PRIMARY KEY,
    file_name TEXT NOT NULL,
    id_speaker INTEGER NOT NULL,
    file_type VALID_FILE_TYPES NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
'"""
cursor.execute(sql)

logging.info('Creating postgresql timestamp: TRIGGER for file_index...')
sql = """'
CREATE TRIGGER set_file_index_timestamp
BEFORE UPDATE ON file_index
FOR EACH ROW
EXECUTE PROCEDURE TRIGGER_SET_TIMESTAMP();
'"""
cursor.execute(sql)

logging.info('Creating postgresql TABLE: feature_index...')
sql = """'
CREATE TABLE IF NOT EXISTS feature_index (
    id SERIAL PRIMARY KEY,
    mfccs TEXT NOT NULL,
    chroma TEXT INTEGER NOT NULL,
    mel TEXT NOT NULL,
    contrast TEXT NOT NULL,
    tonnetz TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
'"""
cursor.execute(sql)

logging.info('Creating postgresql timestamp: TRIGGER for feature_index...')
sql = """'
CREATE TRIGGER set_feature_index_timestamp
BEFORE UPDATE ON feature_index
FOR EACH ROW
EXECUTE PROCEDURE TRIGGER_SET_TIMESTAMP();
'"""
cursor.execute(sql)

# conn.commit()
conn.close()
