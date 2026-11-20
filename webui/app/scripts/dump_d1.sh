#!/bin/sh
# web.db (webui/prepdb.py) -> web.sql that D1 accepts: drops the old tables
# first so it can be re-run, and strips the dump's BEGIN/COMMIT (D1 rejects them).
# The hit_marks, submodel_merges and type_settings tables (edits made in the app) are only created if missing, never dropped.
DB="${DATA_DIR:-$HOME/Documents/dolphin_web_data}/web.db"
{
  echo "DROP TABLE IF EXISTS display_set;"
  echo "DROP TABLE IF EXISTS linkages;"
  echo "CREATE TABLE IF NOT EXISTS hit_marks (results_id INTEGER PRIMARY KEY, mark TEXT NOT NULL);"
  echo "CREATE TABLE IF NOT EXISTS type_settings (classification TEXT PRIMARY KEY, enabled INTEGER NOT NULL, min_llr REAL NOT NULL, min_conf REAL NOT NULL);"
  echo "CREATE TABLE IF NOT EXISTS submodel_merges (child INTEGER PRIMARY KEY, parent INTEGER NOT NULL);"
  sqlite3 "$DB" .dump | grep -v -e '^BEGIN TRANSACTION;' -e '^COMMIT;' -e '^PRAGMA'
} > web.sql
echo "wrote web.sql from $DB"
