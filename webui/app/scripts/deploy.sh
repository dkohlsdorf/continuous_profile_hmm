#!/bin/sh
# Deploy the web UI to Cloudflare (Workers + D1 + R2). Run from anywhere.
#
#   webui/app/scripts/deploy.sh [db] [images] [app]     (no args = all three)
#
#   db      create the D1 database if missing, write its id into wrangler.jsonc,
#           load DATA_DIR/web.db (~150k writes: needs the paid Workers plan or two days)
#   images  create the R2 bucket if missing, enable its public r2.dev URL, write it into
#           wrangler.jsonc as IMAGE_BASE_URL, rclone-copy spectrograms/ + region_spectrograms/
#   app     wrangler deploy
#
# Every step is idempotent: rerun `db` after a new prepdb.py, `images` after new
# spectrograms (rclone skips files already uploaded), `app` after code changes.
#
# Needs: `npx wrangler login` done once, rclone (brew install rclone) for `images`, and
# for `images` an R2 API token (dashboard: R2 > Manage API tokens, Object Read & Write):
#   CLOUDFLARE_ACCOUNT_ID  R2_ACCESS_KEY_ID  R2_SECRET_ACCESS_KEY
# Env: DATA_DIR (default ~/Documents/dolphin_web_data), DB_NAME (dolphin-web),
# BUCKET (dolphin-web-images).
#
# Afterwards put the site behind Cloudflare Access (Zero Trust > Access > Applications):
# the edit endpoints have no login.
set -eu

cd "$(dirname "$0")/.."
DATA_DIR=${DATA_DIR:-$HOME/Documents/dolphin_web_data}
DB_NAME=${DB_NAME:-dolphin-web}
BUCKET=${BUCKET:-dolphin-web-images}
STEPS=${*:-db images app}
WRANGLER="npx wrangler"

# set_config KEY VALUE: replaces "KEY": "..." in wrangler.jsonc
set_config() {
  perl -pi -e "s|(\"$1\":\\s*)\"[^\"]*\"|\$1\"$2\"|" wrangler.jsonc
  echo "wrangler.jsonc: $1 = $2"
}

[ -d node_modules ] || npm install

for step in $STEPS; do
  case $step in
  db)
    echo "== D1 database $DB_NAME"
    db_id() {
      $WRANGLER d1 list --json | node -e \
        'const n=process.argv[1];let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const d=JSON.parse(s).find(x=>x.name===n);if(d)console.log(d.uuid)})' "$DB_NAME"
    }
    ID=$(db_id)
    if [ -z "$ID" ]; then
      $WRANGLER d1 create "$DB_NAME"
      ID=$(db_id)
    fi
    set_config database_id "$ID"
    DATA_DIR=$DATA_DIR npm run db:remote
    ;;
  images)
    echo "== R2 bucket $BUCKET"
    : "${CLOUDFLARE_ACCOUNT_ID:?set CLOUDFLARE_ACCOUNT_ID}"
    : "${R2_ACCESS_KEY_ID:?set R2_ACCESS_KEY_ID}"
    : "${R2_SECRET_ACCESS_KEY:?set R2_SECRET_ACCESS_KEY}"
    $WRANGLER r2 bucket list | grep -qE "^name:[[:space:]]+$BUCKET\$" || $WRANGLER r2 bucket create "$BUCKET"
    URL=$($WRANGLER r2 bucket dev-url get "$BUCKET" | grep -o 'https://[^ '\'']*\.r2\.dev' || true)
    if [ -z "$URL" ]; then
      $WRANGLER r2 bucket dev-url enable "$BUCKET" --force
      URL=$($WRANGLER r2 bucket dev-url get "$BUCKET" | grep -o 'https://[^ '\'']*\.r2\.dev')
    fi
    set_config IMAGE_BASE_URL "$URL"
    # rclone remote "r2" from env, no rclone.conf needed
    export RCLONE_CONFIG_R2_TYPE=s3 RCLONE_CONFIG_R2_PROVIDER=Cloudflare \
      RCLONE_CONFIG_R2_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID \
      RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY \
      RCLONE_CONFIG_R2_ENDPOINT="https://$CLOUDFLARE_ACCOUNT_ID.r2.cloudflarestorage.com" \
      RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true
    for dir in spectrograms region_spectrograms; do
      rclone copy "$DATA_DIR/$dir" "r2:$BUCKET/$dir" --transfers 32 --progress
    done
    ;;
  app)
    echo "== Worker"
    $WRANGLER deploy
    ;;
  *)
    echo "unknown step: $step (expected db, images, app)" >&2
    exit 1
    ;;
  esac
done
