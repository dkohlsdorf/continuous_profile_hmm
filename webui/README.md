# Web UI

A browser for the motif hits of one extend-find run: one row per cluster
(submodel), rows ordered by greedy seriation on a chosen linkage, each row
showing the cluster's hits as spectrograms. Clusters can be merged, and hits
can be marked as noise or as wrongly assigned. Everything runs on Cloudflare
(Workers + D1 + R2) and can be run fully locally.

```
webui/
  preprocessing.sh          full data pipeline (the four scripts below, in order)
  preprocessing.py          1. run hits -> spectrograms / region spectrograms / embeddings on Drive
  display_set.py            2. pick the display set, download its files
  greedy_seriation_prep.py  3. pairwise DTW + submodel linkages
  prepdb.py                 4. build web.db (SQLite)
  gdrive_helper.py          Google Drive helpers shared by 1 and 2
  app/                      the Cloudflare Worker + page
  app/scripts/deploy.sh     deploy to Cloudflare (D1 + R2 + Worker)
```

## 1. Data preparation

### Requirements

- Python with the repo's dependencies (pandas, numpy, librosa, soundfile,
  matplotlib, google-api-python-client), and `ffmpeg` on the `PATH`.
- The compiled DTW extension in `lib_phmm/`. If
  `lib_phmm/hierarchical_kmedian_dtw*.so` is missing, build it:
  `cd lib_phmm && python setup.py build_ext --inplace`
- Google Drive credentials, the same ones the RunPod handler uses: either
  `GOOGLE_REFRESH_TOKEN` + `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET`, or
  `GOOGLE_SERVICE_ACCOUNT_JSON` (base64).

### Run

```sh
webui/preprocessing.sh RUN_FOLDER_ID CACHE_FOLDER_ID WAV_FOLDER_ID [WAV_FOLDER_ID ...]
```

- `RUN_FOLDER_ID`: Drive id of the `motif_extend_find_<timestamp>` folder
- `CACHE_FOLDER_ID`: Drive id of `motif_embeddings_cache`
- `WAV_FOLDER_ID`: Drive ids of the recording folders

Environment: `DATA_DIR` (default `~/Documents/dolphin_web_data`) and
`EMBEDDING_MODEL` (default `whisper`).

The script runs these four steps and stops at the first one that fails:

| step | script | output |
|---|---|---|
| 1 | `preprocessing.py` | `webui_preprocessing/` on Drive: `spectrograms/`, `region_spectrograms/`, `embeddings/`, `results.csv` |
| 2 | `display_set.py --run-folder …` | `DATA_DIR/results.csv`, `DATA_DIR/results_display_set.csv` (top 15 hits by `mean_llr_score` per submodel with classification confidence > 0.8; submodels with < 5 such hits dropped), plus the display set's files in `DATA_DIR/{spectrograms,region_spectrograms,embeddings}/` |
| 3 | `greedy_seriation_prep.py` | `DATA_DIR/results_dtw.csv` (DTW between every pair of hits, ~1 GB) and `DATA_DIR/submodel_dtw.csv` (single / complete / average / medoid linkage between every pair of submodels). Takes ~35 min for ~5k hits. |
| 4 | `prepdb.py` | `DATA_DIR/web.db` with tables `display_set` and `linkages` |

Steps 1 and 2 can be restarted: preprocessing skips recordings listed in
`done.txt`, and the downloads skip files already on disk. Each step can also be
run on its own; see the docstring at the top of each script (`--help`).

## 2. Run the app locally

Needs Node.js (v20+) and the `sqlite3` CLI. Run all commands from `webui/app/`.

```sh
cd webui/app
npm install           # once: installs wrangler (the Cloudflare CLI)
npm run db:local      # load DATA_DIR/web.db into the local D1 database
```

Then start two servers, each in its own terminal:

```sh
npm run images        # serves DATA_DIR on http://localhost:8788 (the spectrogram PNGs)
npm run dev           # the app on http://localhost:8787
```

Open http://localhost:8787.

- Rerun `npm run db:local` after every `prepdb.py`. It replaces `display_set`
  and `linkages` but keeps the edits made in the app (see below).
- The local database lives in `webui/app/.wrangler/`. Deleting that folder
  deletes all local edits.
- Local dev reads images from `IMAGE_BASE_URL` in `app/.dev.vars`
  (`http://localhost:8788`). Without `npm run images`, every image shows as broken.
- `DATA_DIR` changes the data folder for `npm run images` and `npm run db:local`.

### Using the app

- **Linkage** (header): which linkage orders the rows. Seriation starts from the
  closest pair of clusters, then repeatedly adds the cluster nearest to the last row.
- **Per type** (BURST / ECHO / WHISTLE): show/hide checkbox, minimum
  `mean_llr_score` and minimum `classification_confidence`. These settings are
  saved in the database and shared by everyone using the app.
- **Row label**: submodel (or merged group and its members), majority type,
  hit count, distance to the row above, and the mean ± sd of llr and confidence
  over the shown hits.
- **Click a spectrogram**: shows the region spectrogram with the hit outlined,
  its metadata, and these actions: mark as noise, "not this cluster", or mark
  the whole cluster (all merged submodels) as noise. Marked hits move to the
  **noise** / **wrongly assigned** rows at the top. Click a hit there to restore it.
- **↑ merge**: merges the row into the row above. **×** next to a group member
  unlinks that submodel from its group.
- **Reset**: removes all merges and all marks (asks first; cannot be undone).

### Tables in D1

| table | written by | contents |
|---|---|---|
| `display_set` | `prepdb.py` | one row per hit |
| `linkages` | `prepdb.py` | `model_i, model_j, single, complete, average, medoid` |
| `hit_marks` | the app | `results_id, mark` (`noise` or `wrong`) |
| `submodel_merges` | the app | `child, parent`: union-find parent pointers between submodel ids |
| `type_settings` | the app | `classification, enabled, min_llr, min_conf` |

`npm run db:local` / `db:remote` only create the last three tables if they are
missing, so edits survive a database rebuild. If a new run renumbers its
`results_id`s or submodels, press **Reset** first, because the old edits would
point at different hits.

Export the edits:

```sh
npx wrangler d1 execute DB --local --command "SELECT * FROM hit_marks" --json > hit_marks.json
npx wrangler d1 execute DB --local --command "SELECT * FROM submodel_merges" --json > merges.json
```

## 3. Deploy to Cloudflare

`app/scripts/deploy.sh` creates the D1 database and the R2 bucket, loads the
data, uploads the spectrograms and deploys the Worker. Only the login and the
Access setup are manual.

### Once

```sh
cd webui/app
npx wrangler login
brew install rclone
```

Create an R2 API token for the image upload (dashboard: **R2 → Manage API
tokens**, permission *Object Read & Write*) and export it with your account id:

```sh
export CLOUDFLARE_ACCOUNT_ID=...  R2_ACCESS_KEY_ID=...  R2_SECRET_ACCESS_KEY=...
```

### Deploy

```sh
webui/app/scripts/deploy.sh               # db + images + app
webui/app/scripts/deploy.sh app           # after code changes
webui/app/scripts/deploy.sh db images     # after a new data run
```

| step | what it does |
|---|---|
| `db` | creates the D1 database `dolphin-web` if missing, writes its id into `app/wrangler.jsonc`, loads `DATA_DIR/web.db` (`npm run db:remote`) |
| `images` | creates the R2 bucket `dolphin-web-images` if missing, enables its public r2.dev URL, writes it into `app/wrangler.jsonc` as `IMAGE_BASE_URL`, uploads `spectrograms/` and `region_spectrograms/` with rclone (~4.8 GB; files already uploaded are skipped) |
| `app` | `wrangler deploy`; prints the URL `https://dolphin-web.<subdomain>.workers.dev` |

Every step can be rerun. Env: `DATA_DIR`, `DB_NAME` (default `dolphin-web`),
`BUCKET` (default `dolphin-web-images`).

- `db` rewrites all ~150k linkage rows on every run. That is more than the
  free tier's 100k D1 writes per day, so it needs the $5/month Workers plan or
  two days. Don't run it for code-only changes.
- `db` writes the real `database_id` into `app/wrangler.jsonc`, which is
  tracked in git. That id is not a secret, but check the diff before committing.
- Export the edits from the deployed app with the commands in section 2,
  using `--remote` instead of `--local`.

### Restrict access (Cloudflare Access)

The edit endpoints (`/api/marks`, `/api/merges`, `/api/reset`) have no login:
anyone with the URL can change or wipe the edits. Before sharing the URL:

1. **Workers & Pages → dolphin-web → Settings → Domains & Routes**, and next to
   `workers.dev` click **Enable Cloudflare Access**. The first time, this asks
   you to set up Zero Trust: pick a team name and the **Free** plan (up to 50
   users; it may ask for a payment method).
2. **Manage Cloudflare Access** opens the generated application. Edit its
   policy: action **Allow**, include **Emails** (list of addresses) or
   **Emails ending in** (`@yourlab.org`).
3. Open the URL in a private window: you get a Cloudflare login that emails a
   one-time code, and only the listed addresses get in.

Add or remove people by editing the policy; no redeploy needed. With a custom
domain or no toggle, create it by hand instead: **Zero Trust → Access →
Applications → Add → Self-hosted**, domain `dolphin-web.<subdomain>.workers.dev`
with an empty path, and the same policy.

The r2.dev image URL is not covered by Access: the spectrograms stay publicly
readable to anyone who knows a file name. Only the data and the edit endpoints
are protected. To protect the images too, give the bucket a custom domain on a
zone you own and add a second Access application for it.
