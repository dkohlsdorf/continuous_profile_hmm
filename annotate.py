"""
annotate.py -- merge one recording's motif_hits.csv (from `motif_discovery.py
find`) into the matching rows of a PVL shotlog spreadsheet, producing
annotated_motif_hits.csv in the same output folder.

Adapted from a batch script that walked a whole directory of per-encounter
output folders against a year's PVL in one pass (folder_map()/the original
annotate()). This version is scoped to exactly one output folder + one PVL
file, since that's what the serverless find worker calls per recording (see
serverless/runpod_worker/handler.py's run_annotate()) -- the batch/multi-year
orchestration itself isn't carried over; wrap this in your own loop (or ask
for it back) if you still want that standalone.

Encounter number: taken from the output folder's own name via key(), e.g.
"E1234_whatever" -> 1234. The folder must follow that convention -- there's
no other way to know which PVL rows belong to this recording. If your find
output folders aren't named this way, rename them (or symlink) before
calling this, or pass --encounter to override the folder-name guess.

Usage:
    python annotate.py <output_folder> <pvl_path> [--encounter N]

    output_folder   a `find` output directory containing motif_hits.csv
                     (written next to it: annotated_motif_hits.csv)
    pvl_path        the PVL .xlsx shotlog

Column mapping (find -> PVL shotlog):
    classification        -> shotlog::AC
    submodel_id            -> shotlog::BEHdescription
    mean_llr_score          -> SPECIAL COMMENTS
    start_time_s (as H:MM:SS, zero-padded to HH:MM:SS) -> shotlog::timecode
    the PVL row's own Date/ENC # (ffilled, so blank cells inherit the last
    seen value -- PVL shotlogs typically only stamp Date/ENC # on a
    sighting's first row) are copied onto every motif row for this
    encounter, then all rows (PVL + motif) are merged and time-sorted.
"""
import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd

PVL_COLUMNS = ['Date', 'ENC #', 'shotlog::AC', 'shotlog::BEHdescription', 'shotlog::timecode', 'SPECIAL COMMENTS']


def key(folder_name):
    """"E1234_whatever" -> 1234. Raises ValueError if it doesn't fit that shape."""
    return int(folder_name.split('_')[0][1:])


def fix_timecode(timecode):
    """PVL timecodes carry a trailing "-<frames>" (e.g. "00:05:23-12") -- drop it."""
    return str(timecode).split('-')[0]


def process_pvl(pvl_path):
    """
    Load a PVL shotlog and forward-fill Date/ENC # -- PVL sightings typically
    only stamp those on a row's first appearance, leaving subsequent rows of
    the same sighting blank, so every row needs its own copy to be groupable
    by encounter.
    """
    df = pd.read_excel(pvl_path)
    df = df[PVL_COLUMNS]
    filled = df.ffill()
    df['Date'] = filled['Date']
    df['ENC #'] = filled['ENC #'].astype(int)
    df['shotlog::timecode'] = df['shotlog::timecode'].apply(fix_timecode)
    return df


def annotate_one(output_folder, pvl_path, encounter=None):
    """
    Merge output_folder/motif_hits.csv into pvl_path's rows for one
    encounter, writing output_folder/annotated_motif_hits.csv. Returns the
    written path.

    encounter defaults to key(output_folder's name) -- pass it explicitly
    to override when the folder isn't named "E<encounter>_...".
    """
    output_folder = Path(output_folder)
    if encounter is None:
        try:
            encounter = key(output_folder.name)
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"couldn't read an encounter number from folder name '{output_folder.name}' "
                f"(expected \"E<number>_...\"): {e}. Pass --encounter to override."
            ) from e

    motif_path = output_folder / 'motif_hits.csv'
    if not motif_path.exists():
        raise ValueError(f"{motif_path} not found -- run `find` on this folder first")

    df = process_pvl(pvl_path)
    filtered = df[df['ENC #'] == encounter].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"No PVL rows found for encounter {encounter} "
                          f"(from folder name '{output_folder.name}') in {pvl_path}")

    motifs = pd.read_csv(motif_path)
    motifs['shotlog::AC'] = motifs['classification']
    motifs['SPECIAL COMMENTS'] = motifs['mean_llr_score']
    motifs['shotlog::BEHdescription'] = motifs['submodel_id']
    motifs['Date'] = filtered['Date'][0]
    motifs['ENC #'] = filtered['ENC #'][0]
    # Zero-padded to match fix_timecode()'s "HH:MM:SS" PVL format -- only
    # correct for encounters under 10 hours (str(timedelta) drops the
    # leading zero on the hour field below that, which the prepended '0'
    # restores; at >=10 hours it would prepend onto an already 2-digit
    # hour instead). Matches the original script's assumption exactly.
    motifs['shotlog::timecode'] = motifs['start_time_s'].apply(
        lambda x: '0' + str(datetime.timedelta(seconds=int(x)))
    )
    motifs = motifs[PVL_COLUMNS]

    annotated = pd.concat([motifs, filtered])
    annotated = annotated.sort_values('shotlog::timecode')
    annotated = annotated[PVL_COLUMNS]

    annotated_path = output_folder / 'annotated_motif_hits.csv'
    annotated.to_csv(annotated_path)
    print(f"encounter {encounter}: {len(motifs)} motif hits + {len(filtered)} PVL rows "
          f"-> {annotated_path}")
    return annotated_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_folder", help="`find` output directory containing motif_hits.csv")
    parser.add_argument("pvl_path", help="PVL .xlsx shotlog")
    parser.add_argument("--encounter", type=int, default=None,
                         help="override the encounter number instead of guessing it from output_folder's name")
    args = parser.parse_args()

    try:
        annotate_one(args.output_folder, args.pvl_path, encounter=args.encounter)
    except (ValueError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
