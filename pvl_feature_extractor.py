import sqlite3
import sys
import os
import glob
import pandas as pd
import re
import numpy as np
import json


CONTEXTS = {"DIVE", "CALF", "NUZZLING", "ASCENT", "BARRACUDA", "FISH", "BITE", "BODY CHARGE", 
            "BODY TO BODY", "TUSSLE", "SARGASSUM", "BUBBLE RING", "SURFACE SPLASH", "BUBBLE TRAIL",
            "THRASHES", "CHASE", "COPULATING", "SEAGRASS", "DESCENT", "DIG", "DISCIPLINE", "RUBBING", 
            "FORAG", "SWIM", "FLYING", "ZOOM", "BUZZ", "TRAVELING", "HEAD TO HEAD", "HEAD BOBBING", "PEC TO PEC", 
            "MELON TO MELON RUB", "DISCIPLINES", "TACTILE", "OPEN MOUTH", "PEC SLAP", "PEC RUBBING", "PEC TO BELLY", 
            "PEC TO BODY", "PEC TO FLUKE", "PEC TO GENITAL", "PEC TO MELON", "REEF SHARK", "ROSTRUM TO BODY", "SEA STAR",
            "SHARK", "TAIL SLAP", "JUMPS", "VERTICAL HANG", "VERTICAL SINK"
           }


ACTORS = {"KANSAS", "SELKIE", "REMI", "AMANDA", "SPECKELD", "ONYX", "EEYORE", "NERA", "NAIA",
         "NALA", "NAVEL", "HADES", "JUVENILE", "GROUP", "DOLPHIN", "MAMBA", "POINDEXTER", "REMI", "REMORA", "SAGA",
          "MAMBA", "SHRIMPY", "MOTTLED", "SPOOK", "TWIX", "TOOTHES", "U", "VENTI", "ZEST", 
         }


def find_files(roots, pattern):
    files = []
    for root in roots:
        files.extend(sorted(
            glob.glob(os.path.join(root, "**", pattern),
                      recursive=True)))
    return files


def is_int(x):
    try:
        i = int(x)
        return i
    except:
        return x


def clean_line(x):
    x = re.sub('[0-9?!_]+', '', x).strip()
    x = x.replace('/', 'AND')
    x = x.replace(',', 'AND')
    x = re.sub('[0-9?_]+', '', x).strip()
    x = re.sub(r'AND\s*$', '', x).strip()
    x = x.replace('AND  ', '')
    x = x.replace('  ', '')

    return x 


def is_valid_string(x):
    return type(x) != int and x != 'AND' and x != '' and not pd.isna(x)


def clean(df):
    df = df[df['shotlog::AC'] != 'NOISE'].copy()
    df['shotlog::BEHdescription'] = df['shotlog::BEHdescription'].apply(lambda x: is_int(x))
    df['is_motif'] = df['shotlog::BEHdescription'].apply(lambda x: type(x) == int or pd.isna(x) )
    mask = df.is_motif == False 
    df.loc[mask, 'shotlog::BEHdescription'] = (
        df.loc[mask, 'shotlog::BEHdescription']
        .apply(clean_line)
    )
    df['is_valid_string'] = df['shotlog::BEHdescription'].apply(is_valid_string)
    df = df[['Date', 'ENC #', 'shotlog::AC', 'shotlog::BEHdescription',
             'shotlog::timecode', 'SPECIAL COMMENTS', 'audio_path', 'is_valid_string']]
    return df


def add_path(df, path, audio_map):
    path = path.split('/')[-2]
    df['audio_path'] = audio_map.get(path, 'UNKNOWN').split('/')[-1]
    return df


def audio_map(paths):
    mk_key = lambda x: x.replace('.m4a', '').replace('.wav', '').split('/')[-1]
    return {mk_key(path):path for path in paths}


def context_features(df):
    ctx_key = lambda x: f"is_{x.lower().replace(' ', '_')}"
    for context in CONTEXTS:
        k = ctx_key(context)
        df[k] = df['shotlog::BEHdescription'].apply(lambda x: context in str(x))
    return df 


def actors_features(df):
    actors_key = lambda x: f"has_{x.lower().replace(' ', '_')}"
    for actor in ACTORS:
        k = actors_key(actor)
        df[k] = df['shotlog::BEHdescription'].apply(lambda x: actor in str(x))    
    return df 


def iter_windows(df, window, include_anchor=True):
    for enc, g in df.groupby('ENC #', sort=False):
        tc = g['tc'].to_numpy()
        mask = (g['is_valid_string'] & (g['n_cols_set'] > 0)).to_numpy()
        anchor_pos = np.flatnonzero(mask)
        if len(anchor_pos) == 0:
            continue
        start_pos = np.searchsorted(tc, tc[anchor_pos] - window, side='left')
        for start, end in zip(start_pos, anchor_pos):
            yield enc, g.index[end], g.iloc[start:end + 1 if include_anchor else end]


def features(anchor, context, ctx_index, n_clusters, n_cols):
    y = np.zeros(n_clusters + n_cols)
    for col, pos in ctx_index.items():
        y[pos] = 1 if anchor[col] else 0
    for cid in context['cluster_id'].dropna():
        y[int(cid)] = 1
    return y



def hotcode_df(df, window=pd.Timedelta(seconds=10), keep_empty_context=False):
    df = df.copy()

    # cluster ids as numbers (handles "3" and "3.0")
    df['cluster_id'] = pd.to_numeric(df['shotlog::BEHdescription'], errors='coerce')
    df['is_valid_string'] = df['is_valid_string'].fillna(False).astype(bool)

    ids = df.loc[~df['is_valid_string'], 'cluster_id'].dropna()
    n_clusters = int(ids.max()) + 1 if len(ids) else 0  # ids 0..max -> max+1 slots

    flag_cols = [c for c in df.columns
                 if c.startswith(('is_', 'has_')) and c != 'is_valid_string']
    df[flag_cols] = df[flag_cols].fillna(False).astype(bool)
    ctx_index = {c: n_clusters + i for i, c in enumerate(flag_cols)}
    n_cols = len(ctx_index)

    df['n_cols_set'] = df[flag_cols].sum(axis=1)

    s = (df['shotlog::timecode'].astype('string')
           .str.replace(r'[^0-9:]', '', regex=True)
           .str.rstrip(':'))
    df['tc'] = pd.to_timedelta(s, errors='coerce')
    df = df.sort_values(['ENC #', 'tc'], kind='stable')
    df['tc'] = df.groupby('ENC #')['tc'].ffill()

    rows, anchor_idxs = [], []
    for enc, anchor_idx, win in iter_windows(df, window):
        context = win.iloc[:-1]
        context = context[~context['is_valid_string']]
        if context.empty and not keep_empty_context:
            continue
        rows.append(features(win.iloc[-1], context, ctx_index, n_clusters, n_cols))
        anchor_idxs.append(anchor_idx)

    hotcoded = np.vstack(rows) if rows else np.empty((0, n_clusters + n_cols))
    meta = {
        'column_index': ctx_index,
        'n_clusters': n_clusters,
        'n_cols': n_cols,
        'rows': [int(i) for i in anchor_idxs],
    }
    return hotcoded, meta


    
if __name__ == '__main__':
    print("Insert to mysql")
    if len(sys.argv) < 2:
        print("Usage: python sqlite_converter.py PATH_TO_ANNOTATIONS PATH")
    else:
        print(f"Processing: {sys.argv[1]}")
        root  = [sys.argv[1]]
        audio = [sys.argv[2]]
        files = find_files(root, 'annotated_motif_hits.csv')
        audio = audio_map(find_files(audio, '*.m4a') + find_files(audio, '*.wav')) 
        dataframes = [add_path(pd.read_csv(f), f, audio) for f in files]
        df = clean(pd.concat(dataframes).reset_index())
        df = context_features(df)
        df = actors_features(df)
        
        conn = sqlite3.connect("pvl.db")
        df.to_sql("pvl", conn, if_exists="replace", index=False)
        conn.close()
        df.to_csv('pvl_annotated_audio.csv', index=False)

        hotcoded, hotcoding_meta = hotcode_df(df)
        np.savetxt('hotcoded.out', hotcoded)
        with open('hotcoding_meta.json', 'w') as f:
            json.dump(hotcoding_meta, f)
