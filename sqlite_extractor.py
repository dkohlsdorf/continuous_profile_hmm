import sqlite3
import sys
import os
import glob
import pandas as pd
import re


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
