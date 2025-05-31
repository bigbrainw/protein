import os, io, requests, pandas as pd
import re
from Bio import SeqIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1) read your JSON
df = pd.read_json("data.json")

# 2) set up a session to reuse TCP connections
session = requests.Session()

def fetch_wt_seq(uniprot_id):
    """Fetch once per unique ID."""
    url  = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    resp = session.get(url, timeout=10)
    resp.raise_for_status()
    return str(SeqIO.read(io.StringIO(resp.text), "fasta").seq)

# 3) pre-fetch all unique sequences in parallel
unique_ids = df["uniprot"].unique()
wt_seqs = {}

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(fetch_wt_seq, uid): uid for uid in unique_ids}
    for fut in tqdm(as_completed(futures),
                    total=len(futures),
                    desc="Fetching FASTAs",
                    unit="seq"):
        uid = futures[fut]
        try:
            wt_seqs[uid] = fut.result()
        except Exception as e:
            print(f"  → failed to fetch {uid}: {e}")

# 4) mutation function (unchanged)
def apply_mutations(wt_seq, mut_codes):
    seq = wt_seq
    for code in mut_codes.split(','):
        code = code.strip()
        orig, pos, new = code[0], int(code[1:-1]), code[-1]
        if seq[pos-1] != orig:
            raise ValueError(f"Expected {orig}@{pos} but found {seq[pos-1]}")
        seq = seq[:pos-1] + new + seq[pos:]
    return seq

# 5) apply mutations—no more HTTP calls inside this loop!
records = []
for rec in tqdm(df.itertuples(index=False),
                total=len(df),
                desc="Applying mutations",
                unit="mut"):
    uid = rec.uniprot
    try:
        mut_seq = apply_mutations(wt_seqs[uid], rec.mutation_code)
    except Exception as e:
        print(f"Skipping {rec.id}: {e}")
        continue

    pdb = rec.pdb_mutant or rec.PDB_wild
    records.append({
        "id":       rec.id,
        "sequence": mut_seq,
        "pdb_file": f"{pdb}.pdb",
        "ddg":      rec.ddg
    })

# 6) write out as before
labels = pd.DataFrame(records, columns=["id","sequence","pdb_file","ddg"])
labels.to_csv("labels.csv", index=False)
