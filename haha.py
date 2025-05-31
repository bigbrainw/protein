import os
import re
import io
import requests
import pandas as pd
from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# 1) Load your JSON
df = pd.read_json("data.json")

# 2) Drop any malformed mutation codes up‐front
code_re = re.compile(r'^([A-Z]\d+[A-Z])(,[A-Z]\d+[A-Z])*$')
mask = df['mutation_code'].astype(str).apply(lambda s: bool(code_re.match(s)))
bad = df.loc[~mask, 'mutation_code'].unique()
if len(bad):
    print("Dropping malformed codes:", bad)
df = df.loc[mask].reset_index(drop=True)

# 3) Helper to extract valid PDB IDs from a string
def extract_pdb_ids(s):
    if not isinstance(s, str):
        return []
    # split on any non-alphanumeric
    tokens = re.split(r'[^0-9A-Za-z]+', s)
    # PDB IDs are 4-char, start with digit
    return [tok.upper() for tok in tokens if re.match(r'^[0-9][A-Za-z0-9]{3}$', tok)]

# 4) Prepare HTTP session & PDB dir
session = requests.Session()
pdb_dir = "pdbs"
os.makedirs(pdb_dir, exist_ok=True)

# 5) Gather all PDB IDs you’ll need
pdb_ids = set()
for rec in df.itertuples(index=False):
    for field in (rec.pdb_mutant, rec.PDB_wild):
        pdb_ids.update(extract_pdb_ids(field))

# 6) Download each PDB once
for pid in tqdm(sorted(pdb_ids), desc="Downloading PDBs"):
    out = os.path.join(pdb_dir, f"{pid}.pdb")
    if os.path.exists(out):
        continue
    url = f"https://files.rcsb.org/download/{pid}.pdb"
    try:
        r = session.get(url, timeout=10); r.raise_for_status()
        with open(out, "wb") as f:
            f.write(r.content)
    except Exception as e:
        print(f"  → failed to download {pid}: {e}")

# 7) Fetch wild‐type sequences from UniProt in parallel
def fetch_uniprot(uid):
    fasta_url = f"https://www.uniprot.org/uniprot/{uid}.fasta"
    r = session.get(fasta_url, timeout=10); r.raise_for_status()
    return str(SeqIO.read(io.StringIO(r.text), "fasta").seq)

wt_seqs = {}
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(fetch_uniprot, uid): uid for uid in df["uniprot"].unique()}
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching FASTAs"):
        uid = futures[fut]
        try:
            wt_seqs[uid] = fut.result()
        except Exception as e:
            print(f"  → failed to fetch {uid}: {e}")

# 8) (Optional) helper to grab sequence straight from the PDB
def parse_pdb_seq(pdb_path, chain_id="A"):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("", pdb_path)
    ppb = PPBuilder()
    if chain_id in struct[0]:
        peptides = ppb.build_peptides(struct[0][chain_id])
    else:
        peptides = ppb.build_peptides(next(struct[0].get_chains()))
    if not peptides:
        raise ValueError(f"No peptide in {pdb_path}")
    return "".join(str(pp.get_sequence()) for pp in peptides)

# 9) mutation‐applying function
def apply_mutations(seq, codes):
    s = seq
    for code in codes.split(","):
        orig, pos, new = code[0], int(code[1:-1]), code[-1]
        if s[pos-1] != orig:
            raise ValueError(f"Expected {orig}@{pos} but found {s[pos-1]}")
        s = s[:pos-1] + new + s[pos:]
    return s

# 10) Build labels, logging skip‐reasons
errors = Counter()
records = []

for rec in tqdm(df.itertuples(index=False), desc="Processing", total=len(df)):
    # extract all possible PDB IDs for this record
    ids = extract_pdb_ids(rec.pdb_mutant) or extract_pdb_ids(rec.PDB_wild)
    if not ids:
        errors['no_pdb_field'] += 1
        continue

    # pick the first valid pdb id that we downloaded
    pdb_file = None
    for pid in ids:
        candidate = os.path.join(pdb_dir, f"{pid}.pdb")
        if os.path.exists(candidate):
            pdb_file = candidate
            break
    if not pdb_file:
        errors['missing_download'] += 1
        continue

    # get wild‐type sequence
    wt = wt_seqs.get(rec.uniprot)
    if wt is None:
        errors['no_uniprot'] += 1
        continue

    # if you prefer PDB‐derived sequence, uncomment:
    # wt = parse_pdb_seq(pdb_file)

    # apply mutations
    try:
        mut_seq = apply_mutations(wt, rec.mutation_code)
    except Exception:
        errors['mismatch'] += 1
        continue

    records.append({
        "id":       rec.id,
        "sequence": mut_seq,
        "pdb_file": pdb_file,
        "ddg":      rec.ddg
    })

print("Summary of skips:", dict(errors))

# 11) Save out
labels = pd.DataFrame(records, columns=["id","sequence","pdb_file","ddg"])
labels.to_csv("labels.csv", index=False)
