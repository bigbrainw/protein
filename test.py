import pandas as pd
import re

def convert_full_ddg_dataset(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Drop rows missing critical info
    df = df.dropna(subset=['sequence', 'wild_type', 'mutation', 'position', 'ddG'])

    # Construct mutation string like "A123T"
    df['mutation_str'] = df.apply(
        lambda row: f"{row['wild_type']}{int(row['position'])}{row['mutation']}", axis=1
    )

    # Output compatible DataFrame
    out_df = df[['uniprot_id', 'sequence', 'mutation_str', 'ddG']].copy()
    out_df.columns = ['id', 'wildtype_seq', 'mutation', 'ddg']
    
    # Optional: filter invalid residues
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    out_df = out_df[
        out_df['wildtype_seq'].apply(lambda x: set(str(x)).issubset(valid_aas)) &
        out_df['mutation'].apply(lambda x: re.match(r"^[A-Z]\d+[A-Z]$", str(x)))
    ]

    out_df.to_csv(output_csv, index=False)
    print(f"✅ Saved converted dataset to: {output_csv}")

# Example usage:
convert_full_ddg_dataset("Lysosome.csv", "converted_ddg.csv")
