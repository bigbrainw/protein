from flask import Flask, request, jsonify, send_file
import torch
import numpy as np
import os
import uuid
from transformers import BertTokenizer, BertModel
from flask_cors import CORS

# === (1) OpenFold imports ===
# You need to have `openfold` installed (pip install openfold)
from openfold.inference import AFRunner

# === Flask Setup ===
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORS(app)

# === Load ProtBert + MLP + entropy as before ===
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
protbert = BertModel.from_pretrained(
    "Rostlab/prot_bert", output_hidden_states=True
).to(device).eval()

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.model(x).squeeze()

mlp = MLP()
mlp.load_state_dict(torch.load("mlp_model.pt", map_location=device))
mlp.to(device).eval()

entropy_dict = np.load("entropy.npy", allow_pickle=True).item()

def predict_ddg(mutated_seq, pos):
    tokens = tokenizer(" ".join(mutated_seq), return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        out = protbert(**tokens)
        embedding = out.last_hidden_state[0, pos+1].unsqueeze(0)
        return mlp(embedding).item()

def generate_candidates(sequence, top_k=5):
    results = []
    for pos, ent in sorted(entropy_dict.items(), key=lambda x: -x[1])[:top_k]:
        if pos >= len(sequence): continue
        wt = sequence[pos]
        best = None
        for alt in "ACDEFGHIKLMNPQRSTVWY":
            if alt == wt: continue
            seq2 = list(sequence)
            seq2[pos] = alt
            try:
                ddg = predict_ddg("".join(seq2), pos)
                cand = {
                    "mutation": f"{wt}{pos+1}{alt}",
                    "position": pos+1,
                    "wt": wt,
                    "alt": alt,
                    "entropy": round(ent,4),
                    "ddg": round(ddg,4)
                }
                if best is None or cand["ddg"] < best["ddg"]:
                    best = cand
            except:
                pass
        if best:
            results.append(best)
    return results

@app.route("/suggest_mutations", methods=["POST"])
def suggest_mutations():
    data = request.get_json()
    seq = data.get("sequence")
    if not seq:
        return jsonify({"error": "Missing 'sequence'"}), 400
    return jsonify({"candidates": generate_candidates(seq)})

# === (2) Instantiate an OpenFold AFRunner ===
# You'll need:
#   • weight file: openfold_weights.pt
#   • a data dir with any required lookups (can be empty if you skip MSA)
OPENFOLD_DATA_DIR = "/path/to/openfold_data"
OPENFOLD_WEIGHTS   = "/path/to/openfold_weights.pt"

runner = AFRunner(
    model_params_path=OPENFOLD_WEIGHTS,
    data_dir=OPENFOLD_DATA_DIR,
    use_precomputed_msas=True  # if you just want single‐sequence inference
)

# === (3) New endpoint to predict 3D structure ===
@app.route("/predict_structure", methods=["POST"])
def predict_structure():
    data = request.get_json()
    seq = data.get("sequence")
    if not seq:
        return jsonify({"error": "Missing 'sequence'"}), 400

    # write a FASTA for OpenFold
    job_id = str(uuid.uuid4())
    out_dir = os.path.join("/tmp", job_id)
    os.makedirs(out_dir, exist_ok=True)
    fasta_path = os.path.join(out_dir, f"{job_id}.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">query\n{seq}\n")

    # run the prediction (this is blocking; for production wrap in Celery/RQ)
    prediction_result = runner.predict(fasta_path, out_dir)
    pdb_path = os.path.join(out_dir, "pred_0.pdb")

    if not os.path.exists(pdb_path):
        return jsonify({"error": "AlphaFold run failed"}), 500

    # return the PDB file directly
    return send_file(pdb_path, mimetype="chemical/x-pdb")

if __name__ == "__main__":
    app.run(debug=True)
