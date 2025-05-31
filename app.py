from flask import Flask, request, jsonify
import torch
import numpy as np
import re
from transformers import BertTokenizer, BertModel
from flask_cors import CORS
# === Flask Setup ===
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORS(app)
# === Load Models ===
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
protbert = BertModel.from_pretrained("Rostlab/prot_bert", output_hidden_states=True).to(device).eval()

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

# === Load Entropy Map ===
entropy_dict = np.load("entropy.npy", allow_pickle=True).item()

# === Core Logic ===
def predict_ddg(mutated_seq, pos):
    tokens = tokenizer(" ".join(mutated_seq), return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        output = protbert(**tokens)
        embedding = output.last_hidden_state[0, pos + 1].unsqueeze(0)  # skip CLS
        ddg = mlp(embedding).item()
    return ddg

def generate_candidates(sequence, top_k=5):
    results = []
    for pos, entropy in sorted(entropy_dict.items(), key=lambda x: -x[1])[:top_k]:
        if pos >= len(sequence):
            continue
        wt_aa = sequence[pos]
        best = None
        for alt in "ACDEFGHIKLMNPQRSTVWY":
            if alt == wt_aa:
                continue
            mutated = list(sequence)
            mutated[pos] = alt
            try:
                ddg = predict_ddg("".join(mutated), pos)
                cand = {
                    "mutation": f"{wt_aa}{pos+1}{alt}",
                    "position": pos + 1,
                    "wt": wt_aa,
                    "alt": alt,
                    "entropy": round(entropy, 4),
                    "ddg": round(ddg, 4)
                }
                if best is None or ddg < best["ddg"]:
                    best = cand
            except Exception:
                continue
        if best:
            results.append(best)
    return results

# === Route ===
@app.route("/suggest_mutations", methods=["POST"])
def suggest_mutations():
    try:
        data = request.get_json()
        sequence = data.get("sequence")
        if not sequence:
            return jsonify({"error": "Missing 'sequence'"}), 400

        candidates = generate_candidates(sequence)
        return jsonify({"candidates": candidates})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Run Server ===
if __name__ == "__main__":
    app.run(debug=True)
