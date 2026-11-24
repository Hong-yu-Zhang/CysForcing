#!/usr/bin/env python
"""CysForcing in the browser: the three models are loaded once onto one GPU and a page is served on top.

    python webapp/server.py --gpu 0 --port 7860 --saprot /path/to/SaProt_650M_AF2

then open http://localhost:7860. POST /api/predict takes JSON, either {"uniprot": "P78417"}, for which the
AlphaFold DB model is fetched, or {"pdb": "<PDB file text>", "name": "x.pdb"}; it returns one probability per
cysteine, computed exactly as simple_inference.py does. Only the standard library is used on top of the
packages CysForcing already needs.
"""
import argparse
import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def parse_args():
    p = argparse.ArgumentParser(description="CysForcing web interface (one GPU)")
    p.add_argument("--gpu", default=None, help="GPU index to run on; sets CUDA_VISIBLE_DEVICES")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept other machines")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--model", default=os.path.join(ROOT, "configs/model/CysForcing.yaml"))
    p.add_argument("--weight", default=os.path.join(ROOT, "weights/CysForcing.pth"))
    p.add_argument("--saprot", default=os.path.join(ROOT, "SaProt_650M_AF2"),
                   help="folder holding SaProt_650M_AF2/ and bin/foldseek")
    p.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "cysforcing_web"),
                   help="scratch folder for uploads and fetched AlphaFold DB models")
    return p.parse_args()


ARGS = parse_args()
if ARGS.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu
for k in ("model", "weight", "saprot", "cache"):
    setattr(ARGS, k, os.path.abspath(getattr(ARGS, k)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402  (CUDA_VISIBLE_DEVICES must be set first)
from Bio import SeqIO  # noqa: E402

import simple_inference as si  # noqa: E402
from configs import get_cfg_defaults  # noqa: E402
from models import CysForcing  # noqa: E402
from utils import set_seed  # noqa: E402

MAX_RES = 1023                 # ESM-2 sees 1,024 tokens and SaProt 2,046 characters: both keep 1,023 residues
THRESHOLD = 0.5                # the operating point of the paper
MAX_BODY = 12 * 1024 * 1024
UNIPROT = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$")
AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{}"
STATIC = {
    "/": (os.path.join(HERE, "static", "index.html"), "text/html; charset=utf-8"),
    "/app.css": (os.path.join(HERE, "static", "app.css"), "text/css; charset=utf-8"),
    "/app.js": (os.path.join(HERE, "static", "app.js"), "application/javascript; charset=utf-8"),
    "/favicon.png": (os.path.join(ROOT, "docs", "static", "favicon.png"), "image/png"),
    "/grain.png": (os.path.join(ROOT, "docs", "static", "grain.png"), "image/png"),
}


class InputError(Exception):
    """A problem with what the user sent; the message is shown to them as is."""


class Predictor:
    def __init__(self):
        cfg = get_cfg_defaults()
        cfg.merge_from_file(ARGS.model)
        set_seed(cfg.SOLVER.SEED)
        t0 = time.time()
        self.esm = si.ESM_model()
        self.saprot = si.SAProt_model(model_folder=ARGS.saprot)
        self.model = CysForcing(**cfg)
        self.model.load_state_dict(torch.load(ARGS.weight, map_location="cpu"))
        self.model.to(si.device).eval()
        self.lock = threading.Lock()
        self.waiting = 0
        self.gpu = torch.cuda.get_device_name(si.device) if si.device.type == "cuda" else "CPU"
        print(f"models loaded on {self.gpu} in {time.time() - t0:.1f}s", flush=True)

    def predict(self, pdb_path):
        seq = si.extract_sequence_from_pdb(pdb_path)
        seq_source = "SEQRES"
        if not seq:      # files without SEQRES records (e.g. ColabFold): read the modelled residues of chain A
            seq = next((str(r.seq) for r in SeqIO.parse(pdb_path, "pdb-atom") if r.annotations.get("chain") == "A"), "")
            seq_source = "ATOM records"
        seq = seq.upper()
        if not seq or "C" not in seq:
            raise InputError("No cysteine found in chain A of this structure.")
        self.waiting += 1
        try:
            with self.lock:
                t0 = time.time()
                esm_feat = self.esm(seq)
                try:
                    saprot_feat = self.saprot(pdb_path)
                except Exception:
                    raise InputError("foldseek could not read chain A of this file; is it a PDB-format protein model?")
                if esm_feat.shape[0] != saprot_feat.shape[0]:
                    raise InputError(
                        f"The sequence ({esm_feat.shape[0]} residues) and the modelled residues of chain A "
                        f"({saprot_feat.shape[0]}) differ. CysForcing expects a full-length predicted model, "
                        "such as AlphaFold2's.")
                with torch.no_grad():
                    prob = si.Inference_one(self.model, {"ESM": esm_feat, "SAProt": saprot_feat})
                prob = prob.view(-1).cpu().tolist()
                if si.device.type == "cuda":
                    torch.cuda.synchronize()
                return seq, seq_source, prob, time.time() - t0
        finally:
            self.waiting -= 1


def plddt_by_residue(pdb_text):
    """B-factor of each C-alpha of chain A, in order; AlphaFold writes its pLDDT there."""
    out = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == "A":
            try:
                out.append(float(line[60:66]))
            except ValueError:
                return None
    return out


def fetch_afdb(acc):
    path = os.path.join(ARGS.cache, f"AF-{acc}-F1.pdb")
    meta_path = path + ".json"
    if os.path.exists(path) and os.path.exists(meta_path):
        with open(meta_path) as f:
            return path, json.load(f)
    try:
        with urllib.request.urlopen(AFDB_API.format(acc), timeout=30) as r:
            entries = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (400, 404, 422):
            raise InputError(f"AlphaFold DB has no model for {acc}.")
        raise InputError(f"AlphaFold DB did not answer (HTTP {e.code}); try again, or upload the PDB file.")
    except (urllib.error.URLError, TimeoutError):
        raise InputError("AlphaFold DB could not be reached from the server; upload the PDB file instead.")
    entry = next((e for e in entries if e.get("uniprotAccession") == acc), entries[0] if entries else None)
    url = (entry or {}).get("pdbUrl", "")
    if urllib.parse.urlparse(url).netloc != "alphafold.ebi.ac.uk":
        raise InputError(f"AlphaFold DB has no PDB model for {acc}.")
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    meta = {
        "file": os.path.basename(urllib.parse.urlparse(url).path),
        "entry": entry.get("entryId"),
        "gene": entry.get("gene"),
        "description": entry.get("uniprotDescription"),
        "organism": entry.get("organismScientificName"),
        "version": entry.get("latestVersion"),
    }
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return path, meta


def run(payload):
    acc = str(payload.get("uniprot") or "").strip().upper()
    pdb_text = payload.get("pdb")
    if acc:
        if not UNIPROT.match(acc):
            raise InputError(f"'{acc[:20]}' is not a UniProt accession (e.g. P78417).")
        path, meta = fetch_afdb(acc)
        with open(path) as f:
            pdb_text = f.read()
        name, source, tmp = meta["file"], "AlphaFold DB", None
    elif isinstance(pdb_text, str) and pdb_text.strip():
        if "\nATOM  " not in "\n" + pdb_text:
            raise InputError("This does not look like a PDB file (no ATOM records).")
        # the file name never reaches the disk or the foldseek command line
        tmp = os.path.join(ARGS.cache, f"upload_{uuid.uuid4().hex}.pdb")
        with open(tmp, "w") as f:
            f.write(pdb_text)
        path, meta, source = tmp, {}, "upload"
        name = re.sub(r"[^\w.\-]+", "_", os.path.basename(str(payload.get("name") or "structure.pdb")))[:80]
    else:
        raise InputError("Send a UniProt accession or a PDB file.")
    try:
        seq, seq_source, prob, secs = PREDICTOR.predict(path)
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)

    scored = seq[:MAX_RES]
    plddt = plddt_by_residue(pdb_text) if source == "AlphaFold DB" else None
    if plddt is not None and len(plddt) < len(scored):
        plddt = None
    sites = []
    for i, aa in enumerate(seq):
        if aa != "C":
            continue
        p = prob[i] if i < len(scored) else None
        sites.append({
            "pos": i + 1,
            "prob": p,
            "call": None if p is None else p >= THRESHOLD,
            "left": seq[max(0, i - 7):i],
            "right": seq[i + 1:i + 8],
            "plddt": plddt[i] if plddt else None,
        })
    # the same record simple_inference.py writes to result.json, one-element lists included
    cli = {f"C_{i + 1}": [prob[i]] for i, aa in enumerate(scored) if aa == "C"}
    cli["Sequence"] = scored
    return {
        "name": name,
        "source": source,
        "meta": meta,
        "sequence": seq,
        "sequence_source": seq_source,
        "length": len(seq),
        "scored_length": len(scored),
        "threshold": THRESHOLD,
        "sites": sites,
        "seconds": round(secs, 2),
        "device": PREDICTOR.gpu,
        "result_json": [{name: cli}],
        "pdb": pdb_text if source == "AlphaFold DB" else None,   # an upload is already in the browser
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "CysForcing"

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = urllib.parse.urlparse(self.path).path
        if route == "/api/health":
            return self._send(200, {"ok": True, "device": PREDICTOR.gpu, "waiting": PREDICTOR.waiting,
                                    "max_residues": MAX_RES, "threshold": THRESHOLD})
        if route in STATIC:
            path, ctype = STATIC[route]
            with open(path, "rb") as f:
                return self._send(200, f.read(), ctype)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/api/predict":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                raise InputError(f"The request must be between 1 byte and {MAX_BODY // 2**20} MB.")
            try:
                payload = json.loads(self.rfile.read(n))
            except ValueError:
                raise InputError("The request body is not JSON.")
            if not isinstance(payload, dict):
                raise InputError("The request body must be a JSON object.")
            self._send(200, run(payload))
        except InputError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # keep the server up, and say what failed
            print(f"error: {type(e).__name__}: {e}", flush=True)
            self._send(500, {"error": f"Prediction failed ({type(e).__name__}). See the server log."})

    def log_message(self, fmt, *a):
        print(f"{self.address_string()} {fmt % a}", flush=True)


if __name__ == "__main__":
    os.makedirs(ARGS.cache, exist_ok=True)
    os.chdir(ARGS.cache)          # foldseek_util writes its scratch files into the working directory
    httpd = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)   # bind first: a busy port fails before the models load
    PREDICTOR = Predictor()
    print(f"CysForcing is serving on http://{ARGS.host}:{ARGS.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
