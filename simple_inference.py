from models import CysForcing
from time import time
from utils import set_seed
from configs import get_cfg_defaults
import torch
import argparse
import warnings, os
import pandas as pd
from Bio import SeqIO
import glob
from transformers import EsmTokenizer, EsmForMaskedLM
from torch import nn
import torch.nn.functional as F
from foldseek_util import get_struc_seq
from tqdm import tqdm
import esm
import json


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu') 

def parse_args():
    parser = argparse.ArgumentParser(description="CysForcing single-shot inference")
    parser.add_argument('--model', required=True, help="path to model config file", type=str)
    parser.add_argument('--weight', required=True, help="path to model weight", type=str)
    parser.add_argument('--pdb_folder', required=True, help="path to pdb files", type=str)
    parser.add_argument('--out_dir', default='./result_inference',
                        help="output folder (default: ./result_inference); per-Cys probs are written to <out_dir>/result.json")
    parser.add_argument('--out_name', default='result.json',
                        help="output JSON filename inside --out_dir (default: result.json)")
    parser.add_argument('--saprot', default='./SaProt_650M_AF2',
                        help="folder holding SaProt_650M_AF2/ and bin/foldseek")
    return parser.parse_args()

class ESM_model(nn.Module):
    def __init__(self, ):
        super(ESM_model, self).__init__()
        self.device = device
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model = self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

    def forward(self, seq):
        tmp = []
        tmp.append(('',seq))
        data = tmp
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens[:, 0:1024]
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        token_representations = token_representations.squeeze(dim=0)
        feat = token_representations[1:len(seq)+1, :]
        feat = feat.cpu().detach()
        
        return feat
    
class SAProt_model(nn.Module):
    def __init__(self, model_folder):
        super(SAProt_model, self).__init__()
        model_path = os.path.join(model_folder, "SaProt_650M_AF2")
        self.tokenizer = EsmTokenizer.from_pretrained(model_path)
        self.model = EsmForMaskedLM.from_pretrained(model_path)
        self.foldseek_path = os.path.join(model_folder, "bin/foldseek")
        self.model.eval()
        self.model.to(device)
    
    def forward(self, pdb_file):
        parsed_seqs = get_struc_seq(self.foldseek_path, pdb_file, ["A"], plddt_mask=False)["A"]
        _, _, seq = parsed_seqs 
        seq = seq[:2046]
        inputs = self.tokenizer(seq, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["output_hidden_states"] = True
        with torch.no_grad():
            outputs = self.model(**inputs)
        feat = outputs["hidden_states"][-1]
        feat = feat.squeeze(dim=0)
        feat = feat[1:len(seq)//2 + 1, :]
        feat = feat.cpu().detach()

        return feat


def extract_sequence_from_pdb(pdb_file):
    for record in SeqIO.parse(pdb_file, "pdb-seqres"):
        return str(record.seq)
    return None

def embedding(pdb_folder, saprot_path='./SaProt_650M_AF2'):
    pdb_file_paths = []
    
    pdb_file_paths = glob.glob(f"{pdb_folder}/*.pdb", recursive=True)

    esm_model = ESM_model()
    saprot_model = SAProt_model(model_folder=saprot_path)

    embedding_list = []

    print('Embedding for SAProt & ESM start!')
    for pdb_path in tqdm(pdb_file_paths):
        pdb = pdb_path
        seq = extract_sequence_from_pdb(pdb_path)

        print(pdb_path)

        esm_feat = esm_model(seq)
        saprot_feat = saprot_model(pdb)

        embedding_list.append({"Name": pdb.split('/')[-1],
                               "Sequence": seq[:1023],
                               "ESM": esm_feat,
                               "SAProt": saprot_feat}
                            )
    return embedding_list

def Inference_one(model, data):
    saprot_feat = data['SAProt'].to(device)
    esm_feat = data['ESM'].to(device)
    result = model(esm_feat, saprot_feat, None)
    binary = result['binary'][-1]
    return binary.sigmoid()

def Inference(pdb_folder, weight_path, model_cfg, saprot_path='./SaProt_650M_AF2',
              result_folder='./result_inference', result_name='result.json'):
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(model_cfg)
    set_seed(cfg.SOLVER.SEED)

    print(f"Model Config: {model_cfg}")

    feat_list = embedding(pdb_folder=pdb_folder, saprot_path=saprot_path)

    model = CysForcing(**cfg)
    torch.backends.cudnn.benchmark = True

    model.load_state_dict(torch.load(weight_path))
    model.to(device)

    result_list = []
    for feat in feat_list:
        binary = Inference_one(model, feat)
        binary = binary.cpu().detach().tolist()
        seq = feat['Sequence']
        name = feat['Name']
        c_scores = {f"C_{i+1}": score for i, (residue, score) in enumerate(zip(seq, binary)) if residue == "C"}
        c_scores['Sequence'] = seq 
        result_list.append({name: c_scores})

    os.makedirs(result_folder, exist_ok=True)
    output_file = os.path.join(result_folder, result_name)
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(result_list, json_file, indent=4, ensure_ascii=False)
    print(f"Wrote predictions: {output_file}")
    return output_file


if __name__ == '__main__':
    args = parse_args()
    s = time()
    Inference(
        pdb_folder=args.pdb_folder,
        weight_path=args.weight,
        model_cfg=args.model,
        saprot_path=args.saprot,
        result_folder=args.out_dir,
        result_name=args.out_name,
    )
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
