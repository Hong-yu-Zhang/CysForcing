import copy
import os

import torch
from prettytable import PrettyTable
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from tqdm import tqdm

from models import binary_cross_entropy, cysforcing_loss

class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, output, alpha=1, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.alpha = alpha
        self.n_class = config["DECODER"]["NUM_CLASSES"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0

        self.best_model = None
        self.best_epoch = None
        self.best_f1 = 0
        self.best_mcc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = output

        self.align_weight = config["LOSS"]["ALIGN_WEIGHT"]
        self.ratio_weight = config["LOSS"]["RATIO_WEIGHT"]
        self.temp = config["LOSS"]["TEMP"]

        valid_metric_header = ["# Epoch", "F1", "MCC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "MCC", "ACCURACY", "PRECISION", "RECALL", "F1", "Test_loss"]
        
        train_metric_header = ["# Epoch", "Train_loss"]
        
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            
            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            auroc, auprc, accuracy, mcc, precision, recall, f1, val_loss = self.test(dataloader="val")
        
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [f1, mcc, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            if mcc >= self.best_mcc and f1 >= self.best_f1:
                self.best_model = copy.deepcopy(self.model)
                self.best_f1 = f1
                self.best_mcc = mcc
                self.best_epoch = self.current_epoch
                torch.save(self.best_model.state_dict(), os.path.join(self.output_dir, f"best_model_epoch.pth"))      
            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss: ' + str(val_loss), " AUROC: "
                  + str(auroc) + " AUPRC: " + str(auprc) + ' Accuracy: ' + str(accuracy) + ' MCC: ' + str(mcc) + ' Precison: ' + str(precision) + ' Recall: ' + str(recall)
                  + ' F1: ' + str(f1))
        auroc, auprc, accuracy, mcc, precision, recall, f1, test_loss = self.test(dataloader="test")
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, mcc, accuracy, precision, recall, f1, test_loss]))
        self.test_table.add_row(test_lst)
        print('Test at Epoch ' + str(self.best_epoch) + ' with test loss: ' + str(test_loss), " AUROC: "
                  + str(auroc) + " AUPRC: " + str(auprc) + ' Accuracy: ' + str(accuracy) + ' MCC: ' + str(mcc) + ' Precison: ' + str(precision) + ' Recall: ' + str(recall)
                  + ' F1: ' + str(f1))
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics['mcc'] = mcc
        self.test_metrics['accuracy'] = accuracy
        self.test_metrics["best_epoch"] = self.best_epoch
        
        self.save_result()
    
        return self.test_metrics

    def save_result(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        torch.save(self.best_model.state_dict(),
                   os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
       
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        loss_cl_epoch = 0
        loss_ratio_epoch = 0
        num_batches = len(self.train_dataloader)
        
        for esm, saprot, labels, ratios, _cys_mask in tqdm(self.train_dataloader):
            self.step += 1
            esm, saprot = esm.to(self.device), saprot.to(self.device)
            labels = labels.float().to(self.device)
            ratios = ratios.float().to(self.device)

            self.optim.zero_grad()
            output = self.model(esm, saprot)
            binary_list = output['binary']
            loss = 0

            # The measured ratio is non-zero exactly at measured Cys sites; the
            # fused head and both auxiliary heads are supervised on those.
            mask = ratios.bool()
            for binary in binary_list:
                binary_at_cys = torch.masked_select(binary.squeeze(-1), mask)
                labels_at_cys = torch.masked_select(labels, mask)
                _, loss_sub = binary_cross_entropy(binary_at_cys, labels_at_cys)
                loss += loss_sub

            esm_feat, saprot_feat = output['feat'][0], output['feat'][1]
            esm_feat = esm_feat[mask.unsqueeze(-1).expand_as(esm_feat)].view(-1, esm_feat.size(-1))
            saprot_feat = saprot_feat[mask.unsqueeze(-1).expand_as(saprot_feat)].view(-1, saprot_feat.size(-1))
            ratio = torch.masked_select(ratios, mask)

            align_loss, cl_loss, ratio_loss = cysforcing_loss(
                esm_feat, saprot_feat, ratio,
                temp=self.temp, lambda_ratio=self.ratio_weight)

            loss = loss + self.align_weight * align_loss

            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            loss_cl_epoch += cl_loss.item()
            loss_ratio_epoch += ratio_loss.item()

        loss_epoch = loss_epoch / num_batches
        loss_cl_epoch = loss_cl_epoch / num_batches
        loss_ratio_epoch = loss_ratio_epoch / num_batches
        print(f'Epoch {self.current_epoch}: loss {loss_epoch:.4f} | '
              f'contrastive {loss_cl_epoch:.4f} | ratio prior {loss_ratio_epoch:.4f}')
        return loss_epoch

    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for esm, saprot, labels, ratios, _cys_mask in tqdm(data_loader):
                esm, saprot = esm.to(self.device), saprot.to(self.device)
                labels = labels.float().to(self.device)
                ratios = ratios.float().to(self.device)

                net = self.model if dataloader == "val" else self.best_model
                binary = net(esm, saprot)['binary'][-1].squeeze(-1)

                mask = ratios.bool()
                binary_at_cys = torch.masked_select(binary, mask)
                labels_at_cys = torch.masked_select(labels, mask)
                _, loss = binary_cross_entropy(binary_at_cys, labels_at_cys)
                test_loss += loss.item()

                y_label.extend(labels_at_cys.to("cpu").tolist())
                y_pred.extend(binary_at_cys.to("cpu").tolist())

        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        # Thresholded before the sigmoid, matching how the published numbers
        # were produced; see the note in reproduce/eval_homology.py.
        y_pred_s = [1 if i > 0.5 else 0 for i in y_pred]
        balanced_acc = balanced_accuracy_score(y_label, y_pred_s)
        mcc = matthews_corrcoef(y_label, y_pred_s)
        precision = precision_score(y_label, y_pred_s)
        recall = recall_score(y_label, y_pred_s)
        f1 = f1_score(y_label, y_pred_s)

        test_loss = test_loss / num_batches

        auroc = round(auroc, 4)
        auprc = round(auprc, 4)
        accuracy = round(balanced_acc, 4)
        mcc = round(mcc, 4)
        test_loss = round(test_loss, 4)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)

        return auroc, auprc, accuracy, mcc, precision, recall, f1, test_loss