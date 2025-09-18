import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from rollout import generate_rollout
from patient_data import DataMatrix, PatientRecord


class ModelInfer:
    '''class containing all the info about the training process and handling the actual
    training function'''

    def __init__(
            self,
            model,
            dataloaders,
            class_count,
            device,
            save_path):
        self.model = model
        self.dataloaders = dataloaders
        self.device = device
        self.save_path = save_path
        self.class_count = class_count
        self.data_obj = DataMatrix()

    def launch_infering(self):
        '''initializes training process.'''
        #Save logits for test
        test_metrics = self._infer_split('test')

        #self.data_obj.save_hdf5(self.save_path / "patient_data.h5")
        return self.model, test_metrics['conf_matrix']

    def _infer_split(
        self,
        split: str,
        *,
        save_logits: bool = True,
    ) -> Dict[str, float]:
        self.model.eval()
        dataloader = self.dataloaders[split]

        preds, label_preds, labels, patients = [], [], [], []
        running_loss = 0.0
        with torch.no_grad():
            for bag, label, img_paths, patient_id in tqdm(dataloader, desc=f"Inference {split}"):
                patient_id = patient_id[0] 
                bag   = bag.to(self.device)
                label = label.to(self.device)

                latent, logits = self.model(bag, return_latent=True)
                loss = nn.CrossEntropyLoss()(logits, label)
                running_loss += loss.item() * bag.size(0)

                cls_attention_scores = generate_rollout(self.model,bag,start_layer=0)
                cls_attention_scores = cls_attention_scores.squeeze(0)

                label_prediction = torch.argmax(logits, dim=1).item()
                label_groundtruth = label.item()

                self.data_obj.add_patient(
                        patient_id,
                        PatientRecord.from_tensors(
                        true_label=label_groundtruth,
                        pred_label=label_prediction,
                        latent=latent,
                        attention=cls_attention_scores,
                        image_paths=img_paths,
                        prediction_vector=logits,
                        loss=loss,
                    ),
                )

                label_preds.append(label_prediction)
                labels.append(label_groundtruth)
                preds.append(logits)
                patients.append(patient_id)


        preds = torch.cat(preds, dim=0).detach().cpu().numpy()

        epoch_loss = running_loss / len(dataloader.dataset)
        acc = accuracy_score(labels, label_preds)
        bal_acc = balanced_accuracy_score(labels, label_preds)
        f1_weighted = f1_score(labels, label_preds, average='weighted')
        conf_matrix = confusion_matrix(labels, label_preds, labels=list(range(self.class_count)))

        print(f"{split}: loss={epoch_loss:.4f} acc={acc:.3f} balAcc={bal_acc:.3f} f1w={f1_weighted:.3f}")

        if save_logits:
            pd.DataFrame({
                "patient": patients,
                "label": labels,
                "prediction": [list(p) for p in preds],
            }).to_csv(self.save_path / f"metadata_results_{split}.csv", index=False)

        return {
            "loss": epoch_loss,
            "acc": acc,
            "balanced_acc": bal_acc,
            "f1": f1_weighted,
            "conf_matrix": conf_matrix,
        }