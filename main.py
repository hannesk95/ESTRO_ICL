from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score, f1_score
import mlflow
from model import VisionLanguageModel
from sklearn.model_selection import StratifiedKFold
from glob import glob
import os
from tqdm import tqdm
import numpy as np


def main(model_name, task, shots, sampling, decomposition):

    mlflow.log_param("model_name", model_name)
    mlflow.log_param("task", task)
    mlflow.log_param("shots", shots)
    mlflow.log_param("sampling", sampling)
    mlflow.log_param("decomposition", decomposition)

    match task:
        case "sarcoma_binary":
            files = sorted(glob(f"./data/sarcoma/binary/T1FsGd/*{decomposition}.png"))
            files = sorted([f for f in files if not "label" in f])
            labels = [int(os.path.basename(f).split("_")[2]) for f in files]
            labels = [0 if l == 1 else 1 for l in labels]
        case "sarcoma_multiclass":
            files = ...
            labels = ...
        case _:
            raise NotImplementedError(f"Task {task} not implemented.")

    model = VisionLanguageModel(model_name=model_name,
                                shots=shots,
                                sampling=sampling,
                                task=task,
                                decomposition=decomposition)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    bacc_list = []
    mcc_list = []
    f1_list = []
    auroc_list = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(files, labels)):        

        train_files = [files[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_files = [files[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        true_labels = []
        pred_labels = []
        pred_scores = []
        for test_file, test_label in tqdm(zip(test_files, test_labels), total=len(test_files), desc=f"Fold {fold}"):        
            output = model(test_file, train_files, train_labels)

            pred_label = output["answer"]
            pred_label = 0 if pred_label == "low-grade" else 1
            pred_score = float(output["score"])
            pred_score = pred_score if pred_label == 1 else 1 - pred_score

            true_labels.append(test_label)
            pred_labels.append(pred_label)
            pred_scores.append(pred_score)

        # Compute metrics
        bal_acc = balanced_accuracy_score(test_labels, pred_labels)
        mcc = matthews_corrcoef(test_labels, pred_labels)
        f1 = f1_score(test_labels, pred_labels, average="weighted")
        roc_auc = roc_auc_score(test_labels, pred_scores)

        # Log metrics
        mlflow.log_metric(f"fold_{fold}_bcc", bal_acc)
        mlflow.log_metric(f"fold_{fold}_mcc", mcc)
        mlflow.log_metric(f"fold_{fold}_f1-score", f1)
        mlflow.log_metric(f"fold_{fold}_auroc", roc_auc)
    
        bacc_list.append(bal_acc)
        mcc_list.append(mcc)
        f1_list.append(f1)
        auroc_list.append(roc_auc)

    # Log average metrics
    bacc_mean = np.mean(bacc_list)
    bacc_std = np.std(bacc_list)
    mlflow.log_metric("bacc_mean", bacc_mean)
    mlflow.log_metric("bacc_std", bacc_std)

    mcc_mean = np.mean(mcc_list)
    mcc_std = np.std(mcc_list)
    mlflow.log_metric("mcc_mean", mcc_mean)
    mlflow.log_metric("mcc_std", mcc_std)

    f1_mean = np.mean(f1_list)
    f1_std = np.std(f1_list)
    mlflow.log_metric("f1_mean", f1_mean)
    mlflow.log_metric("f1_std", f1_std)
    
    auroc_mean = np.mean(auroc_list)
    auroc_std = np.std(auroc_list)
    mlflow.log_metric("auroc_mean", auroc_mean)
    mlflow.log_metric("auroc_std", auroc_std)
    

if __name__ == "__main__":

    # for model_name in ["google/medgemma-4b-it", "google/gemma-3-4b-it"]:
    for model_name in ["google/medgemma-4b-it", "google/gemma-3-12b-it"]:
        # for task in ["sarcoma_binary", "sarcoma_multiclass"]:
        for task in ["sarcoma_binary"]:
            # for shots in [0, 1, 3, 5, 7, 10, -1]:
            for shots in [0, 3, 5, 10]:
                # for sampling in ["random", "radiomics"]:
                for sampling in ["radiomics_2D", "radiomics_3D", "random", "worst-case_2D", "worst-case_3D"]:
                    # for decomposition in ["axial", "axial+", "mip"]:
                    for decomposition in ["mip", "axial", "axial+"]:

                        # mlflow.set_experiment(f"{task}")
                        mlflow.start_run()
                        main(model_name, task, shots, sampling, decomposition)
                        mlflow.end_run()
