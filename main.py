from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score, f1_score
import mlflow
from model import VisionLanguageModel
from sklearn.model_selection import StratifiedKFold
from glob import glob
import os
from tqdm import tqdm
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def main(model_name, task, shots, sampling, decomposition, dataset_fraction):

    mlflow.log_param("model_name", model_name)
    mlflow.log_param("task", task)
    mlflow.log_param("shots", shots)
    mlflow.log_param("sampling", sampling)
    mlflow.log_param("decomposition", decomposition)
    mlflow.log_param("dataset_fraction", dataset_fraction)

    match task:
        case "sarcoma_binary_t1":
            files = sorted(glob(f"./data/sarcoma/binary/T1FsGd/*{decomposition}.png"))
            files = sorted([f for f in files if not "label" in f])
            labels = [int(os.path.basename(f).split("_")[2]) for f in files]
            labels = [0 if l == 1 else 1 for l in labels]
        case "sarcoma_binary_t2":
            files = sorted(glob(f"./data/sarcoma/binary/T2Fs/*{decomposition}.png"))
            files = sorted([f for f in files if not "label" in f])
            labels = [int(os.path.basename(f).split("_")[2]) for f in files]
            labels = [0 if l == 1 else 1 for l in labels]
        case "glioma_binary_t1c":
            files = sorted(glob(f"./data/glioma/binary/T1c/*image*{decomposition}.png"))
            labels = [int(os.path.basename(f).split("_")[-5]) for f in files]
            labels = [0 if l < 4 else 1 for l in labels]
        case "glioma_binary_flair":
            files = sorted(glob(f"./data/glioma/binary/FLAIR/*image*{decomposition}.png"))
            labels = [int(os.path.basename(f).split("_")[-5]) for f in files]
            labels = [0 if l < 4 else 1 for l in labels]
        case _:
            raise NotImplementedError(f"Task {task} not implemented.")

    model = VisionLanguageModel(model_name=model_name,
                                shots=shots,
                                sampling=sampling,
                                task=task,
                                decomposition=decomposition)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if dataset_fraction < 1.0:
        # files, _, labels, _ = train_test_split(files, labels, train_size=dataset_fraction, stratify=labels, random_state=42)

        files = np.array(files)
        labels = np.array(labels)
        
        # Indices per class
        idx_0 = np.where(labels == 0)[0]  # majority
        idx_1 = np.where(labels == 1)[0]  # minority

        # Keep all minority samples
        selected_idx = idx_1.copy()

        # How many total samples do we want?
        n_total = int(0.75 * len(labels))

        # How many additional majority samples do we need?
        remaining = n_total - len(selected_idx)
        remaining = max(0, min(remaining, len(idx_0)))  # clamp in case of edge cases

        # Randomly select that many majority samples
        sampled_idx_0 = np.random.choice(idx_0, remaining, replace=False)

        # Combine and shuffle
        selected_idx = np.concatenate([selected_idx, sampled_idx_0])
        np.random.shuffle(selected_idx)

        # Final lists
        files_reduced = files[selected_idx].tolist()
        labels_reduced = labels[selected_idx].tolist()

        files = files_reduced
        labels = labels_reduced

    bacc_list = []
    mcc_list = []
    f1_list = []
    # auroc_list = []
    specificity_list = []
    sensitivity_list = []
    precision_list = []
    recall_list = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(files, labels)):        

        train_files = [files[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_files = [files[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        true_labels = []
        pred_labels = []
        # pred_scores = []
        for test_file, test_label in tqdm(zip(test_files, test_labels), total=len(test_files), desc=f"Fold {fold}"):        
            output = model(test_file, train_files, train_labels)

            pred_label = output["answer"]
            pred_label = 0 if pred_label == "low-grade" else 1
            # pred_score = float(output["score"])
            # pred_score = pred_score if pred_label == 1 else 1 - pred_score

            true_labels.append(test_label)
            pred_labels.append(pred_label)
            # pred_scores.append(pred_score)

        # Compute metrics
        bal_acc = balanced_accuracy_score(test_labels, pred_labels)
        mcc = matthews_corrcoef(test_labels, pred_labels)
        f1 = f1_score(test_labels, pred_labels, average="weighted")
        # roc_auc = roc_auc_score(test_labels, pred_scores)
        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        # Log metrics
        mlflow.log_metric(f"fold_{fold}_bcc", bal_acc)
        mlflow.log_metric(f"fold_{fold}_mcc", mcc)
        mlflow.log_metric(f"fold_{fold}_f1-score", f1)
        # mlflow.log_metric(f"fold_{fold}_auroc", roc_auc)
        mlflow.log_metric(f"fold_{fold}_specificity", specificity)
        mlflow.log_metric(f"fold_{fold}_sensitivity", sensitivity)
        mlflow.log_metric(f"fold_{fold}_precision", precision)
        mlflow.log_metric(f"fold_{fold}_recall", recall)
    
        bacc_list.append(bal_acc)
        mcc_list.append(mcc)
        f1_list.append(f1)
        # auroc_list.append(roc_auc)
        specificity_list.append(specificity)
        sensitivity_list.append(sensitivity)
        precision_list.append(precision)
        recall_list.append(recall)

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
    
    # auroc_mean = np.mean(auroc_list)
    # auroc_std = np.std(auroc_list)
    # mlflow.log_metric("auroc_mean", auroc_mean)
    # mlflow.log_metric("auroc_std", auroc_std)

    specificity_mean = np.mean(specificity_list)
    specificity_std = np.std(specificity_list)
    mlflow.log_metric("specificity_mean", specificity_mean)
    mlflow.log_metric("specificity_std", specificity_std)

    sensitivity_mean = np.mean(sensitivity_list)
    sensitivity_std = np.std(sensitivity_list)
    mlflow.log_metric("sensitivity_mean", sensitivity_mean)
    mlflow.log_metric("sensitivity_std", sensitivity_std)

    precision_mean = np.mean(precision_list)
    precision_std = np.std(precision_list)
    mlflow.log_metric("precision_mean", precision_mean)
    mlflow.log_metric("precision_std", precision_std)

    recall_mean = np.mean(recall_list)
    recall_std = np.std(recall_list)
    mlflow.log_metric("recall_mean", recall_mean)
    mlflow.log_metric("recall_std", recall_std)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_id", type=int, default=0)
    args = argparser.parse_args()

    # models = ["google/medgemma-4b-it", "google/medgemma-27b-it", "google/gemma-3-4b-it", "google/gemma-3-27b-it"]
    # model_name = models[args.model_id]
    model_name = "google/medgemma-4b-it"

    dataset_fractions = [0.10, 0.25, 0.50, 0.75]
    dataset_fraction = dataset_fractions[args.model_id]

    for task in ["sarcoma_binary_t1", "sarcoma_binary_t2", "glioma_binary_t1c", "glioma_binary_flair"]:
        for shots in [0, 3, 5, 10]:
            # for sampling in ["random", "radiomics_2D", "radiomics_3D", "worst-case_2D", "worst-case_3D", "dinov3"]:
            for sampling in ["dinov3", "radiomics_2D", "random"]:
                # for decomposition in ["mip", "axial", "axial+"]:
                for decomposition in ["axial"]:

                    print(f"Starting experiment: {model_name}, {task}, {shots}, {sampling}, {decomposition}")

                    mlflow.set_experiment("fractional_data_experiments")
                    mlflow.start_run()
                    main(model_name, task, shots, sampling, decomposition, dataset_fraction)
                    mlflow.end_run()
