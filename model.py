from transformers import pipeline
import torch
from huggingface_hub import login
from PIL import Image
import re
import json
import ast
import numpy as np
import os
import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler

class VisionLanguageModel:
    def __init__(self, 
                 model_name="google/medgemma-4b-it",
                 shots: int = 10,
                 sampling: str = "radiomics", 
                 task: str = "sarcoma_binary",
                 decomposition: str = "axial"
                 ):

        assert model_name in ["google/medgemma-4b-it", "google/gemma-3-4b-it", "google/medgemma-27b-it", "google/gemma-3-27b-it", "google/gemma-3-12b-it"]
        assert shots in [0, 1, 3, 5, 7, 10, -1]
        assert sampling in ["radiomics_2D", "radiomics_3D", "random"]
        assert task in ["sarcoma_binary", "sarcoma_multiclass"]
        assert decomposition in ["axial", "axial+", "mip"]

        self.modle_name = model_name
        self.shots = shots
        self.sampling = sampling
        self.task = task
        self.decomposition = decomposition
        
        login(token=torch.load("hf_token.pt"))

        self.pipe = pipeline("image-text-to-text",
                             model=model_name,
                             dtype=torch.bfloat16,
                             device="cuda")    
    
    def cosine_similarity_matrix(self, patient_ids, vectors):
        """
        Computes the NxN cosine similarity matrix for patient feature vectors.
        
        Args:
            patient_ids (list): List of patient identifiers (length N).
            vectors (list or np.ndarray): List/array of feature vectors (shape NxD).
        
        Returns:
            pd.DataFrame: NxN cosine similarity matrix with patient_ids as row/column labels.
        """
        if len(patient_ids) != len(vectors):
            raise ValueError("Length of patient_ids and vectors must be the same.")
        
        X = np.array(vectors, dtype=float)

        # Scale each feature to zero mean, unit variance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[:-1])  # Exclude last vector (test sample) from scaling fit
        X_scaled = np.vstack([X_scaled, scaler.transform(X[-1].reshape(1, -1))])  # Transform last vector

        # Normalize each vector to unit length (for cosine)
        norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
        X_normed = X_scaled / norms

        # Compute cosine similarity
        sim_matrix = X_normed @ X_normed.T
        
        # Return labeled DataFrame
        return pd.DataFrame(sim_matrix, index=patient_ids, columns=patient_ids)
    
    def get_top_n_similar(self, sim_df, patient_id, n=5, include_self=False):
        """
        Returns the N most similar patients to the given patient_id based on the cosine similarity DataFrame.
        
        Args:
            sim_df (pd.DataFrame): NxN cosine similarity matrix (patients x patients).
            patient_id (str): The patient ID for which to find similar patients.
            n (int): Number of top similar patients to return.
            include_self (bool): Whether to include the patient itself in the result.
        
        Returns:
            pd.Series: Top-N most similar patients with similarity scores (sorted descending).
        """
        if patient_id not in sim_df.index:
            raise ValueError(f"Patient ID '{patient_id}' not found in the similarity matrix.")
        
        # Extract similarity scores for the patient
        similarities = sim_df.loc[patient_id]
        
        # Optionally remove self-comparison
        if not include_self:
            similarities = similarities.drop(patient_id, errors='ignore')
        
        # Sort descending and take top N
        top_n = similarities.sort_values(ascending=False).head(n)
        
        # return top_n
        return top_n.index.tolist(), top_n.values.tolist()
    
    def create_system_prompt(self) -> dict:

        match self.task:
            case "sarcoma_binary":
                file_path = "/home/johannes/Data/SSD_2.0TB/ICL-VL/johannes/prompts/sarcoma/binary/system_prompt.txt"
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            case _:
                raise NotImplementedError(f"Task {self.task} not implemented.")

        system_prompt_dict = {
            "role": "system",
            "content": [
                {
                    "type": "text", 
                    "text": content,
                },
            ],
        }

        return system_prompt_dict

    def create_user_prompt(self, test_sample, train_samples, train_labels) -> dict:

        # Zero-shot
        if self.shots == 0:
            image = Image.open(test_sample)

            match self.task:
                case "sarcoma_binary":
                    file_path = "/home/johannes/Data/SSD_2.0TB/ICL-VL/johannes/prompts/sarcoma/binary/user_prompt_zero_shot.txt"
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                case _:
                    raise NotImplementedError(f"Task {self.task} not implemented.")
            
            content = [
                {"type": "text", "text": content},
                {"type": "image", "image": image}
            ]

            user_prompt_dict = {
                "role": "user",
                "content": content
            }          

        # Few-shot
        else:        

            if self.sampling == "random":
                images_low_grade = [train_samples[i] for i in range(len(train_samples)) if train_labels[i] == 0]
                images_high_grade = [train_samples[i] for i in range(len(train_samples)) if train_labels[i] == 1]
                low_grade_indices = np.random.choice(range(len(images_low_grade)), size=self.shots, replace=False)
                high_grade_indices = np.random.choice(range(len(images_high_grade)), size=self.shots, replace=False)
                images_low_grade = np.array(images_low_grade)[low_grade_indices]
                images_high_grade = np.array(images_high_grade)[high_grade_indices]

            elif self.sampling in ["radiomics_2D", "radiomics_3D"]:

                similarity_type = "2D" if self.sampling == "radiomics_2D" else "3D"
                files = [f.replace(f"_preprocessed_{self.decomposition}.png", f"_features{similarity_type}.pt") for f in train_samples]
                files.append(test_sample.replace(f"_preprocessed_{self.decomposition}.png", f"_features{similarity_type}.pt"))                

                vectors = []
                patient_ids = []

                for f in files:                   
                 
                    data = torch.load(f)
                    vector = [data[k] for k in data.keys()]
                    vectors.append(vector)

                    patient_id = os.path.basename(f).split("_")[0]
                    patient_ids.append(patient_id)
                    

                df = self.cosine_similarity_matrix(patient_ids, vectors)                
                
                test_sample_id = os.path.basename(test_sample).split("_")[0]
                top_similar_ids, top_similar_scores = self.get_top_n_similar(df, patient_id=test_sample_id, n=df.shape[0], include_self=False)

                images_low_grade = []
                images_high_grade = []
                for pid in top_similar_ids:
                    for i in range(len(train_samples)):
                        if pid in train_samples[i]:
                            if train_labels[i] == 0 and len(images_low_grade) < self.shots:
                                images_low_grade.append(train_samples[i])
                                break
                            elif train_labels[i] == 1 and len(images_high_grade) < self.shots:
                                images_high_grade.append(train_samples[i])
                                break
                    if len(images_low_grade) >= self.shots and len(images_high_grade) >= self.shots:
                        break

                images_low_grade = np.array(images_low_grade)
                images_high_grade = np.array(images_high_grade)

            images_low_grade_pil = [Image.open(img_path) for img_path in images_low_grade]
            images_high_grade_pil = [Image.open(img_path) for img_path in images_high_grade]
            test_image = Image.open(test_sample)

            match self.task:
                case "sarcoma_binary":
                    file_path = "/home/johannes/Data/SSD_2.0TB/ICL-VL/johannes/prompts/sarcoma/binary/user_prompt_few_shot.txt"
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                case "sarcoma_multiclass":
                    file_path = "/home/johannes/Data/SSD_2.0TB/ICL-VL/johannes/prompts/sarcoma/multiclass/user_prompt_few_shot.txt"
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                case _:
                    raise NotImplementedError(f"Task {self.task} not implemented.")            

            user_query_pre = content.split("-----------")[0].strip()
            user_query_post = content.split("-----------")[1].strip()

            content = []
            content.append({"type": "text", "text": user_query_pre})

            for i in range(self.shots):
                content.append({"type": "text", "text": "The following image shows a low-grade sarcoma example:"})
                content.append({"type": "image", "image": images_low_grade_pil[i]})
                content.append({"type": "text", "text": "The following image shows a high-grade sarcoma example:"})
                content.append({"type": "image", "image": images_high_grade_pil[i]})

            content.append({"type": "text", "text": user_query_post})
            content.append({"type": "image", "image": test_image})

            user_prompt_dict = {
                "role": "user",
                "content": content
            }

        return user_prompt_dict


    def create_message(self, test_sample, train_samples, train_labels):
        system_prompt_dict = self.create_system_prompt()
        user_prompt_dict = self.create_user_prompt(test_sample, train_samples, train_labels)

        messages = [system_prompt_dict, user_prompt_dict]

        return messages    
    
    def __call__(self, test_sample, train_samples, train_labels):

        message = self.create_message(test_sample, train_samples, train_labels)
        output = self.pipe(text=message, max_new_tokens=200)

        output = output[0]["generated_text"][-1]["content"]       
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", output.strip(), flags=re.DOTALL)     
        data = json.loads(cleaned)    

        return data







