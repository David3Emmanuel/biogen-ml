from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO


class CancerImageWrapper(nn.Module):
    def __init__(self, model_path, cancer_type):
        super().__init__()
        # Task must be 'classify' for this model structure
        self.backbone = YOLO(model_path, task='classify')
        self.cancer_type = cancer_type
        
        # Shape: (Num_Classes, 2) -> [1yr_Risk, 3yr_Risk]
        self.register_buffer('risk_matrix', None)

    def _build_risk_matrix(self, risk_lookup, time_decay_factors):
        """
        Dynamically builds the (N, 2) matrix aligned with YOLO's specific class ID order.
        """
        model_names = self.backbone.names 
        num_classes = len(model_names)
        
        # Initialize Matrix (Rows=Classes, Cols=TimeHorizons)
        matrix = torch.zeros((num_classes, 2))
        
        print(f"--- Building Risk Matrix for {self.cancer_type} ---")
        
        for class_id, class_name in model_names.items():
            # Clean name matching
            clean_name = next((k for k in risk_lookup if k in class_name.lower()), None)
            
            if clean_name:
                base_risk = risk_lookup[clean_name]
                r1 = base_risk * time_decay_factors['1yr']
                r3 = base_risk * time_decay_factors['3yr']
                matrix[class_id, 0] = r1
                matrix[class_id, 1] = r3
            else:
                print(f"⚠️ Warning: Class '{class_name}' not found. Defaulting to 0.0")

        self.risk_matrix = matrix
        
    def train(self, mode=True):
        """
        Standard PyTorch train override.
        Crucial: We must set the *inner* model to train/eval mode.
        """
        self.training = mode
        if mode:
            self.backbone.model.train() 
        else:
            self.backbone.model.eval()
        return self

    def forward(self, x):
        """
        Forward pass that preserves gradients for GradCAM.
        """
        # 1. DIRECT CALL to underlying PyTorch model.
        # self.backbone(x) would detach gradients!
        outputs = self.backbone.model(x)
        
        # Ultralytics models often return a tuple (predictions, features) or just predictions
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
            
        # 2. Handle Logits vs Probabilities
        # The underlying model returns raw logits.
        # We must manually apply softmax because we bypassed the inference wrapper.
        probs = torch.softmax(outputs, dim=1)
            
        # 3. Apply Risk Matrix
        risk_logits = torch.matmul(probs, self.risk_matrix)
        
        return risk_logits

    def predict_clinical(self, x):
        """Helper for demo/inference (no gradients needed here)"""
        with torch.no_grad():
            logits = self.forward(x)
            
            return [{
                'type': self.cancer_type,
                '1yr': logits[i, 0].item(),
                '3yr': logits[i, 1].item()
            } for i in range(logits.shape[0])]


class HerlevWrapper(CancerImageWrapper):
    def __init__(self, model_path):
        super().__init__(model_path, cancer_type='cervical')
        
        lookup = {
            'carcinoma_in_situ': 1.00, 'severe_dysplastic': 0.40,
            'moderate_dysplastic': 0.15, 'light_dysplastic': 0.01,
            'normal_columnar': 0.00, 'normal_intermediate': 0.00, 'normal_superficiel': 0.00
        }
        decay = {'1yr': 0.30, '3yr': 1.00}
        self._build_risk_matrix(lookup, decay)


class INbreastWrapper(CancerImageWrapper):
    def __init__(self, model_path):
        super().__init__(model_path, cancer_type='breast')
        
        lookup = {
            'bi-rads_6': 1.00, 'bi-rads_5': 0.95, 'bi-rads_4': 0.40,
            'bi-rads_3': 0.02, 'bi-rads_2': 0.00, 'bi-rads_1': 0.00
        }
        decay = {'1yr': 0.95, '3yr': 1.00}
        self._build_risk_matrix(lookup, decay)


class SpecializedImageWrapper(nn.Module):
    def __init__(self, cancer_type):
        super().__init__()
        if cancer_type == 'cervical':
            self.wrapper = HerlevWrapper(Path(__file__).parent / 'trained' / 'herlev-best.pt')
        elif cancer_type == 'breast':
            self.wrapper = INbreastWrapper(Path(__file__).parent / 'trained' / 'inbreast-best.pt')
        else:
            raise ValueError(f"Unsupported cancer type: {cancer_type}")

    def forward(self, x):
        return self.wrapper.forward(x)
    
    def predict_clinical(self, x):
        return self.wrapper.predict_clinical(x)