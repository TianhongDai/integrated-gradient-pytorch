import torch
import torchmetrics
from torchmetrics import MetricCollection,MeanAbsoluteError , MeanSquaredError,SpearmanCorrcoef,PearsonCorrcoef,Accuracy

def get_metrics_collections_base(prefix,is_regressor:bool=True
                            # device="cuda" if torch.cuda.is_available() else "cpu",
                            
                            ):
    if is_regressor:
        metrics = MetricCollection(
                {
                    "MeanAbsoluteError":MeanAbsoluteError(),
                    "MeanSquaredError":MeanSquaredError(),
                    "SpearmanCorrcoef":SpearmanCorrcoef(),
                    "PearsonCorrcoef":PearsonCorrcoef()
                    
                },
                prefix=prefix
                )
    else:
         metrics = MetricCollection(
            {
                "Accuracy":Accuracy(),
                "Top_3":Accuracy(top_k=3),
                # "Top_5" :Accuracy(top_k=5),
                # "Precision_micro":Precision(num_classes=NUM_CLASS,average="micro"),
                # "Precision_macro":Precision(num_classes=NUM_CLASS,average="macro"),
                # "Recall_micro":Recall(num_classes=NUM_CLASS,average="micro"),
                # "Recall_macro":Recall(num_classes=NUM_CLASS,average="macro"),
                # "F1_micro":torchmetrics.F1(NUM_CLASS,average="micro"),
                # "F1_macro":torchmetrics.F1(NUM_CLASS,average="micro"),

            },
            prefix=prefix
            )
    return metrics