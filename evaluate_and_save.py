import torch
import json
import csv
import os
from datetime import datetime
from main_finetune import get_args_parser, main
import torch.nn as nn

if __name__ == "__main__":
    # Create results directory
    os.makedirs("./evaluation_results", exist_ok=True)
    
    # Run evaluation
    import sys
    sys.argv = [
        "main_finetune.py",
        "--data_path", "./oimhs_data",
        "--nb_classes", "2",
        "--batch_size", "8",
        "--model", "RETFound_mae",
        "--model_arch", "retfound_mae",
        "--input_size", "224",
        "--eval",
        "--resume", "./output_dir/OIMHS_finetune/checkpoint-best.pth"
    ]
    
    args = get_args_parser().parse_args()
    
    # Redirect output to capture metrics
    from io import StringIO
    import sys as sys_io
    old_stdout = sys_io.stdout
    sys_io.stdout = StringIO()
    
    criterion = nn.CrossEntropyLoss()
    main(args, criterion)
    
    output = sys_io.stdout.getvalue()
    sys_io.stdout = old_stdout
    
    # Parse metrics from output
    import re
    accuracy_match = re.search(r"Accuracy:\s+([0-9.]+)", output)
    f1_match = re.search(r"F1 Score:\s+([0-9.]+)", output)
    auc_match = re.search(r"ROC AUC:\s+([0-9.]+)", output)
    precision_match = re.search(r"Precision:\s+([0-9.]+)", output)
    recall_match = re.search(r"Recall:\s+([0-9.]+)", output)
    
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": args.resume,
        "accuracy": float(accuracy_match.group(1)) if accuracy_match else None,
        "f1_score": float(f1_match.group(1)) if f1_match else None,
        "roc_auc": float(auc_match.group(1)) if auc_match else None,
        "precision": float(precision_match.group(1)) if precision_match else None,
        "recall": float(recall_match.group(1)) if recall_match else None,
        "num_parameters": 303303682,
        "dataset": "OIMHS",
        "num_classes": 2,
        "input_size": 224,
        "model": "RETFound_mae"
    }
    
    # Save as JSON
    json_path = "./evaluation_results/final_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {json_path}")
    
    # Save as CSV
    csv_path = "./evaluation_results/final_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    print(f"Results saved to {csv_path}")
    
    # Print results
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    for key, value in results.items():
        print(f"{key}: {value}")
    print("="*50)