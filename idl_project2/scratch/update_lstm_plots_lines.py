import json
import os

def update_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    found = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'def evaluate_model' in ''.join(cell['source']):
            new_source = []
            skip = False
            for line in cell['source']:
                if 'plt.figure' in line:
                    skip = True
                    # Insert the new plotting logic
                    new_source.append("    plt.figure(figsize=(10, 6))\n")
                    new_source.append("    plt.plot(sorted_y_test, color='blue', label='Actual RUL', marker='.', markersize=5, linewidth=1)\n")
                    new_source.append("    plt.plot(sorted_preds, color='red', label='Predicted RUL', marker='*', markersize=5, linewidth=1)\n")
                    new_source.append("    plt.axhline(y=125, color='black', linestyle='--', label='Rectified (125)')\n")
                    new_source.append("    plt.title(f'Sorted Prediction for Testing Engine Units ({dataset_id} - RMSE: {rmse:.2f})')\n")
                    new_source.append("    plt.xlabel('Test unit with increasing RUL')\n")
                    new_source.append("    plt.ylabel('Remaining useful life')\n")
                    new_source.append("    plt.legend()\n")
                    new_source.append("    plt.grid(True, linestyle='--', alpha=0.7)\n")
                    new_source.append("    plt.show()\n")
                    continue
                
                if skip:
                    if 'plt.' in line and 'plt.figure' not in line:
                        continue
                    else:
                        skip = False
                
                if not skip:
                    new_source.append(line)
            
            cell['source'] = new_source
            found = True
            break
            
    if found:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully updated {file_path}")
    else:
        print(f"Could not find target cell in {file_path}")

if __name__ == "__main__":
    update_notebook('g:/My Drive/cmu/idl/final_project/idl_project2/baseline_lstm/lstm_multi_v2.ipynb')
