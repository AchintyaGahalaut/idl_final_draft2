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
                    new_source.append("    # Sort results by true RUL for a better visualization (matches paper format)\n")
                    new_source.append("    y_test = np.array(y_test)\n")
                    new_source.append("    sort_idx = np.argsort(y_test)\n")
                    new_source.append("    sorted_y_test = y_test[sort_idx]\n")
                    new_source.append("    sorted_preds = preds[sort_idx]\n")
                    new_source.append("\n")
                    new_source.append("    plt.figure(figsize=(10, 6))\n")
                    new_source.append("    plt.plot(sorted_y_test, color='blue', label='True RUL', marker='o', markersize=3, linestyle='None')\n")
                    new_source.append("    plt.plot(sorted_preds, color='red', label='Predicted RUL', marker='x', markersize=3, linestyle='None')\n")
                    new_source.append("    plt.axhline(y=125, color='gray', linestyle='--', alpha=0.5, label='Max RUL Cap')\n")
                    new_source.append("    plt.title(f'Sorted RUL Predictions - {dataset_id} (RMSE: {rmse:.2f})')\n")
                    new_source.append("    plt.xlabel('Test Units (Sorted by True RUL)')\n")
                    new_source.append("    plt.ylabel('Remaining Useful Life (Cycles)')\n")
                    new_source.append("    plt.legend()\n")
                    new_source.append("    plt.grid(True, alpha=0.3)\n")
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
