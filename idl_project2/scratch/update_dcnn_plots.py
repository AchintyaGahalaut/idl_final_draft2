import json
import os

def update_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    found = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'def plot_sorted_predictions' in ''.join(cell['source']):
            new_source = []
            for line in cell['source']:
                if 'def plot_sorted_predictions(model, X_test, y_test, device=\'cpu\'):' in line:
                    new_source.append("def plot_sorted_predictions(model, X_test, y_test, dataset_id, device='cpu'):\n")
                elif 'plt.title(\'Sorted Prediction for Testing Engine Units\')' in line:
                    new_source.append("    from sklearn.metrics import mean_squared_error\n")
                    new_source.append("    rmse = np.sqrt(mean_squared_error(y_test, predictions))\n")
                    new_source.append("    plt.title(f'Sorted Prediction for Testing Engine Units ({dataset_id} - RMSE: {rmse:.2f})')\n")
                elif 'plot_sorted_predictions(model, data[\'X_test\'], data[\'y_test\'], device=device)' in line:
                    new_source.append("    plot_sorted_predictions(model, data['X_test'], data['y_test'], dataset_id, device=device)\n")
                else:
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
    update_notebook('g:/My Drive/cmu/idl/final_project/idl_project2/dcnn_variant/final/final_dcnn_variant.ipynb')
