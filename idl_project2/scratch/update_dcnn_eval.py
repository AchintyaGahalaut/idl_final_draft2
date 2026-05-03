import json
import os

def update_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    found = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'for dataset_id, model in trained_models.items():' in ''.join(cell['source']):
            new_source = []
            for line in cell['source']:
                if 'for dataset_id, model in trained_models.items():' in line:
                    # Replace the loop to use 'datasets' and load the .pth files
                    new_source.append("datasets = ['FD001', 'FD002', 'FD003', 'FD004']\n")
                    new_source.append("seq_lengths = {'FD001': 30, 'FD002': 20, 'FD003': 30, 'FD004': 15}\n")
                    new_source.append("selected_sensors = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]\n")
                    new_source.append("\n")
                    new_source.append("for dataset_id in datasets:\n")
                    new_source.append("    # Instantiate model to load weights\n")
                    new_source.append("    model = PaperDeepCNN(sequence_length=seq_lengths[dataset_id], num_features=len(selected_sensors)).to(device)\n")
                    new_source.append("    model.load_state_dict(torch.load(f'dcnn_model_{dataset_id}.pth', map_location=device))\n")
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
