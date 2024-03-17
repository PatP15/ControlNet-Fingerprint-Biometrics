import os
import numpy as np

AVG = 'avg'
STD = 'std'
N = 'n'

def read_metric(filepath, metric):
    assert os.path.exists(filepath)
    with open(filepath, 'r') as fin:
        for curr_line in fin.readlines():
            if metric not in curr_line:
                continue
            value = float(curr_line.strip().split()[-1])
            assert value >= 0 and value <= 1
            return value
    raise ValueError(f'could not find metric {metric} in {filepath}')
    

def get_metric(home_dir, model_name='finetune_embeddings_diffusion2', metric='auc'):
    the_metric_values = list()
    for filename in os.listdir(home_dir):
        if model_name in filename:
            filepath = os.path.join(home_dir, filename)
            the_metric_values.append(read_metric(filepath, metric))
    avg_value = np.mean(the_metric_values)
    std_value = np.std(the_metric_values, ddof=1)
    num_samples = len(the_metric_values)

    return {AVG: avg_value, STD: std_value, N: num_samples}

# Thanks ChatGPT!
def dicts_to_latex_table(entries, dataset_name, metric_name):
    """
    Convert a list of (name, dict) tuples into a LaTeX table.
    
    Parameters:
    entries (list of tuples): Each tuple contains a name (str) and a dictionary with 'avg', 'std', and 'n' keys.
    
    Returns:
    str: LaTeX code for the table.
    """
    header = ["Pre-Train", f"$\mu_{{{metric_name}}}$", f"$\sigma_{{{metric_name}}}$", "n"]
    latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lccc}\n\\hline\n"
    latex_table += " & ".join(header) + " \\\\\n\\hline\n"
    
    for name, data in entries:
        row = [name, f"{data['avg']:.4f}", f"{data['std']:.4f}", str(data['n'])]
        latex_table += " & ".join(row) + " \\\\\n"
    
    latex_table += f"\\hline\n\\end{{tabular}}\n\\caption{{{metric_name} for {dataset_name}}}\n\\label{{tab:{metric_name}{dataset_name}}}\n\\end{{table}}"
    return latex_table

if __name__ == "__main__":
    all_model_names = {'finetune_embeddings_diffusion2':'OpenDiffFinger',
                       'finetune_embeddings_printsgan2':'PrintsGAN',
                       'finetune_embeddings_imagenet':'ImageNet', 
                       'model_embeddings_baseline':'Random Init'}
    all_metrics = {'auc':'AUC', 'best accuracy':'Best Accuracy'}
    all_home_dirs = {'/home/gabeguo/pat_fingerprint_results/results':'SD301', 
                     '/home/gabeguo/pat_fingerprint_results/results2':'SD302',
                     '/home/gabeguo/pat_fingerprint_results/results3':'SD300'}

    for curr_metric in all_metrics:
        for curr_home_dir in all_home_dirs:
            print(f'Metric: {curr_metric}; Home Dir: {curr_home_dir}')
            curr_experiment_entries = list()
            for curr_model_name in all_model_names:
                # print(f'Metric: {curr_metric}; Home Dir: {curr_home_dir}; Model Name: {curr_model_name}')
                # print(get_metric(home_dir=curr_home_dir, model_name=curr_model_name, metric=curr_metric))
                curr_experiment_entries.append(
                    (all_model_names[curr_model_name], 
                     get_metric(home_dir=curr_home_dir, model_name=curr_model_name, metric=curr_metric)))
            print(dicts_to_latex_table(curr_experiment_entries, 
                                       dataset_name=all_home_dirs[curr_home_dir],
                                       metric_name=all_metrics[curr_metric]))
            print('\n\n')
