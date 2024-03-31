import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from glob import glob


def load_metrics(path):
    '''
        path is the path to the metrics.npz file
        metrics keys are the metric name,
        values are the metric values
    '''
    metrics = np.load(path)
    
    return pd.DataFrame({k: metrics[k] for k in metrics.keys()}, index=[0])

def summarize_metrics(prob_methods, loss_types, alphas, knns, data_names):
    result = []
    # each row is a different experiment
    # record the metrics and the parameters for each row
    for prob_method in prob_methods:
        for loss_type in loss_types:
            for alpha in alphas:
                for knn in knns:
                    for data_name in data_names:
                        # eg. noisy_4_paths_17580_3000_1_0.5_0.5_all.npz, seed = 4
                        if loss_type == 'kl':
                            path = f'../results/sepa_{prob_method}_{loss_type}_a{alpha}_knn{knn}_{data_name}'
                        else:
                            path = f'../results/sepa_{prob_method}_{loss_type}_a{alpha}_knn{knn}_{data_name}'
                        path = os.path.join(path, 'metrics.npz')
                        if not os.path.exists(path):
                            #print('skipping ', path, ' ...')
                            continue
                        
                        print('computing ', path, ' ...')
                        seed = data_name.split('_')[1]
                        data_method = data_name.split('_')[2]
                        bvc = data_name.split('_')[6]
                        dropout = data_name.split('_')[7]

                        metrics = load_metrics(path)

                        metrics['prob_method'] = prob_method
                        metrics['loss_type'] = loss_type
                        metrics['alpha'] = alpha
                        metrics['knn'] = knn
                        metrics['data_name'] = data_name
                        metrics['data_method'] = data_method
                        metrics['bvc'] = bvc
                        metrics['dropout'] = dropout
                        metrics['seed'] = seed

                        result.append(metrics)

    return pd.concat(result, ignore_index=True)

def main():
    prob_methods = ['gaussian', 'sym_gaussian', 'tstudent', 'adjusted_gaussian', 'heat_kernel', 'distance']
    loss_types = ['kl', 'jsd', 'mdiv']
    alphas = [1.0, 10]
    knns = [5]

    root = '/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/synthetic_data3/'
    data_paths = sorted(glob(root+'noisy_*_all.npz'))
    data_names = [os.path.basename(path) for path in data_paths]

    metrics = summarize_metrics(prob_methods, loss_types, alphas, knns, data_names)
    
    metrics.to_csv('../results/splatter_metrics.csv')

    # group by prob_method, alpha, knn, data_name
    metrics1 = metrics.groupby(['prob_method', 'alpha', 'knn', 'data_name']).agg(['mean', 'std'])
    metrics1 = metrics1.applymap(lambda x: f'{x:.3f}')

    metrics1.to_csv('../results/splatter_metrics_groupedbyname.csv')
    # check 'Test', 'PHATE', 'TSNE' columns 
    print('test')
    print(metrics1[['Test', 'PHATE', 'TSNE', 'UMAP', 'DiffMap']])

    # group by prob_method, alpha, knn, data_method, bvc, dropout
    metrics2 = metrics.groupby(['prob_method', 'loss_type', 'alpha', 'knn', 'data_method', 'bvc', 'dropout']).agg(['mean', 'std'])
    metrics2 = metrics2.applymap(lambda x: f'{x:.3f}')

    metrics2.to_csv('../results/splatter_metrics_groupedbyparam.csv')
    # check 'Test', 'PHATE', 'TSNE' columns
    print(metrics2[['Test', 'PHATE', 'TSNE', 'UMAP', 'DiffMap']])

    # visualize(metrics2)

    return



def visualize(grouped_metrics):
    print(grouped_metrics.columns)
    print(grouped_metrics.index)
    print(grouped_metrics.index.get_level_values('prob_method').unique())
    prob_method = grouped_metrics.index.get_level_values('prob_method').unique().tolist()
    alpha = grouped_metrics.index.get_level_values('alpha').unique().tolist()
    knn = grouped_metrics.index.get_level_values('knn').unique().tolist()
    data_method = grouped_metrics.index.get_level_values('data_method').unique().tolist()

    line_names = ['Test', 'PHATE', 'TSNE', 'UMAP', 'DiffMap']

    # n_rows = len(prob_method) * len(alpha) * len(knn)
    # n_cols = len(data_method)
    
    # fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
    # for i, (prob, a, k, data) in enumerate(grouped_metrics.index):
    #     for j, line in enumerate(line_names):
    #         ax = axs[i, j]
    #         ax.set_title(f'{prob}, {a}, {k}, {data}, {line}')
    #         ax.plot(grouped_metrics.loc[(prob, a, k, data), line])
    
    # plt.tight_layout()
    # save_path = '../results/splatter_metrics_groupedbyparam.png'
    # plt.savefig(save_path)



if __name__ == "__main__":
    main()
