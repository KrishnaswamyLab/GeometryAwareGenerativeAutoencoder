import os
import numpy as np
import pandas as pd


def load_metrics(path):
    '''
        path is the path to the metrics.npz file
        metrics keys are the metric name,
        values are the metric values
    '''
    metrics = np.load(path)
    
    return pd.DataFrame({k: metrics[k] for k in metrics.keys()}, index=[0])

def summarize_metrics(prob_methods, alphas, knns, data_names, data_noises, seeds):
    result = []
    # each row is a different experiment
    # record the metrics and the parameters for each row
    for prob_method in prob_methods:
        for alpha in alphas:
            for knn in knns:
                for data_name in data_names:
                    for data_noise in data_noises:
                        for seed in seeds:
                            path = f'../results/sepa_{prob_method}_a{alpha}_knn{knn}_{data_name}_noise{data_noise:.1f}_seed{seed}'
                            path = os.path.join(path, 'metrics.npz')
                            if not os.path.exists(path):
                                print('skipping ', path, ' ...')
                                continue
                            
                            print('computing ', path, ' ...')
                            metrics = load_metrics(path)
                            metrics['prob_method'] = prob_method
                            metrics['alpha'] = alpha
                            metrics['knn'] = knn
                            metrics['data_name'] = data_name
                            metrics['data_noise'] = data_noise
                            metrics['seed'] = seed
                            result.append(metrics)

    return pd.concat(result, ignore_index=True)

def main():
    prob_methods = ['gaussian', 'tstudent',]
    alphas = [1, 10]
    knns = [5]
    data_names = ['swiss_roll', 's_curve', 'tree']
    data_noises = [0.1, 0.5, 1.0, 2.0]
    seeds = [1, 2, 3, 4, 5]

    metrics = summarize_metrics(prob_methods, alphas, knns, data_names, data_noises, seeds)
    
    metrics.to_csv('../results/toy_metrics.csv')

    # group by prob_method, alpha, knn, data_name, data_noise
    # and compute the mean and std for each metric, format mean+-std
    # only show .3f, format mean(+-std)
    metrics = metrics.groupby(['prob_method', 'alpha', 'knn', 'data_name', 'data_noise']).agg(['mean', 'std'])
    metrics = metrics.applymap(lambda x: f'{x:.3f}')

    metrics.to_csv('../results/toy_metrics_summary.csv')
    # check 'Test', 'PHATE', 'TSNE' columns 
    print(metrics[['Test', 'PHATE', 'TSNE', 'UMAP', 'DiffMap']])

if __name__ == "__main__":
    main()
