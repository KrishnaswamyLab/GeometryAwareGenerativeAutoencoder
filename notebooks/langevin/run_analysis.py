from combined_analysis import main
import traceback
from tqdm import tqdm

total_iterations = len([2, 3]) * len(['saddle', 'hsphere', 'paraboloid']) * len(['uniform', 'gaussian']) * len(['heatgeo'])

with tqdm(total=total_iterations, desc="Processing") as pbar:
    for dim in [2, 3]:
        for dataset in ['saddle', 'hemisphere', 'paraboloid']:
            for data_type in ['uniform', 'gaussian']:
                for method in ['heatgeo']:
                    try:
                        main(dim=dim, dataset=dataset, data_type=data_type, method=method, save_path='./results')
                    except Exception as e:
                        print(f"Error occurred with parameters: dim={dim}, dataset={dataset}, data_type={data_type}, method={method}")
                        print(f"Error message: {str(e)}")
                        print("Traceback:")
                        traceback.print_exc()
                    finally:
                        pbar.update(1)
