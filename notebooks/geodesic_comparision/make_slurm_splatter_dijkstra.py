prepend = 'cd /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/notebooks/geodesic_comparision/; /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/.conda_envs/geosink/bin/python dijkstra.py'


with open('job.txt', 'w') as file:
    data_names = ["noisy_1_groups_17580_3000_1_0.18_0.5_all","noisy_1_groups_17580_3000_1_0.25_0.5_all","noisy_1_groups_17580_3000_1_0.5_0.5_all","noisy_1_groups_17580_3000_1_0_0.5_all","noisy_1_paths_17580_3000_1_0.18_0.5_all","noisy_1_paths_17580_3000_1_0.25_0.5_all","noisy_1_paths_17580_3000_1_0.5_0.5_all","noisy_1_paths_17580_3000_1_0_0.5_all","noisy_2_groups_17580_3000_1_0.18_0.5_all","noisy_2_groups_17580_3000_1_0.25_0.5_all","noisy_2_groups_17580_3000_1_0.5_0.5_all","noisy_2_groups_17580_3000_1_0_0.5_all","noisy_2_paths_17580_3000_1_0.18_0.5_all","noisy_2_paths_17580_3000_1_0.25_0.5_all","noisy_2_paths_17580_3000_1_0.5_0.5_all","noisy_2_paths_17580_3000_1_0_0.5_all","noisy_3_groups_17580_3000_1_0.18_0.5_all","noisy_3_groups_17580_3000_1_0.25_0.5_all","noisy_3_groups_17580_3000_1_0.5_0.5_all","noisy_3_groups_17580_3000_1_0_0.5_all","noisy_3_paths_17580_3000_1_0.18_0.5_all","noisy_3_paths_17580_3000_1_0.25_0.5_all","noisy_3_paths_17580_3000_1_0.5_0.5_all","noisy_3_paths_17580_3000_1_0_0.5_all","noisy_4_groups_17580_3000_1_0.18_0.5_all","noisy_4_groups_17580_3000_1_0.25_0.5_all","noisy_4_groups_17580_3000_1_0.5_0.5_all","noisy_4_groups_17580_3000_1_0_0.5_all","noisy_4_paths_17580_3000_1_0.18_0.5_all","noisy_4_paths_17580_3000_1_0.25_0.5_all","noisy_4_paths_17580_3000_1_0.5_0.5_all","noisy_4_paths_17580_3000_1_0_0.5_all","noisy_5_groups_17580_3000_1_0.18_0.5_all","noisy_5_groups_17580_3000_1_0.25_0.5_all","noisy_5_groups_17580_3000_1_0.5_0.5_all","noisy_5_groups_17580_3000_1_0_0.5_all","noisy_5_paths_17580_3000_1_0.18_0.5_all","noisy_5_paths_17580_3000_1_0.25_0.5_all","noisy_5_paths_17580_3000_1_0.5_0.5_all","noisy_5_paths_17580_3000_1_0_0.5_all"]

    for data_name in data_names:
        command = f"{prepend} --cfg_main_data_name \"{data_name}\" --n_steps 20"
        file.write(command + '\n')

    data_names = ["true_1_groups_17580_3000_1_all","true_1_paths_17580_3000_1_all","true_2_groups_17580_3000_1_all","true_2_paths_17580_3000_1_all","true_3_groups_17580_3000_1_all","true_3_paths_17580_3000_1_all","true_4_groups_17580_3000_1_all","true_4_paths_17580_3000_1_all","true_5_groups_17580_3000_1_all","true_5_paths_17580_3000_1_all"]

    for data_name in data_names:
        command = f"{prepend} --cfg_main_data_name \"{data_name}\" --n_steps 200"
        file.write(command + '\n')

