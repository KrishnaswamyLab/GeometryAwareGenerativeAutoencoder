prepend = 'cd /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/notebooks/geodesic_comparision/; /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/.conda_envs/geosink/bin/python run_splatter.py'


with open('job.txt', 'w') as file:
    ae_sweep_id = 'ulxztdsf'
    # disc_sweep_id = 'pu8f08s6'
    disc_sweep_id = '72v3ghe5'
    data_names = ["noisy_1_groups_17580_3000_1_0.18_0.5_all","noisy_1_groups_17580_3000_1_0.25_0.5_all","noisy_1_groups_17580_3000_1_0.5_0.5_all","noisy_1_groups_17580_3000_1_0_0.5_all","noisy_1_paths_17580_3000_1_0.18_0.5_all","noisy_1_paths_17580_3000_1_0.25_0.5_all","noisy_1_paths_17580_3000_1_0.5_0.5_all","noisy_1_paths_17580_3000_1_0_0.5_all","noisy_2_groups_17580_3000_1_0.18_0.5_all","noisy_2_groups_17580_3000_1_0.25_0.5_all","noisy_2_groups_17580_3000_1_0.5_0.5_all","noisy_2_groups_17580_3000_1_0_0.5_all","noisy_2_paths_17580_3000_1_0.18_0.5_all","noisy_2_paths_17580_3000_1_0.25_0.5_all","noisy_2_paths_17580_3000_1_0.5_0.5_all","noisy_2_paths_17580_3000_1_0_0.5_all","noisy_3_groups_17580_3000_1_0.18_0.5_all","noisy_3_groups_17580_3000_1_0.25_0.5_all","noisy_3_groups_17580_3000_1_0.5_0.5_all","noisy_3_groups_17580_3000_1_0_0.5_all","noisy_3_paths_17580_3000_1_0.18_0.5_all","noisy_3_paths_17580_3000_1_0.25_0.5_all","noisy_3_paths_17580_3000_1_0.5_0.5_all","noisy_3_paths_17580_3000_1_0_0.5_all","noisy_4_groups_17580_3000_1_0.18_0.5_all","noisy_4_groups_17580_3000_1_0.25_0.5_all","noisy_4_groups_17580_3000_1_0.5_0.5_all","noisy_4_groups_17580_3000_1_0_0.5_all","noisy_4_paths_17580_3000_1_0.18_0.5_all","noisy_4_paths_17580_3000_1_0.25_0.5_all","noisy_4_paths_17580_3000_1_0.5_0.5_all","noisy_4_paths_17580_3000_1_0_0.5_all","noisy_5_groups_17580_3000_1_0.18_0.5_all","noisy_5_groups_17580_3000_1_0.25_0.5_all","noisy_5_groups_17580_3000_1_0.5_0.5_all","noisy_5_groups_17580_3000_1_0_0.5_all","noisy_5_paths_17580_3000_1_0.18_0.5_all","noisy_5_paths_17580_3000_1_0.25_0.5_all","noisy_5_paths_17580_3000_1_0.5_0.5_all","noisy_5_paths_17580_3000_1_0_0.5_all"]

    for data_name in data_names:
        command = f"{prepend} --cfg_main_data_name \"{data_name}\" --cfg_main_ae_sweep_id {ae_sweep_id} --cfg_main_disc_sweep_id {disc_sweep_id}"
        file.write(command + '\n')
