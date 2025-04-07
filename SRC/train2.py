import argparse
import logging
import matplotlib.pyplot as plt
import os
from genetic import *
from slca import *
from utils import*

data_folder = '/content/drive/MyDrive/slca-main-pavan/data'  
parameter_folder = '/content/drive/MyDrive/slca-main-pavan/config'

# args = parser.parse_args()
# data_folder = args.data
# parameter_folder = args.config

basic_loader = ParamsLoader(parameter_folder)
#parse the parameters of the genetic algorithm
gens_num, gen_size, model_type, participants, metrics, n_metrics, lca_model = GA_ParamsLoader(parameter_folder).load('ga_parameters')
    #parameters of simulation
trial_length, n_trials, desired_res = basic_loader.load('slca_parameters_sim').values()
data = DataLoader(data_folder, desired_res)
#parse the parameters range of the SLCA
params_range = SLCA_ParamsRangeLoader(parameter_folder).load('slca_parameters_range')

#parse the initial parameters of the SLCA
params_init = SLCA_ParamsInitLoader(parameter_folder).load('slca_parameters_init')

#parse fixed parameters of the SLCA
fixed_parameters = SLCA_ParamsFixedLoader(parameter_folder).load('slca_parameters_fixed')
def configure_logging():
    logging.basicConfig(filename='ga_log.txt', level=logging.INFO)



def visualize_human_gaze(image_path, saliency_map, human_gaze_coordinates, output_path='human_gaze_prediction.png'):
    # Load the image
    image = imread(image_path)

    # Plot the image
    plt.imshow(image)

    # Plot the saliency map
    plt.imshow(saliency_map, cmap='viridis', alpha=0.5, origin='lower')

    # Plot human gaze coordinates
    plt.scatter(human_gaze_coordinates[:, 1], human_gaze_coordinates[:, 0], c='red', marker='x')

    # Set title and show the plot
    plt.title('Human Gaze Prediction')
    plt.savefig(output_path)
    plt.show()


def main():
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='local', help='local or global model')
    parser.add_argument('--participant', '-p', type=int, default=0, help='participant number')
    parser.add_argument('--data', '-d', type=str, default='data', help='path to data folder')
    args = parser.parse_args()
    model_type, participant, data_folder = args.model, args.participant, args.data

    n_units = 8160
    time = 700
    project_path = r"/content/drive/MyDrive/slca-main-pavan"
    #project_path = r"/content/drive/MyDrive/Colab-Notebooks/slca_project/slca-main-pavan"
    data_folder = f'{project_path}/data'
    stim_path = f'{project_path}/data/smaps/eml-net1'
    human_rt_path = f'{project_path}/data/all_rts.json'
    human_coords_path = f'{project_path}/data/all_coords.json'
    image_folder = f'{project_path}/data/smaps/eml-net1'
    gens_num = 2
    params = np.zeros(shape=(gens_num, 12, 8))

    if model_type == 'local':
        params_first = np.array([[0.256, 0.5, 0.37, 0.64, 0.097, 0.31, 1.043, 0.465],
                                  [0.256, 0.9, 0.25, 0.1, 0.097, 0.31, 3.0, 0.465],
                                  [0.256, 0.8, 0.25, 0.1, 0.097, 0.31, 4.0, 0.2]])
        params_diap = ((0.1, 0.5), (0.0005, 1.0), (0.05, 0.5), (0.1, 0.9), (0.2, 1.0), (0.0, 10.0), (0.2, 5.0), (0.05, 1.5))
    elif model_type == 'global':
        params_first = np.array([[0.4, 0.024, 0.4, 0.1, 1.0, 0.1, 1., 0.178],
                                  [0.3, 0.003, 0.4, 0.2, 0.8, 0.1, 2., 0.5],
                                  [0.25, 0.01, 0.4, 0.1, 0.9, 0.2, 3., 0.3]])
        params_diap = ((0.1, 0.5), (0.0005, 0.1), (0.05, 0.5), (0.1, 0.9), (0.2, 1.0), (0.0, 10.0), (0.2, 5.0), (0.05, 1.5))

    ga = GA(lca_model=model_type,gen_size=gen_size, time=time, n_units=n_units,fixed=fixed_parameters,params_range=params_range, stim_path=stim_path, human_rt_path=human_rt_path,
            human_coords_path=human_coords_path, params_diap=params_diap, part=participant)
    #initial parameter sets
    params = np.zeros(shape=(gens_num, gen_size, len(basic_loader.all_params)))
    print('init_params', params.shape)
    params[0] = ga.first_gen()
    
    for i, param_set in enumerate(params_init):
        params[0, i, list(param_set.keys())] = list(param_set.values())

    for g in range(1, gens_num):  # Start from 1 for next_gen
            logging.info(f'GENERATION {g}/{gens_num}')
            params[g] = ga.next_gen(g, params[g - 1])
            print(f'Shape of params[{g}]: {params[g].shape}')
            print(f'Shape of next_gen result: {ga.next_gen(g, params[g - 1]).shape}')

            logging.info(f'GENERATION {g} RES {list(params[g])}')

            # Initialize salient - replace 0.5 with the actual value you want to use
            salient = 0.5
            #  Visualize human gaze prediction for each image in a folder
            for image_file in os.listdir(image_folder):
                if image_file.endswith('.jpg') or image_file.endswith('.png'):
                    image_path = os.path.join(image_folder, image_file)
                    # Use LCA model initialization
                   # lca_model, salient, input_seq = ga.initialize_lca_model(params[g], salient)
                    
                    saliency_map = ga.compute_saliency_map(params[g], salient)
                    human_gaze_coordinates = np.array([[row, col] for row in range(68) for col in range(120)])
                    output_path = f'human_gaze_prediction_gen_{g}_{image_file.replace(".", "_")}.png'
                    visualize_human_gaze(image_path, saliency_map, human_gaze_coordinates, output_path=output_path)

        # Visualize salience map for the best parameters
        # Use your compute_saliency_map function
            best_saliency_map = ga.compute_saliency_map(best_inds[0], salient, input_seq)
            output_path = f'best_salience_map_gen_{g}.png'
            visualize_salience_map(best_saliency_map, output_path=output_path)
            
           
            plt.show()


if __name__ == '__main__':
    main()
