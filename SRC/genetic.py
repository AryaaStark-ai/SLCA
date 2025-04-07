from PIL.Image import NONE
from google.colab import output
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
from scipy.stats import anderson
from scipy.ndimage import gaussian_filter
import multiprocessing as mp
import json
import logging
import os
from slca import *
from utils import *
from measure import *
import traceback
from matplotlib.image import imread
import matplotlib.pyplot as plt
import seaborn as sns

class GA():


    def __init__(self, lca_model, n_units, time, params_diap, human_rt_path, human_coords_path,fixed,stim_path,params_range,part,pop_size=1,gen_size=3):
        self.params_diap = params_diap
        self.time = time
        self.n_units = n_units


        self.dt_t = 0.1
        self.threshold = 5.0

        self.n_desc = int(gen_size / 4)
        self.input_seq = None

        self.human_rt = self.load_human_data(human_rt_path)
        self.human_coords = self.get_human_gaze(human_coords_path)
        self.stim_path = stim_path
        self.stim = self.load_stim(stim_path)
        self.fixed = fixed
        self.params_range = params_range
        self.unfixed_inds = tuple(set(range(min(len(self.params_range),len(self.params_diap)))) - set(self.fixed.keys()))
        self.part = part
        self.lca_model = lca_model
        self.output = output
        self.gen_size = gen_size
       

     


    def load_human_data(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

    def get_human_gaze(self, human_coords_path):
        human_coords = self.load_human_data(human_coords_path)
        data = {}
        for stim in human_coords:
            data[stim] = {}

            for participant in human_coords[stim]:
                gazepoints = human_coords[stim][participant]
                if len(gazepoints) > 0:
                    lens = self.human_rt[stim][participant]
                    gaze = np.zeros((68, 120))
                    for i in range(len(gazepoints)):
                        y, x = gazepoints[i]
                        try:
                            gaze[int(x / 16)][int(y / 16)] += lens[i]
                        except:
                            pass
                    gaze = (gaze - np.min(gaze)) / (np.max(gaze) - np.min(gaze))
                    data[stim][participant] = np.array(gaze, dtype=np.float64)
        return data

    def load_stim(self, stim_folder):
        pics = {}
        for pic in self.human_coords:
            im_path = f'{stim_folder}/{pic[:-4]}_smap.png'
            if os.path.isfile(im_path):
                eml_net = imread(im_path)
                eml_net = eml_net[::16, ::16]
                eml_net = (eml_net - np.min(eml_net)) / (np.max(eml_net) - np.min(eml_net))
                pics[pic] = {'flat': eml_net.flatten(), 'map': eml_net, 'salient': len(eml_net[eml_net > 0.6])}
        return pics

    # def find_pareto_efficient(self, costs):
    #     best_inds = np.arange(costs.shape[0])
        
    #     for i, c in enumerate(costs):
    #       best_inds[i] = sum([list(p[:,j]).index(i) for j in len(costs[0])], []) 
        
    #     return best_inds.argsort()
    best_inds=[]
    costs=[]
    def find_pareto_efficient(self, costs):
        p = costs.argsort(0)
        best_inds = np.ones(len(p))

            # Check if costs array is empty, return an empty array
        if len(costs) == 0:
          return np.array([])
        for i in range(len(p)):
            best_inds[i] = sum([list(p[:,j]).index(i) for j in range(len(costs[0]))])   
        best_inds = best_inds.argsort()
        
        # # fill with other variants if there are <3 optimal
        # if len(best_inds) < self.n_desc():
        #     fits_sorted = np.array([ind for ind in np.mean(costs, axis=1).argsort() if ind not in best_inds])
        #     best_inds = np.r_[best_inds, fits_sorted[:self.n_desc - len(best_inds)]]
        return best_inds


    def calculate_fitness(self, parameters, stim):
        # Directory for saving output
        output_dir = '/content/output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        all_coords,ad_stat, all_rt, all_ks, all_aj, all_human_rt = [], [], [], [], [],[]

        self.input_seq = []

        for stim_key in list(self.stim.keys()):
            salient = self.stim[stim_key]['salient']

            T = 20
            input_seq = np.tile(self.stim[stim_key]['flat'], (T, 1))
            self.input_seq = input_seq
            threshold = self.threshold + salient * parameters[7]
            print('new thresh ' + str(threshold))

            if self.lca_model == 'local':
                lca = SLCA_local(n_units=self.n_units, dt_t=self.dt_t, leak=parameters[0], competition=parameters[1],
                                self_excit=parameters[2], w_input=parameters[3], w_cross=parameters[4], offset=parameters[5],
                                noise_sd=parameters[6].astype(np.float64), threshold=threshold)
            elif self.lca_model == 'global':
                lca = SLCA_global(n_units=self.n_units, dt_t=self.dt_t, leak=parameters[0], competition=parameters[1],
                                  self_excit=parameters[2], w_input=parameters[3], w_cross=parameters[4], offset=parameters[5],
                                  noise_sd=parameters[6].astype(np.float64), threshold=threshold)

            print(mp.current_process().name + ' LCA started ' + stim_key + ' ' + ', '.join(
                [str(p) for p in parameters]) + ' ' + str(threshold))
            coords, rt = lca.run_2dim(stimuli=input_seq)
            import gc;
            gc.collect()
            print(mp.current_process().name + ' LCA finished ' + stim_key + ' ' + ', '.join([str(p) for p in rt]))

            trial_coord = np.array(coords, dtype=np.float64)
            trials_h = list(trial_coord // 120)
            trials_w = list(trial_coord % 120)

            lca_map = np.zeros(shape=(68, 120), dtype=np.float64)
            for i in range(len(rt)):
                lca_map[int(trials_h[i]), int(trials_w[i])] = rt[i]
            lca_smap = gaussian_filter(lca_map, sigma=1.5)
            lca_smap = (lca_smap - np.min(lca_smap)) / (np.max(lca_smap) - np.min(lca_smap))
            
            all_coords.append(coords)
            all_rt.extend(rt)
            participant = list(self.human_rt[stim_key].keys())[self.part]
            #iterate over participants
            # for participant in self.data.human_rt[stim]:
            #     if participant in self.participants and len(self.data.human_rt[stim][participant]) == 0:
            #         print(f'NO STIM {stim} for participant {participant}')
            #     elif participant in self.participants:
            if 2 > 1:
                if len(self.human_rt[stim_key][participant]) == 0:
                    print(f'NO STIM {stim_key} for participant {participant}')
                else:
                    try:
                        ks = stats.ks_2samp(self.human_rt[stim_key][participant], rt)
                    except Exception as e:
                           ks = None
                           print(
                                    f'ks stim {stim_key}, participant {participant}:: {str(e)}. RT {rt}, human RT {self.human_rt[stim_key][participant]}')

                if ks and ks.statistic:
                      all_ks.append(ks.statistic)
                else:
                    all_ks.append(0.99999)

                all_human_rt.extend(self.human_rt[stim_key][participant])

                if participant in self.human_coords[stim_key]:
                      try:
                          aj = auc_judd(lca_smap, self.human_coords[stim_key][participant])
                          print(
                                        f'aj GOOD stim {stim_key}, participant {participant}:: SM max {np.max(lca_smap)}, min: {np.min(lca_smap)}, mean {np.mean(lca_smap)}')
                      except Exception as e:
                                    aj = 0.00001
                                    print(
                                        f'aj stim {stim_key}, participant {participant}:: {str(e)}. SM max {np.max(lca_smap)}, min {np.min(lca_smap)}, mean {np.mean(lca_smap)}')

                      print(
                                f'stim {stim_key}, participant {participant}, params {parameters}:: ks {ks}, 1-aj {1 - aj}')
                      all_aj.append(1 - aj)

                      # Anderson-Darling test
                            
                if participant in self.human_coords[stim_key]:
                      try:
                         aj = auc_judd(lca_smap, self.human_coords[stim_key][participant])
                         print(
                                      f'aj GOOD stim {stim_key}, participant {participant}:: SM max {np.max(lca_smap)}, min: {np.min(lca_smap)}, mean {np.mean(lca_smap)}')

                        # Anderson-Darling test
                         ad_stat, ad_critical_values, ad_significance_level = anderson(rt, dist='norm')
                         print(
                                        f'Anderson-Darling stat: {ad_stat}, critical values: {ad_critical_values}, significance level: {ad_significance_level}')

                      except Exception as e:
                                    aj = 0.00001
                                    print(
                                        f'aj stim {stim_key}, participant {participant}:: {str(e)}. SM max {np.max(lca_smap)}, min {np.min(lca_smap)}, mean {np.mean(lca_smap)}')

                      print(
                                f'stim {stim_key}, participant {participant}, params {parameters}:: ks {ks}, 1-aj {1 - aj}, AD stat {ad_stat}')

            all_aj.append(1 - aj)
            # Create a unique identifier for each parameter set
            param_set_id = f"params_{np.array2string(parameters, precision=2, separator='_')}"
            
            # Visualization of the saliency map for each stim and parameter set
            saliency_map_output_path = os.path.join(output_dir, f'{param_set_id}_saliency_map_{stim_key}.png')
            self.visualize_salience_map(lca_smap, output_path=saliency_map_output_path)

            # Visualization of human gaze for each stim and parameter set
            if participant in self.human_coords[stim_key]:
                human_gaze_output_path = os.path.join(output_dir, f'{param_set_id}_human_gaze_{stim_key}.png')
                image_path = f'{self.stim_path}/{stim_key.split(".")[0]}'  + "_smap.png"
                self.visualize_human_gaze(image_path, lca_smap, self.human_coords[stim_key][participant], output_path=human_gaze_output_path)

        all_human_rt = np.random.choice([el for el in all_human_rt if 50 <= el <= 750], 50)
        if len(all_rt) > 50:
            all_rt = np.random.choice(all_rt, 50)

        ks_fin = stats.ks_2samp(all_human_rt, all_rt).statistic
        res = {'params': parameters, 'fitness': (np.mean(all_ks), np.mean(all_aj))}
        return res


    def process_fit(self, param_set, output, idx):
        print(f'{mp.current_process().name} STARTED params_set {idx}')
        res = self.calculate_fitness(param_set,self.stim)
        print(f'{mp.current_process().name} FINISHED params_set {idx}')
        output.put(res)
        return res

    def select_best(self, sets):
        output = mp.Queue()
      
        processes = []

        for i in range(len(sets)):
            p = mp.Process(target=self.process_fit, args=(sets[i], output, i))
            processes.append(p)
            print(f'{p.name} started')
            p.start()
        for p in processes:
            p.join()

        results = []
        while not output.empty():
            o = output.get()
            results.append(o)

        fits = np.array([el['fitness'] for el in results])
        print(f'fits {len(fits)} {fits}')

        # find pareto optimal
        print('started computing pareto')
        best_inds = self.find_pareto_efficient(fits)
        print('best_pareto ' + ' '.join([str(i) for i in best_inds]))

        # fill with other variants if there are <3 optimal
        if len(best_inds) < 3:
            fits_sorted = np.array([ind for ind in np.mean(fits, axis=1).argsort() if ind not in best_inds])
            best_inds = np.r_[best_inds, fits_sorted[:3]]

        results = np.array(results)
        print(f'BEST RES: {results[best_inds]}')
        return np.array([res['params'] for res in results[best_inds]])

        #       # find best descendants
        # if self.n_desc == 1:
        #     best_inds = fits.argsort()[-self.n_desc:][::-1]
        # else:
        #     print('started computing pareto')
        #     best_inds = self.find_pareto_efficient(fits)[:self.n_desc]


    def random_gen(self, sets_num):
        gen = np.array([np.random.uniform(low=self.params_range[d][0], high=self.params_range[d][1], size=sets_num) for d in self.params_range]).T
        gen[:,list(self.fixed.keys())] = list(self.fixed.values())
        return gen
    def first_gen(self):
        return self.random_gen(self.gen_size)

    def mutate(self, set_params):
        local_unfixed_inds = tuple(set(range(min(len(self.params_range),len(self.params_diap)))) - set(self.fixed.keys()))
        param_num = np.random.choice(local_unfixed_inds)
        new_params = set_params.copy()
        new_params[param_num] *= np.random.choice([0.95, 1.05])
        if new_params[param_num] < self.params_diap[param_num][0]:
            new_params[param_num] = self.params_diap[param_num][0]
        elif new_params[param_num] > self.params_diap[param_num][1]:
            new_params[param_num] = self.params_diap[param_num][1]
        print("param_num:", param_num)
        print("length of self.params_diap:", len(self.params_diap))
        return new_params


    def crossover(self, parent0, parent1):
        cross_ind = np.random.choice(self.unfixed_inds)
        child = parent0.copy()
        child[cross_ind:] = parent1[cross_ind:]
        return child

    def next_gen(self, g, all_sets):
        print(f'GENERATION {g}')
        print('REMOVING DUPLICATES')
        unq, count = np.unique(all_sets, axis=0, return_counts=True)
        repeated_groups = unq[count > 1]
        for repeated_group in repeated_groups:
            repeated_idx = np.argwhere(np.all(all_sets == repeated_group, axis=1))
            repeated_idx = repeated_idx.ravel()[1:]
            all_sets[repeated_idx] = np.apply_along_axis(self.mutate, 1, all_sets[repeated_idx])

        print('SELECTION')
        gens_best = self.select_best(all_sets)
        print(f'gens_best shape, {gens_best.shape}')

        print('MUTATION')
        gens_mutated = np.apply_along_axis(self.mutate, 1, gens_best)
        print(f'mutated shape, {gens_mutated.shape}')

        print('CROSSOVER')
        parents0 = gens_best.copy()
        parents1_inds = [np.random.choice(np.delete(np.arange(len(gens_best)), [i]))
                         for i in range(len(gens_best))]
        parents1 = gens_best[parents1_inds]

        gens_crossover = np.array([self.crossover(parents0[i], parents1[i]) for i in range(len(parents0))])
        print(f'crossover shape, {gens_crossover.shape}')

        print('RANDOM')
        gens_random = self.random_gen(self.n_desc)
        print(f'random shape, {gens_random.shape}')

        gens = np.concatenate([gens_best, gens_mutated, gens_crossover, gens_random])
        print(f'all shape, {gens.shape}')
        return gens , gens_best
    
    def initialize_lca_model(self, params, salient):
        # Implement your LCA model initialization here based on the given parameters
        # Return the initialized LCA model, salient value, and input sequence\
        print(params)
        threshold = self.threshold + salient * params[7]

        if self.lca_model == 'local':
            lca = SLCA_local(n_units=self.n_units, dt_t=self.dt_t, leak=params[0], competition=params[1],
                             self_excit=params[2], w_input=params[3], w_cross=params[4], offset=params[5],
                             noise_sd=params[6].astype(np.float64), threshold=threshold)
        elif self.lca_model == 'global':
            lca = SLCA_global(n_units=self.n_units, dt_t=self.dt_t, leak=params[0], competition=params[1],
                              self_excit=params[2], w_input=params[3], w_cross=params[4], offset=params[5],
                              noise_sd=params[6].astype(np.float64), threshold=threshold)

        return lca, salient

    def compute_saliency_map(self, lca_model, salient,input_seq):
        # Implement your saliency map computation using the initialized LCA model
        # Return the computed saliency map
        coords, rt = lca_model.run_2dim(stimuli=self.input_seq)
        import gc
        gc.collect()

        trial_coord = np.array(coords, dtype=np.float32)
        trials_h = list(trial_coord // 120)
        trials_w = list(trial_coord % 120)

        lca_map = np.zeros(shape=(68, 120), dtype=np.float32)
        for i in range(len(rt)):
            lca_map[int(trials_h[i]), int(trials_w[i])] = rt[i]
        lca_smap = gaussian_filter(lca_map, sigma=1.5)
        lca_smap = (lca_smap - np.min(lca_smap)) / (np.max(lca_smap) - np.min(lca_smap))

        return lca_smap

    # Function to visualize the salience map
    def visualize_salience_map(self, saliency_map, output_path='salience_map.png'):
        # Visualize and save the salience map
        plt.imshow(saliency_map, cmap='viridis')
        plt.title('Saliency Map')
        plt.savefig(output_path)
        # plt.show()

    # Function to visualize human gaze prediction
    def visualize_human_gaze(self, image_path, saliency_map, human_gaze_coordinates, output_path='human_gaze_prediction.png'):
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
        # plt.show()
