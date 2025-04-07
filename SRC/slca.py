import json
import logging
from matplotlib.image import imread
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import seaborn as sns
import scipy.stats as stats
import scipy.signal as signal
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

def local_inhibition(vector, w=120, h=68, pad=1):
    filt = [[-pad,-pad,-pad],[-pad,1,-pad],[-pad,-pad,-pad]]
    vector = np.reshape(vector,[h,w])
    m = np.pad(vector, (1,1), 'constant', constant_values=np.median(vector))
    return np.reshape(signal.convolve(m, filt, mode='same')[1:-1, 1:-1], [w*h,])

def auc_judd(s_map, gt):
    # thresholds are calculated from the salience map, only at places where fixations are present
    thresholds = []
    for i in range(0, gt.shape[0]):
        for k in range(0, gt.shape[1]):
            if gt[i][k] > 0:
                thresholds.append(s_map[i][k])

    gt[gt > 0] = 1
    num_fixations = np.sum(gt)
    # num fixations is no. of salience map values at gt >0

    thresholds = sorted(set(thresholds))

    area = [(0.0, 0.0)]
    for thresh in thresholds:
        # in the salience map, keep only those pixels with values above threshold
        temp = np.zeros(s_map.shape)
        temp[s_map >= thresh] = 1.0

        assert np.max(gt) == 1.0, 'something is wrong with ground truth..not discretized properly max value > 1.0'
        assert np.max(s_map) == 1.0, 'something is wrong with salience map..not normalized properly max value > 1.0'

        num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap with gt / total number of non fixated pixels
        # this becomes nan when gt is full of fixations..this won't happen
        fp = (np.sum(temp) - num_overlap) / ((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)

        area.append((round(tp, 4), round(fp, 4)))

    area.append((1.0, 1.0))
    area.sort(key=lambda x: x[0])
    tp_list = [x[0] for x in area]
    fp_list = [x[1] for x in area]
    return np.trapz(np.array(tp_list), np.array(fp_list))

class SLCA_global():
    """The leaky competing accumulator class in numpy.
    """

    def __init__(
            self, n_units, dt_t, leak, competition, self_excit=0,
            w_input=1, w_cross=0, offset=0, noise_sd=0, threshold=1):
        """Initialize a leaky competing accumulator.
        Parameters
        ----------
        n_units : int
            the number of accumulators in the LCA
        dt_t : float
            the dt / tao term, representing the time step size
        leak : float
            the leak term
        competition : float
            the lateral inhibition across accumulators (i vs. j)
        self_excit : float
            the self excitation of a accumulator (i vs. i)
        w_input : float
            input strengh of the feedforward weights
        w_cross : float
            cross talk of the feedforward weights
        offset : float
            the additive drift term of the LCA process
        noise_sd : float
            the sd of the noise term of the LCA process
        """
        self.threshold = threshold
        self.n_units = n_units
        self.dt_t = dt_t
        self.leak = leak
        self.competition = competition
        self.self_excit = self_excit
        self.w_input = w_input
        self.w_cross = w_cross
        self.offset = offset
        self.noise_sd = noise_sd
        # the input / recurrent weights
        self.W_i = make_weights(w_input, w_cross, n_units)
        self.W_r = make_weights(self_excit, -competition, n_units)
        # print('W_i', self.W_i.shape)
        # print('W_r', self.W_r.shape)
        # check params
        self._check_model_config()

    def run_2dim(self, stimuli, time=700):
        # input validation
        # self._check_inputs(stimuli, self.threshold)
        T, _ = np.shape(stimuli)

        # precompute noise for all time points
        # precompute the transformed input, for all time points
        inp = stimuli[0] @ self.W_i
        inp = np.repeat(inp[np.newaxis, ...], time, axis=0)
        # inp = np.transpose(inp, (1,0,2))
        sqrt_dt_t = np.sqrt(self.dt_t)
        # precompute offset for all units
        offset = self.offset * np.ones(self.n_units, )
        # precompute sqrt(dt/tao)
        trial_coord = np.zeros(T, dtype=np.float32) - 1
        trial_rt = np.zeros(T, dtype=np.float32) - 1
        V = np.zeros(shape=(T, time, self.n_units), dtype=np.float32)
        init_val = np.ones(self.n_units, dtype=np.float32)

        for trial in range(T):
            noise = np.random.normal(scale=self.noise_sd, size=(time, self.n_units)).astype(np.float16)
            # the LCA computation at time t
            for time_moment in range(time):
                V_prev = init_val if time_moment == 0 else V[trial, time_moment - 1, :]
                V[trial, time_moment, :] = V_prev + offset + noise[time_moment] * sqrt_dt_t + (inp[trial] - \
                                                                                               self.leak * V_prev + self.W_r @ V_prev) * self.dt_t
                V[trial, time_moment, :][V[trial, time_moment, :] < 0] = 0

                if np.any(V[trial, time_moment, :] > self.threshold):
                    coords = np.where(V[trial, time_moment, :] > self.threshold)[0]
                    trial_coord[trial] = coords[np.argmin(abs(coords - trial_coord[trial - 1]))]
                    trial_rt[trial] = time_moment
                    break

            if trial_coord[trial] == -1:
                trial_coord[trial] = trial_coord[trial - 1]

        return trial_coord, trial_rt

    def _check_model_config(self):
        if 0 >= self.leak:
            self.leak = 0.05
        if 0 >= self.competition:
            self.competition = 0.05
        if 0 >= self.self_excit:
            self.self_excit = 0.1
        if 0 >= self.dt_t:
            self.dt_t = 0.01
        if 0 >= self.noise_sd:
            self.noise_sd = 0.05
        if 0 >= self.threshold:
            self.threshold = 10

    def _check_inputs(self, stimuli, threshold):
        _, n_units_ = np.shape(stimuli)
        assert n_units_ == self.n_units, \
            f'stimuli shape inconsistent with the network size = {self.leak}'
        assert threshold > 0, \
            f'Invalid threshold = {threshold}'

    

class SLCA_local():
    """The leaky competing accumulator class in numpy.
    """
    def __init__(
            self, n_units, dt_t, leak, competition, self_excit=0,
            w_input=1, w_cross=0, offset=0, noise_sd=0, threshold=1):
        """Initialize a leaky competing accumulator.
        Parameters
        ----------
        n_units : int
            the number of accumulators in the LCA
        dt_t : float
            the dt / tao term, representing the time step size
        leak : float
            the leak term
        competition : float
            the lateral inhibition across accumulators (i vs. j)
        self_excit : float
            the self excitation of a accumulator (i vs. i)
        w_input : float
            input strengh of the feedforward weights
        w_cross : float
            cross talk of the feedforward weights
        offset : float
            the additive drift term of the LCA process
        noise_sd : float
            the sd of the noise term of the LCA process
        """
        self.threshold = threshold
        self.n_units = n_units
        self.dt_t = dt_t
        self.leak = leak
        self.competition = competition
        self.self_excit = self_excit
        self.w_input = w_input
        self.w_cross = w_cross
        self.offset = offset
        self.noise_sd = noise_sd
        # the input / recurrent weights
        self.W_i = make_weights(w_input, w_cross, n_units)
        self.W_r = make_weights(self_excit, 0, n_units)
        # print('W_i', self.W_i.shape)
        # print('W_r', self.W_r.shape)
        # check params
        self._check_model_config()
    
    def run_2dim(self, stimuli, time=700):
        # input validation
        # self._check_inputs(stimuli, self.threshold)
        T, _ = np.shape(stimuli)
        # precompute noise for all time points
        # precompute the transformed input, for all time points
        inp = stimuli[0] @ self.W_i
        inp = np.repeat(inp[np.newaxis, ...], time, axis=0)
        # inp = np.transpose(inp, (1,0,2))
        sqrt_dt_t = np.sqrt(self.dt_t)
        # precompute offset for all units
        offset = self.offset * np.ones(self.n_units, )
        # precompute sqrt(dt/tao)
        trial_coord = np.zeros(T, dtype=np.float32) - 1
        trial_rt = np.zeros(T, dtype=np.float32) - 1
        V = np.zeros(shape=(T, time, self.n_units), dtype=np.float32)
        init_val = np.ones(self.n_units, dtype=np.float32)

        for trial in tqdm(range(T)):
            import gc; gc.collect()
            print(f"STARTED trial/Total: {trial}/{T}")
            noise = np.random.normal(scale=self.noise_sd, size=(time, self.n_units)).astype(np.float16)
            # the LCA computation at time t
            for time_moment in range(time):
                V_prev = init_val if time_moment == 0 else V[trial, time_moment - 1, :]
                V[trial, time_moment, :] = local_inhibition(V_prev, 120, 68, self.dt_t) + offset + noise[time_moment] * sqrt_dt_t + (inp[trial] - self.leak * V_prev + self.W_r @ V_prev) * self.dt_t
                V[trial, time_moment, :][V[trial, time_moment, :] < 0] = 0
                if np.any(V[trial, time_moment, :] > self.threshold):
                    coords = np.where(V[trial, time_moment, :] > self.threshold)[0]
                    trial_coord[trial] = coords[np.argmin(abs(coords - trial_coord[trial - 1]))]
                    trial_rt[trial] = time_moment
                    break              
                
            if trial_coord[trial] == -1:
                trial_coord[trial] = trial_coord[trial - 1]
                print(f"DONE trial/Total: {trial}/{T}")
        return trial_coord, trial_rt

    def _check_model_config(self):
        if 0 >= self.leak:
            self.leak = 0.05
        if 0 >= self.competition:
            self.competition = 0.05
        if 0 >= self.self_excit:
            self.self_excit = 0.1
        if 0 >= self.dt_t:
            self.dt_t = 0.01
        if 0 >= self.noise_sd:
            self.noise_sd = 0.05
        if 0 >= self.threshold:
            self.threshold = 10

    def _check_inputs(self, stimuli, threshold):
        _, n_units_ = np.shape(stimuli)
        assert n_units_ == self.n_units, \
            f'stimuli shape inconsistent with the network size = {self.leak}'
        assert threshold > 0, \
            f'Invalid threshold = {threshold}'

def make_weights(diag_val, offdiag_val, n_units):
    """Get a connection weight matrix with "diag-offdial structure"
    e.g.
        | x, y, y |
        | y, x, y |
        | y, y, x |
    where x = diag_val, and y = offdiag_val
    Parameters
    ----------
    diag_val : float
        the value of the diag entries
    offdiag_val : float
        the value of the off-diag entries
    n_units : int
        the number of LCA nodes
    Returns
    -------
    2d array
        the weight matrix with "diag-offdial structure"
    """
    try:
        diag_mask = np.eye(n_units, dtype=np.float64)
        offdiag_mask = np.ones((n_units, n_units), dtype=np.float64) - np.eye(n_units, dtype=np.float64)
        weight_matrix = diag_mask * diag_val + offdiag_mask * offdiag_val
    except:
        print("float32 is out of range, using 16 bit precision")
        diag_mask = np.eye(n_units, dtype=np.float16)
        offdiag_mask = np.ones((n_units, n_units), dtype=np.float64) - np.eye(n_units, dtype=np.float64)
        weight_matrix = diag_mask * diag_val + offdiag_mask * offdiag_val
        weight_matrix = weight_matrix.astype(np.float32)
    return weight_matrix

diag_val = 1.0
offdiag_val = 0.5
n_units = 8
result = make_weights(diag_val, offdiag_val, n_units)
print(result)