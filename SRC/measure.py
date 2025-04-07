import numpy as np
import scipy.stats as stats

class Measure:
    """
    Base class for the performance measurument
    """
    def __init__(self, sim, human):
        """
        Parameters 
        ----------
        sim
            simulated data
        human
            human data
        """
        self.human = human
        self.sim = sim
        
    def _count(self):
        """
        Calculate the metric
        
        Returns
        ----------
        np.float32
            the metric value
        """
        return np.mean(abs(np.array(self.human) - np.array(self.sim)))
        
    def run(self):
        """
        Tries to calculate the metric. If it is impossible, returns None
        """
        try:
            res = 1 - self._count()
        except Exception as e:
            res = None
        return res
        
class KolmogorovSmirnov(Measure):
    """
    A non-parametric test for comparing two discrete or continuous distributions
    """
    def __init__(self, sim_rt, human_rt):
        """
        Parameters 
        ----------
        sim_rt
            the sequence of simulated reaction times
        human_rt 
            the sequence of human reaction times
        """
        super().__init__(sim_rt, human_rt)
    
    def _count(self):
        ks = stats.ks_2samp(self.human, self.sim)
        return ks.statistic

class AucJudd(Measure):
    """
    A spatial metric for comparing a saliency map with ground scanpath
    """
    def __init__(self, sim_smap, human_coords):
        """
        Parameters
        ----------
        sim_smap
            the simulated saliency map
        human_coords
            the coordinates of ground truth human scanpath
        """
        super().__init__(sim_smap, human_coords)
    
    def _count(self):
        # thresholds are calculated from the salience map, only at places where fixations are present
        thresholds = []
        for i in range(0, self.human.shape[0]):
            for k in range(0, self.human.shape[1]):
                if self.human[i][k] > 0:
                    thresholds.append(self.sim[i][k])

        self.human[self.human > 0] = 1
        num_fixations = np.sum(self.human)
        # num fixations is no. of salience map values at gt >0

        thresholds = sorted(set(thresholds))

        area = [(0.0, 0.0)]
        for thresh in thresholds:
            # in the salience map, keep only those pixels with values above threshold
            temp = np.zeros(self.sim.shape)
            temp[self.sim >= thresh] = 1.0

            assert np.max(self.human) == 1.0, 'something is wrong with ground truth..not discretized properly max value > 1.0'
            assert np.max(self.sim) == 1.0, 'something is wrong with salience map..not normalized properly max value > 1.0'

            num_overlap = np.where(np.add(temp, self.human) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)

            # total number of pixels > threshold - number of pixels that overlap with gt / total number of non fixated pixels
            # this becomes nan when gt is full of fixations..this won't happen
            fp = (np.sum(temp) - num_overlap) / ((np.shape(self.human)[0] * np.shape(self.human)[1]) - num_fixations)

            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]
        return np.trapz(np.array(tp_list), np.array(fp_list))