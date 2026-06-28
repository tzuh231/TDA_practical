
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from typing import Dict

class DynamicSingleLinkage:

    
    def __init__(
            self,
            N_bins:int = 30,
            lowerMergerThreshold:float = 0.1,
            epsilon_strategy:str="with_threshold",
            debug:bool = False):

        # histogram based epsilon finder variables    
        self.N_bins : int = N_bins
        self.lowerMergerThreshold : float = lowerMergerThreshold
        # choose right epsilon finder strategy:
        #print(" epsilon strategy: ",epsilon_strategy)
        match(epsilon_strategy):
            case "only_threshold":
                self.calc_merge_distance = self.calc_merge_distance_only_from_threshold
            case "without_threshold":
                self.calc_merge_distance = self.calc_merge_distance_without_threshold
            case "with_threshold":
                self.calc_merge_distance = self.calc_merge_distance_with_threshold
            case _:
                self.calc_merge_distance = self.calc_merge_distance_only_from_threshold 

        # debugging related variables
        self.debug:bool = debug
        self.covercounter = 0
        self.debugData:Dict[int,object] = {}
        
        # base single linkage to build dendogram merge histograms
        self.baseClusterer : AgglomerativeClustering = AgglomerativeClustering(
            distance_threshold=0.0,
            n_clusters=None,
            linkage="single")
        
        # the actually selected clusterer
        self.actualClusterer : AgglomerativeClustering = None
    

    def fit_predict(self, X):
        # Use the base clusterer to compute the "optimal" epsilon
        model = self.baseClusterer.fit(X)
        n = len(X)
        
        merge_distance = self.calc_merge_distance(model.distances_, n)

        if self.debug:
            # Save the model and the optimal epsilon for debugging purposes -> histogram/dendogram
            # print("saving  dendogram/histogram")
            self.debugData[self.covercounter] = {
                "eps": merge_distance,
                "distances": model.distances_,
                "n":X.shape[0]
            }
        
        # covercounter is just for the debug data dictionary
        self.covercounter += 1
        
        #instantiate the actual clusterer to do the actual clustering

        self.actualClusterer = AgglomerativeClustering(
            distance_threshold=merge_distance,
            n_clusters=None,

            linkage="single")
        return self.actualClusterer.fit_predict(X)


    def calc_merge_distance_with_threshold(self, distances, n):

        counts, bin_edges = np.histogram(distances,bins=self.N_bins)
        sum_cnts = 0
        merge_distance = None
        for i, cnt in enumerate(counts):
            sum_cnts += cnt
            if sum_cnts > (n * self.lowerMergerThreshold):
                if cnt == 0:
                    merge_distance = bin_edges[i]
                    break
        
        if merge_distance is None:
            merge_distance = bin_edges[-1]

        return merge_distance


    def calc_merge_distance_without_threshold(self, distances, n):

        counts, bin_edges = np.histogram(distances,bins=self.N_bins)
        sum_cnts = 0
        merge_distance = None
        for i, cnt in enumerate(counts):
            sum_cnts += cnt
            if sum_cnts > 0: #sum_cnts > (n * self.lowerMergerThreshold):
                if cnt == 0:
                    merge_distance = bin_edges[i]
                    break
        
        if merge_distance is None:
            merge_distance = bin_edges[-1]
            
        return merge_distance
    

    def calc_merge_distance_only_from_threshold(self, distances, n):
        merge_distance = np.percentile(distances, self.lowerMergerThreshold * 100)
        return merge_distance


    def get_params(self):
        return self.baseClusterer.get_params()


    def __str__(self):
        return self.actualClusterer.__str__()
