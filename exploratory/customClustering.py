
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from typing import Dict

class DynamicSingleLinkage:
    def __init__(self, N_bins:int = 30, lowerMergerThreshold:float = 0.1, debug:bool = False):
        self.N_bins : int = N_bins
        self.lowerMergerThreshold : float = lowerMergerThreshold
        
        self.debug:bool = debug
        self.covercounter = 0
        self.debugData:Dict[int,object] = {}
        
        self.baseClusterer : AgglomerativeClustering = AgglomerativeClustering(
            distance_threshold=0.0,
            n_clusters=None,
            linkage="single")
        
        # the actually selected clusterer
        self.actualClusterer : AgglomerativeClustering = None

    def fit_predict(self, X):

        # TODO: implement the dendogram/histogram epsilon optimization
        #       use the base Clusterer for the dendograms
        # optimal_eps = 6.7
        # Build SciPy linkage matrix for the dendrogram

        model = self.baseClusterer.fit(X)
        n = len(X)
        counts = np.zeros(len(model.children_))

        merge_distance = np.percentile(model.distances_, self.lowerMergerThreshold * 100)

        for i, (left, right) in enumerate(model.children_):
            counts[i] = (
                (1 if left < n else counts[left - n])
                + (1 if right < n else counts[right - n])
            )

        linkage_matrix = np.column_stack([
            model.children_,
            model.distances_,
            counts,
        ])

        if self.debug:
            print("saving  dendogram/histogram")
            # TODO: save the dendograms, histograms, & chosen epsilon to investigate
            #       after the clustering is finished in the form:
            #self.debugData[self.covercounter] = {
            #    "dend": "the dendogram data",
            #    "hist": "the histogram data",
            #    "eps": optimal_eps             # the chosen optimal epsilon for the actualClusterer
            #}
        
        # covercounter is just for the debug data dictionary
        self.covercounter += 1
        
        # TODO: instantiate the actual clusterer to do the actual clustering
        #self.actualClusterer = AgglomerativeClustering(
        #    distance_threshold=optimal_eps,
        #    n_clusters=None,
        #    linkage="single")

        self.actualClusterer = AgglomerativeClustering(
            distance_threshold=merge_distance,
            n_clusters=None,
            linkage="single")
        return self.actualClusterer.fit_predict(X)

    def get_params(self):
        return self.baseClusterer.get_params()

    def __str__(self):
        return self.actualClusterer.__str__()
