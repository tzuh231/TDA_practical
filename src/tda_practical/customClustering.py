
from sklearn.cluster import AgglomerativeClustering

class DynamicSingleLinkage:
    def __init__(self, N_bins:int = 30, lowerMergerThreshold:float = 0.1):
        self.N_bins = N_bins
        self.lowerMergerThreshold = lowerMergerThreshold
        self.baseClusterer = AgglomerativeClustering(
            distance_threshold=0.0,
            n_clusters=None,
            linkage="single")
        
        # the actually selected clusterer
        self.actualClusterer = None

    def fit_predict(self, X):
        # TODO: implement the dynamic clustering based on histograms in this function
        #       for now I just forward the base clustering
        
        self.actualClusterer = self.baseClusterer
        return actualClusterer.fit_predict(X)

    def get_params(self):
        return self.baseClusterer.get_params()

    def __str__(self):
        return self.actualClusterer.__str__()
