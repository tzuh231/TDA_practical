import pandas as pd
import numpy as np
import kmapper as km
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import networkx as nx
from kmapper.adapter import to_networkx
from customClustering import DynamicSingleLinkage


df_data = pd.read_csv("./data/processed_data.csv")

vote_matrix_09 = df_data[df_data['date'] < '2010-01-01'].pivot(index='icpsr', columns='rollnumber', values='paper_cast_code').fillna(0)
vote_matrix_10 = df_data[df_data['date'] >= '2010-01-01'].pivot(index='icpsr', columns='rollnumber', values='paper_cast_code').fillna(0)

corr_09 = vote_matrix_09.T.corr()
corr_10 = vote_matrix_10.T.corr()

distance_09 = (1 - corr_09).fillna(0)
distance_10 = (1 - corr_10).fillna(0)

U_09, singular_values_09, Vt_09 = np.linalg.svd(distance_09.to_numpy())
U_10, singular_values_10, Vt_10 = np.linalg.svd(distance_10.to_numpy())

lens_09 = (U_09 @ np.diag(singular_values_09))[:,:2]
lens_10 = (U_10 @ np.diag(singular_values_10))[:,:2]


mapper = km.KeplerMapper()

cover = km.Cover(
    n_cubes=120,
    perc_overlap=0.77,
)
#clusterer = AgglomerativeClustering(
#    distance_threshold=0.5,
#    n_clusters=None,
#    linkage="single")
N_bins = 20
clusterer = DynamicSingleLinkage(
    N_bins = N_bins,
    lowerMergerThreshold = 0.50,
    debug = True
)

#clusterer = DBSCAN(
#    metric="precomputed",
#    eps=2.5,
#    min_samples=1,
#)

graph_09 = mapper.map(
    lens_09,
    X=distance_09.to_numpy(),
    precomputed=True,
    cover=cover,
    clusterer=clusterer,
    remove_duplicate_nodes=True,
)
graph_10 = mapper.map(
    lens_10,
    X=distance_10.to_numpy(),
    precomputed=True,
    cover=cover,
    clusterer=clusterer,
    remove_duplicate_nodes=True,
)

print("here")
N_covers = len(clusterer.debugData.keys())

n_datas = []

for i,(k,v) in enumerate(clusterer.debugData.items()):
    n_datas.append(v["n"])
    if v["n"] >20:
        fig,ax = plt.subplots()

        n, counts,bins = ax.hist(v["distances"],bins=N_bins)
        print(i,"/",N_covers)#,v["n"],n)
        ax.vlines(v["eps"],ymin=0,ymax=np.max(counts),color="red",linestyle="dashed")
        plt.show()

fig,ax = plt.subplots()
ax.hist(n_datas)
plt.show()