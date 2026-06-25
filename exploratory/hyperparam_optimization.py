import pandas as pd
import numpy as np
import kmapper as km
from kmapper.visuals import _format_mapper_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
from kmapper.adapter import to_networkx
from sklearn.decomposition import PCA
from customClustering import DynamicSingleLinkage
from matplotlib.colorizer import Colorizer, ColorizingArtist

#### load the data ####
df_data = pd.read_csv("./data/processed_data.csv")

member_info = (
    df_data[["icpsr", "party_name", "bioname"]]
    .drop_duplicates("icpsr")
    .set_index("icpsr")
)

party_color = (
    member_info["party_name"]
    .map({
        "Democrat": 0.0,
        "Republican": 1.0,
    })
    .fillna(0.5)
    .to_numpy()
)

#### prepare the latent PCA representation ####
#If the member has not existed in the house yet, we assign their vote to 0
vote_matrix_09 = df_data[df_data['date'] < '2010-01-01'].pivot(
    index='icpsr', columns='rollnumber', values='paper_cast_code'
    ).fillna(0)
vote_matrix_10 = df_data[df_data['date'] >= '2010-01-01'].pivot(
    index='icpsr', columns='rollnumber', values='paper_cast_code'
    ).fillna(0)

X_09 = vote_matrix_09.to_numpy()
pca_09 = PCA(n_components=2)
pc_09 = pca_09.fit_transform(X_09)
print("expl.var 09:", pca_09.explained_variance_ratio_)


X_10 = vote_matrix_10.to_numpy()
pca_10 = PCA(n_components=2)
pc_10 = pca_10.fit_transform(X_10)
print("expl.var 10:", pca_10.explained_variance_ratio_)

corr_09 = vote_matrix_09.T.corr()
corr_10 = vote_matrix_10.T.corr()
distance_09 = (1 - corr_09).fillna(0)
distance_10 = (1 - corr_10).fillna(0)
lens_09 = pc_09  #(U_09 @ np.diag(singular_values_09))[:,:2]
lens_10 = pc_10  #(U_10 @ np.diag(singular_values_10))[:,:2]


#### build the mapper graphs ####

mapper = km.KeplerMapper()

cover = km.Cover(
    n_cubes=50, #120,
    perc_overlap=1-(1/4),
)

clusterer = AgglomerativeClustering(
    distance_threshold=0.5,
    n_clusters=None,
    linkage="single")
#clusterer = DynamicSingleLinkage(
#    N_bins = 52,
#    lowerMergerThreshold = 0.1,
#    debug = False,
#    epsilon_strategy="without_threshold"
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

G_09 = to_networkx(graph_09)
G_10 = to_networkx(graph_10)

n_components_09_no_singletons = sum(
    1 for comp in nx.connected_components(G_09) if len(comp) > 1
)

print(n_components_09_no_singletons)

n_components_10_no_singletons = sum(
    1 for comp in nx.connected_components(G_10) if len(comp) > 1
)
print(n_components_10_no_singletons)



#### visualization ####
fig, axs = plt.subplots(1,2)

def draw_nx_graph(graph, ax, party_color,title):
    nx.draw(
        graph, ax=ax,
        node_color = [ np.mean(party_color[v['membership']]) for k,v in graph.nodes._nodes.items()],
        node_size = 5, cmap="bwr"
    )
    ax.set_title(title)
def add_cbar_right_of_axis(ax):
    cr = Colorizer(cmap="bwr")
    cr.set_clim(vmin=0.0,vmax=1.0)

    fig.colorbar(ColorizingArtist(cr),ax=ax,label="Democratic <- (member ratio) -> Repuplican")

draw_nx_graph(G_09,axs[0],party_color,title="2009 single linkage")
draw_nx_graph(G_10,axs[1],party_color,title="2010 single linkage")
add_cbar_right_of_axis(axs[1])

plt.show()