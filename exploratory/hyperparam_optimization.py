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
from typing import Dict, Any, Callable,Tuple
from tqdm import tqdm
import random
import pandas as pd
from datetime import datetime



#### define pipelining utility for grouped pipelining ####

def apply_func_to_year_dict_on_value(year_dict:Dict[str,Any], func:Callable[Any,Any]):
    return {year:func(item) for year,item in year_dict.items()}

def apply_func_to_year_dict_on_key(year_dict:Dict[str,Any], func:Callable[Any,Any]):
    return {year:func(year) for year,item in year_dict.items()}

def apply_func_to_year_dict_on_item(year_dict:Dict[str,Any], func:Callable[Any,Any]):
    return {year:func(year,item) for year,item in year_dict.items()}

def apply_func_to_nested_grouping(outer_group, inner_group, func):
    return {
        k_out: {
            k_inn: func(k_out, v_out, k_inn, v_inn)
            for k_inn,v_inn in inner_group.items()
        } for k_out, v_out in outer_group.items()}

#### define the clustering pipeline ####
class ClusteringPreperation:
    def __init__(
            self,
            df_data,
            year:str,
            data_bounds:Tuple[str,str]):

        self.year = year
        df_votes = df_data[(df_data['date'] >= data_bounds[0]) & (df_data['date'] < data_bounds[1])]

        #If the member has not existed in the house yet, we assign their vote to 0
        self.vote_matrix = df_votes.pivot(
            index='icpsr',
            columns='rollnumber',
            values='paper_cast_code' ).fillna(0)
        self.distance_matrix = (
            1 - self.vote_matrix.T.corr() ).fillna(0).to_numpy()


        member_info = (
            df_votes[["icpsr", "party_name", "bioname"]]
            .drop_duplicates("icpsr")
            .set_index("icpsr"))
        self.party_color = (member_info["party_name"]
            .map({
                "Democrat": 0.0,
                "Republican": 1.0,
            })
            .fillna(0.5)
            .to_numpy()
        )

        self.pca = PCA( n_components=2 )
        self.lense = self.pca.fit_transform( self.vote_matrix.to_numpy() )

class ClusterPipeline:
    def __init__(
            self,
            cluster_data:ClusteringPreperation,
            mapper,
            clusterer,
            cover,
            name:str,
            hyperparameters = None):

        self.year = cluster_data.year
        self.name = name
        self.party_color = cluster_data.party_color
        self.hyperparameters = hyperparameters

        self.km_graph = mapper.map(
            cluster_data.lense,
            X=cluster_data.distance_matrix,
            precomputed=True,
            cover=cover,
            clusterer=clusterer,
            remove_duplicate_nodes=True )
        self.nx_graph = to_networkx(self.km_graph)
        self.n_components_bigger_1 = sum(
            1 for comp in nx.connected_components(self.nx_graph)
            if len(comp) > 1)
    
    def get_results(self):
        return self.n_components_bigger_1

    def draw_nx_graph(self,ax):
        nx.draw(
            self.nx_graph, ax=ax,
            node_color = [ np.mean(self.party_color[v['membership']]) for k,v in self.nx_graph.nodes._nodes.items()],
            node_size = 5, cmap="bwr"
        )
        ax.set_title(f"{self.year} {self.name}")
    

#### load the data ####
df_data = pd.read_csv("./data/processed_data.csv")

#### define groupings ####
year_dates = {
    "2009": ('2009-01-01','2010-01-01'),
    "2010": ('2010-01-01','2011-01-01'),
}
sorted_year_keys = sorted(year_dates.keys(),key=int)
year_indx = apply_func_to_year_dict_on_key(
    year_dates,
    lambda year: sorted_year_keys.index(year)
)

clustering_data_per_year = apply_func_to_year_dict_on_item(
    year_dates,
    lambda year, bounds: ClusteringPreperation(df_data,year,bounds)
)

clusterer_generators = {
    "sing.link.": lambda eps: AgglomerativeClustering(
        distance_threshold=eps,
        n_clusters=None,
        linkage="single"),

    "only_threshold": lambda t_lm: DynamicSingleLinkage(
        lowerMergerThreshold = t_lm,
        epsilon_strategy="only_threshold"),
    
    "without_threshold": lambda N_bins: DynamicSingleLinkage(
        N_bins = N_bins,
        epsilon_strategy="without_threshold"),
    
    "with_threshold": lambda N_bins,t_lm: DynamicSingleLinkage(
        N_bins = N_bins,
        lowerMergerThreshold = t_lm,
        epsilon_strategy="with_threshold"),
}

clusterer_good_settings = {
    "sing.link.": {"eps": 0.5},
    "only_threshold": {"t_lm": 0.25},
    "without_threshold": {"N_bins": 52},
    "with_threshold": {"N_bins": 52, "t_lm": 0.25},
}

clusterers = apply_func_to_year_dict_on_item(
    clusterer_generators,
    lambda name, clusterer_gen: clusterer_gen(**(clusterer_good_settings[name]))
)

clusterer_idx = {name:i for i,name in enumerate(clusterers.keys())}


#### build the mapper graphs ####
mapper = km.KeplerMapper()
cover = km.Cover(
    n_cubes=120,
    perc_overlap=1-(1/4),
)

results = {"trial":[],"name":[],"params":[]}
for year in year_dates.keys():
    results[year] = []

for trial in tqdm(range(100)):
    clusterer_settings = {
        "sing.link.": {"eps": random.uniform(0.1,0.5)}, # covers are maximaly 1/3 wide in pca coordinates
        "only_threshold": {"t_lm": random.uniform(0.1,0.9)},               
        "without_threshold": {"N_bins": random.randint(2,50)},
        "with_threshold": {"N_bins": random.randint(2,50), "t_lm": random.uniform(0.1,0.9)},
    }
    clusterings_pc_py = apply_func_to_nested_grouping(
        clusterers, clustering_data_per_year,
        lambda name, clusterer, year,clustering_data: ClusterPipeline(
            clustering_data, mapper, clusterer, cover, name, clusterer_settings[name]
        )
    )
    for name,clustering_per_year in clusterings_pc_py.items():
        results["trial"] = trial
        results["name"].append(name)
        results["params"].append(clusterer_settings[name])
        for year, clustering in clustering_per_year.items():
            results[year].append(clustering.n_components_bigger_1)

results_df = pd.DataFrame(results)

target_09 = 10
target_10 = 37

results_df["loss"] = np.abs(results_df["2009"] - target_09) + np.abs(results_df["2010"] - target_10)

results_df.to_csv(f"./data/opti_results_{datetime.now().strftime('%y-%m-%d_%H-%M')}.csv")


#### report important metrics ####
print("explained pca component variance:\n",apply_func_to_year_dict_on_value(
    clustering_data_per_year,
    lambda clustering:  clustering.pca.explained_variance_ratio_
))
print("number of clusters w\ #nodes > 1:\n",apply_func_to_nested_grouping(
    clusterers,year_dates,
    lambda name,x,year,y:  clusterings_pc_py[name][year].get_results()
))

#### visualization ####
fig, axs = plt.subplots(len(clusterers.keys()),len(year_dates.keys()))

def add_cbar_right_of_axis(ax):
    cr = Colorizer(cmap="bwr")
    cr.set_clim(vmin=0.0,vmax=1.0)

    fig.colorbar(ColorizingArtist(cr),ax=ax,label="Democratic <- (member ratio) -> Repuplican")

apply_func_to_nested_grouping(
    clusterer_idx,year_indx,
    lambda name, i_c, year, i_y: clusterings_pc_py[name][year].draw_nx_graph(
        axs[i_c,i_y]
    ))
add_cbar_right_of_axis(axs[-1,-1])

plt.show()