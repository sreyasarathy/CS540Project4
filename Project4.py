## Written by: Sreya Sarathy
## Attribution: Hugh Liu's solutions for CS540 2021 Epic
## Collaborated with Harshet Anand from CS540

import pandas as pd
import numpy as np
import ast

# The parameters I had were 5, target number of clusters being 6.
# I chose the state of California along with the default Wisconsin.
num_of_parameters = 5
target_num_clusters = 6
target_states = ['Wisconsin', 'California']

# The data is stored in the time_series_covid19_deaths_US.csv file
df = pd.read_csv('time_series_covid19_deaths_US.csv')
all_states = list(set(df.Province_State))


# All states are sorted alphabetically here
all_states.sort()
to_remove_states = ['Grand Princess',
                    'Diamond Princess',
                    'Guam',
                    'American Samoa',
                    'Virgin Islands',
                    'Northern Mariana Islands',
                    'Puerto Rico',
                    'District of Columbia',
                    ]
all_states = [x for x in all_states if x not in to_remove_states]
num_of_states = len(all_states)

# This function returns the cumulative timeseries death data for target states or all states.
# Input: df, target_states
def get_cumulative_timeseries(df, target_states):
    '''This function returns the cumulative timeseries death data for target states or all states
    Input:
        df: df
        target_states: e.g., ['Wisconsin', 'California'], or all states
    '''

    cumulative_timeseries_data_list = []
    first_date_col = df.columns.get_loc("1/22/20")
    for state in target_states:
        state_df = df[df.Province_State == state]

        # death in a million
        state_population = state_df['Population'].sum()/10**6
        if state_population == 0:
            state_population = 1
        state_timeseries = state_df.iloc[:, first_date_col:]

        if target_states == all_states:
            state_cumulative_timeseries = (state_timeseries.sum(axis = 0)/state_population).tolist()
        else:
            state_cumulative_timeseries = (state_timeseries.sum(axis = 0)).tolist()
        cumulative_timeseries_data_list.append(state_cumulative_timeseries)
    return cumulative_timeseries_data_list

# This function returns the timeseries difference data
# The input is a list of cumulative time series
# Return a list of numpy arrays.
def get_time_diff(cumulative_timeseries_data_list):
    '''This function returns the timeseries differnece data
    Input:
        a list of cumulative timeseries data
    Return:
        a list of numpy arrays
    '''
    time_diff_list = []
    for state_cum_ts in cumulative_timeseries_data_list:
        state_time_diff = []
        for i in range(len(state_cum_ts)-1):
            state_time_diff.append(state_cum_ts[i+1] - state_cum_ts[i])
        time_diff_list.append(np.array(state_time_diff))
    return time_diff_list


cumulative_timeseries_data_list = get_cumulative_timeseries(df, target_states)
# wisconsin
wi_cum_ts = cumulative_timeseries_data_list[0]
# california
ca_cum_ts = cumulative_timeseries_data_list[1]

time_diff_list = get_time_diff(cumulative_timeseries_data_list)
# wisconsin
wi_time_diff = time_diff_list[0]
# california
ca_time_diff = time_diff_list[1]

# The following lines of code calculates the cumulative time series data for COVID-19 deaths in all states
# using the get_cumulative_timeseries(df, all_states) function.
# It calculates the time differences between consecutive data points for each state using the
# get_time_diff(all_cum_ts) function, where all_cum_ts contains the cumulative time series data for all states.

def get_beta(state_time_diff):
    above_sum = 0
    below_sum = 0
    for t in range(1, len(state_time_diff)+1):
        above_sum += (state_time_diff[t-1] - mean) * (t - (len(state_time_diff)+1)/2)
        below_sum += np.square(t - (len(state_time_diff)+1)/2)
    beta = above_sum/below_sum
    return beta

def get_pho(state_time_diff):
    above_sum = 0
    below_sum = 0
    for t in range(2, len(state_time_diff)+1):
        above_sum += (state_time_diff[t-1] - mean) * (state_time_diff[t-2] - mean)
    for t in range(1, len(state_time_diff)+1):
        below_sum += np.square(state_time_diff[t-1]-mean)
    if below_sum != 0:
        pho = above_sum/below_sum
    else:
        pho = 1
    return pho

all_cum_ts = get_cumulative_timeseries(df, all_states)
all_time_diff = get_time_diff(all_cum_ts)

# It then computes the following statistics for each state:
# Mean of the time differences.
# Standard deviation of the time differences.
# Median of the time differences.
# Beta value using the get_beta(state_time_diff) function. The beta value is computed based on the time differences.
# Pho value using the get_pho(state_time_diff) function. The pho value is also computed based on the time differences.

means = np.zeros(num_of_states)
stds = np.zeros(num_of_states)
medians = np.zeros(num_of_states)
betas = np.zeros(num_of_states)
phos = np.zeros(num_of_states)

for idx, state_time_diff in enumerate(all_time_diff):
    mean = np.mean(state_time_diff)
    std = np.std(state_time_diff)
    median = np.median(state_time_diff)
    beta = get_beta(state_time_diff)
    pho = get_pho(state_time_diff)
    means[idx] = mean
    stds[idx] = std
    medians[idx] = median
    betas[idx] = beta
    phos[idx] = pho

# https://www.stackvidhya.com/how-to-normalize-data-between-0-and-1-range/
def rescale(array):
    diff = array - np.min(array)
    max_diff = np.max(array) - np.min(array)
    new_array = diff/max_diff
    return new_array

means = rescale(means)
stds = rescale(stds)
medians = rescale(medians)
betas = rescale(betas)
phos = rescale(phos)

params = [means, stds, medians, betas, phos]
param_matrix = np.stack(params, axis=1)

#### HIERARCHICAL CLUSTERING
M = param_matrix

# eu_distance(x, y): This function calculates the Euclidean distance between two vectors x and y using the formula
def eu_distance(x,y):
    p=np.sum((x-y)**2)
    d=np.sqrt(p)
    return d

dist_matrix = np.zeros((num_of_states, num_of_states))
for i in range(num_of_states):
    for j in range(num_of_states):
        if i >= j:
            dist_matrix[i,j] = 10**10
        else:
            dist_matrix[i,j] = eu_distance(M[i], M[j])

# single_linkage_dist(cluster1, cluster2, dist_matrix): This function calculates the distance between two clusters using
# the single linkage method.
# It takes two clusters cluster1 and cluster2 (represented as lists of indices) and the distance matrix dist_matrix.
# It computes the minimum distance between any pair of points in the two clusters.
def single_linkage_dist(cluster1, cluster2, dist_matrix):
    dist_list = []
    for i in cluster1:
        for j in cluster2:
            if i < j:
                dist_list.append(dist_matrix[i,j])
            else:
                dist_list.append(dist_matrix[j,i])
    return min(dist_list)

# complete_linkage_dist(cluster1, cluster2, dist_matrix): This function calculates the distance between two clusters
# using the complete linkage method. Similar to the single linkage function, it takes two clusters cluster1 and cluster2
# (represented as lists of indices)
# and the distance matrix dist_matrix. It computes the maximum distance between any pair of points in the two clusters
def complete_linkage_dist(cluster1, cluster2, dist_matrix):
    dist_list = []
    for i in cluster1:
        for j in cluster2:
            if i < j:
                dist_list.append(dist_matrix[i,j])
            else:
                dist_list.append(dist_matrix[j,i])
    return max(dist_list)

# The following method should be either "single" or "complete".
def cluster_hierarchy(parameter_matrix, target_num_clusters, dist_matrix, method="single"):
    '''method should be either "single" or "complete"
    '''
    clusters = [[i] for i in range(len(parameter_matrix))]
    while len(clusters) > target_num_clusters:
        dmax = np.max(dist_matrix) + 1
        dmin = dmax
        dist_dic = {}
        # clusters with minimal distances
        min_cluster1 = None
        min_cluster2 = None
        for cluster1 in clusters:
            for cluster2 in clusters:
                if cluster1 != cluster2:
                    if method== "single":
                        dist = single_linkage_dist(cluster1, cluster2, dist_matrix)
                    elif method == "complete":
                        dist = complete_linkage_dist(cluster1, cluster2, dist_matrix)
                    else:
                        print("ERROR! METHOD should be either single or complete")
                    dist_dic[f'[{cluster1},{cluster2}]'] = dist
                    if dist < dmin:
                        dmin = dist
                        min_cluster1 = cluster1
                        min_cluster2 = cluster2
        distances = np.array(list(dist_dic.values()))
        dmin_idxs = np.where(distances == dmin)[0]
        cluster_pairs = list(dist_dic.keys())
        if len(dmin_idxs) > 1:
            # cluster pairs with the same dmin
            cluster_pairs_with_dmin = [ast.literal_eval(cluster_pairs[i]) for i in dmin_idxs]
            flat_list = [item for sublist in cluster_pairs_with_dmin for item in sublist]
            min_idx = min(flat_list)
            for cluster_pair in cluster_pairs_with_dmin:
                if min_idx in cluster_pair:
                    cluster_pair_with_min_idx = cluster_pair
            min_cluster1 = cluster_pair_with_min_idx[0]
            min_cluster2 = cluster_pair_with_min_idx[1]
        clusters.remove(min_cluster1)
        clusters.remove(min_cluster2)
        clusters.append(min_cluster1 + min_cluster2)

    clustering_result = []
    for i in range(len(parameter_matrix)):
        for c in clusters:
            c_index = clusters.index(c)
            if i in c:
                clustering_result.append(c_index)
    return clustering_result

#### SINGLE LINKAGE
single_linkage_clustering = cluster_hierarchy(
    M, target_num_clusters, dist_matrix, "single")
# print(single_linkage_clustering)

### COMPLETE LINKAGE
complete_linkage_clustering = cluster_hierarchy(
    M, target_num_clusters, dist_matrix, "complete")
# print(complete_linkage_clustering)

### K-MEANS CLUSTERING

k=target_num_clusters
n,m = M.shape
np.random.seed(2022)
a = np.arange(n)
np.random.shuffle(a)
centers = M[a[:k]]

def d_centers2nodes(M, centers):
    n,m = M.shape
    c,w = centers.shape
    d = M.reshape([n,1,m]) - centers.reshape([1,c,w])
    d = d**2
    d = np.sum(d, axis=2)
    return d

for i in range(100):
    d = d_centers2nodes(M, centers)
    index = np.argmin(d, axis=1)
    for j in range(k):
        centers[j] = np.mean(M[index==j].reshape([-1,m]), axis=0)
    #print(index)

# print(index)

centers = centers.round(decimals=4)

d = d_centers2nodes(M, centers)
index = np.argmin(d, axis=1)
distortion = 0
for j in range(k):
    distortion += np.sum(d[index==j,j])

# print(distortion)


# The following lines of code are used in Question 1
# The code calculates and writes the cumulative time series
# for Wisconsin and California COVID - 19 deaths to question1.txt
df = pd.read_csv('time_series_covid19_deaths_US.csv')

target_states = ['Wisconsin', 'California']

def get_cumulative_timeseries(df, target_states):
    cumulative_timeseries_data_list = []
    first_date_col = df.columns.get_loc("1/22/20")
    for state in target_states:
        state_df = df[df.Province_State == state]
        state_timeseries = state_df.iloc[:, first_date_col:]
        state_cumulative_timeseries = state_timeseries.sum(axis = 0).tolist()
        cumulative_timeseries_data_list.append(state_cumulative_timeseries)
    return cumulative_timeseries_data_list

cumulative_timeseries_data_list = get_cumulative_timeseries(df, target_states)

with open('question1.txt', 'w') as f:
    f.write("Cumulative time series for Wisconsin:\n")
    f.write(', '.join([str(i) for i in cumulative_timeseries_data_list[0]]) + "\n")
    f.write("Cumulative time series for California:\n")
    f.write(', '.join([str(i) for i in cumulative_timeseries_data_list[1]]) + "\n")


# The following lines of code are used for Question 2
# The code writes Wisconsin daily additional deaths to Question2.txt
# followed by California daily additional deaths, both lists are separated by commas.
with open('Question2.txt', 'w') as f:
    f.write("Wisconsin daily additional deaths:\n")
    f.write(', '.join([str(int(i)) for i in wi_time_diff]) + "\n")
    f.write("California daily additional deaths:\n")
    f.write(', '.join([str(int(i)) for i in ca_time_diff]) + "\n")


# The following lines of code are used for Question 4
# The code writes the elements of param_matrix to Question4.txt
# with each row formatted to four decimal places and separated by commas.
with open('Question4.txt', 'w') as f:
    for i in range(len(param_matrix)):
        f.write(",".join(format(x, ".4f") for x in param_matrix[i]) + "\n")

# The following lines of code are used in Question 5
# The code writes complete results to Question5.txt including
# single linkage clustering elements separated by commas.
with open('Question5.txt', 'w') as f:
    f.write("Complete single hierarchical clustering results:\n")
    single_linkage_clusters_str = ', '.join(map(str, single_linkage_clustering))
    f.write(single_linkage_clusters_str + "\n")

# The following lines of code are used Question 6
# The code writes complete linkage hierarchial clustering results
# to Question 6.txt including the complete linkage clustering elements
# separated by commas.
with open('Question6.txt', 'w') as f:
    f.write("Complete linkage hierarchical clustering results:\n")
    complete_linkage_clustering_str = ', '.join(map(str, complete_linkage_clustering))
    f.write(complete_linkage_clustering_str + "\n")

# The following lines of code are used for Question 7
# The code writes K-means clustering results to Question 7.txt,
# including the "index" elements separated by commas.
with open('Question7.txt', 'w') as f:
    f.write("K-means clustering results:\n")
    kmeans_clusters_str = ', '.join(map(str, index))
    f.write(kmeans_clusters_str + "\n")

# The following lines of code are used for Question 8
# The code writes the centers list to Question 8.txt with
# each element formatted to four decimal places on separate lines.
with open('Question8.txt', 'w') as f:
    for center in centers:
        f.write(", ".join(["{:.4f}".format(value) for value in center]) + "\n")


# The following lines of code are used for Question 9
# It writes the distortion variable for question 9
with open('Question9.txt', 'w') as f:
    f.write(str(distortion) + "\n")