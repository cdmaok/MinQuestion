# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, print_function, division
import time,sys
import numpy as np
from .common import knn_initialize
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .doc2vec import getvector
sys.path.insert(0,'..')
import config
from fill import simrank
Filemodel = config.Filemodel
#Text_path = config.Text_path
origin_file = config.origin_file
two_party_flag = config.two_party_flag
Text_path = config.Text_path_tp if two_party_flag == True else config.Text_path

def doc2vec_method(df):
    df = df.replace(np.nan,'')
    text = df.iloc[:,1]
    text = [q.strip().decode('utf-8','ignore')  for q in list(text) if q==q]

    # filemodel ='./wiki_doc2vec.bin'
    # doc2vec.train(text,filemodel)
    # m = doc2vec.getMatrix('doc2vec.matrix',filemodel)
    # vectors = [list(doc2vec.getvector(q,filemodel)) for q in text]
    vectors = [list(getvector(q,Filemodel)) for q in text]
    m = np.asarray(vectors)
    sim = cosine_similarity(m,m)
    return sim

def knn_impute_few_observed(X, simf,missing_mask, k, verbose=False, print_interval=100):

    start_t = time.time()
    sim = None
    if simf == 'simrank':
        sim = simrank.simrank_from_file(origin_file)
    elif simf == 'text':
        df = pd.read_csv(Text_path)
        # df = pd.read_csv('./text_twoparty.csv')
        sim = doc2vec_method(df)
	

    n_rows, n_cols = X.shape
    print ("matrix size %d / %d, similarity size %d / %d"%(n_rows,n_cols,sim.shape[0],sim.shape[1]))
    # put the missing mask in column major order since it's accessed
    # one column at a time
    missing_mask_column_major = np.asarray(missing_mask, order="F")
    observed_mask_column_major = ~missing_mask_column_major
    X_column_major = X.copy(order="F")
    X_row_major, D = knn_initialize(X, missing_mask, verbose=verbose)

    # get rid of infinities, replace them with a very large number
    # finite_distance_distance_mask = np.isfinite(D)
    # effective_infinity = 10 ** 6 * D[finite_distance_distance_mask].max()
    # D[~finite_distance_distance_mask] = effective_infinity
    #
    # D_sorted = np.argsort(D, axis=1)
    # inv_D = 1.0 / D
    # D_valid_mask = D < effective_infinity
    # valid_distances_per_row = D_valid_mask.sum(axis=1)
    # # trim the number of other rows we consider to exclude those
    # # with infinite distances
    # D_sorted = [
    #     D_sorted[i, :count]
    #     for i, count in enumerate(valid_distances_per_row)
    # ]
    # print(D_sorted)

    D_sorted = np.argsort(sim, axis=1)
    # print(D_sorted)   
    inv_D = sim
    
    # print(inv_D)


    dot = np.dot
    for i in range(n_rows):
        #if (sim[i][D_sorted[i][1]] < 0.6) or (sim[i][D_sorted[i][1]] == 1.0) :
        #    continue
        missing_row = missing_mask[i, :]
        missing_indices = np.where(missing_row)[0]
        row_weights = inv_D[i, :]
        # row_sorted_indices = D_sorted_indices[i]
        if verbose and i % print_interval == 0:
            print(
                "Imputing row %d/%d with %d missing columns, elapsed time: %0.3f" % (
                    i + 1,
                    n_rows,
                    len(missing_indices),
                    time.time() - start_t))
        # row_neighbor_indices = neighbor_indices[i]
        candidate_neighbor_indices = D_sorted[i]

        for j in missing_indices:
            observed = observed_mask_column_major[:, j]
            sorted_observed = observed[candidate_neighbor_indices]
            observed_neighbor_indices = candidate_neighbor_indices[sorted_observed]
            k_nearest_indices = observed_neighbor_indices[:k]
            weights = row_weights[k_nearest_indices]
            weight_sum = weights.sum()
            if weight_sum > 0:
                column = X_column_major[:, j]
                values = column[k_nearest_indices]
                X_row_major[i, j] = dot(values, weights) / weight_sum
    return X_row_major
