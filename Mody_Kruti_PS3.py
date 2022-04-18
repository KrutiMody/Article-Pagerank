#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
import math
import time


# In[21]:


# Setting global variables
alpha = 0.85
epsilon = 0.00001
number_of_nodes = 10748
link_file = 'links.txt'


# In[22]:


# import and process data from file,
# make and return H_matrix and an array of all journals
def create_matrix():
    # initial matrix with 10747 nodes of 0 value
    matrix = np.zeros((number_of_nodes, number_of_nodes))
    journal_keys = []
    with open(link_file) as f:
        for _, line in enumerate(f):
            # each line in .txt file is split on the basis of ',' and stored into matrix
            line = line.split(',')
            # first entry states which article is citing
            citing = int(line[0])
            # second entry states the article number it is citing
            cited = int(line[1])
            # third entry states number of times it is cited
            val = int(line[2])
            # with columns being the cited and rows being citing
            matrix[cited][citing] = val
            # add journal to journal_keys array
            journal_keys.append(cited)
    return matrix, journal_keys


# In[23]:


# return normalized matrix (dividing numbers by sum of columns) and an array indicating dangling nodes
# dangling nodes are those nodes that do not cite any article
def normalize_matrix_and_get_dangling(matrix):
    is_dangling_node = []
    # transpose matrix to normalize by column
    # we use enumerate because it keeps a track of index (idx) as well as the element
    for idx, row in enumerate(matrix.T):
        # get column sum to divide
        sum = np.sum(row)
        if sum != 0:
            # if column sum is not 0, normalize column by dividing number by row sum, and mark that entry in dangling node flag vector as 0
            matrix.T[idx] = row / sum
            is_dangling_node.append(0)
        else:  # if column sum is 0, it means that this column is a dangling node, and mark that entry in dangling node flag vector as 1
            is_dangling_node.append(1)
    return matrix, is_dangling_node


# In[24]:


# calculate pi ^ (k + 1) = alpha * H * pi ^ (k) + [alpha * d * pi ^ (k) + ( 1 - alpha )] * artical_vec
def calculate_pi(pi_vec, article_vec, dangling_nodes, H):
    # compute alpha * H * pi ^ (k)
    vec = np.multiply(alpha, H)
    vec = np.dot(vec, pi_vec)
    # compute alpha * d * pi ^ (k) + ( 1 - alpha )
    alpha_d = np.multiply(alpha, dangling_nodes)
    dot_pi = np.dot(alpha_d, pi_vec)
    dot_pi = dot_pi + (1 - alpha)
    # multiply by artical_vec
    dot_a = np.multiply(dot_pi, article_vec)
    # get the new pi-vector by adding two parts together
    pi = vec + dot_a
    return pi


# In[25]:


# check if it's converged: T = pi ^ (k+1) - pi ^ (k) < epsilon
def converged(residual):
    # check every node, find residual >= epsilon
    check = list(filter(lambda x: x >= epsilon, residual))
    # if there is any node where residual >= epsilon, it means it's not converged yet
    # so it will be empty and return true
    return len(check) == 0


# In[26]:


# iterate equation, return the final pi_vecter and number of iterations
def iterate(pi_vec, article_vec, dangling_nodes, H):
    # initial residual
    residual = [1] * number_of_nodes
    count = 0
    while not converged(residual):
        # calculate new pi_vector
        pi_new = calculate_pi(pi_vec, article_vec, dangling_nodes, H)
        # calculate new residual
        residual = pi_new - pi_vec
        # calculated pi_vecter becomes current for the next iteration
        pi_vec = pi_new
        # increment iteration counter
        count += 1
    return pi_vec, count


# In[29]:


def main():
    
    start = time.time()
    print("Initializing execution")


    # 1) Data Input
    # 2) Creating an Adjacency Matrix
    Z_matrix, journal_keys = create_matrix()
    
    # 3) Modifying the Adjacency Matrix
    # setting zero diagnals so it doesn't count self-citing and then pass it to normalizing 
    # and dangling node calculation function
    np.fill_diagonal(Z_matrix, 0)
    
    # 4) Identifying the Dangling Nodes
    # get normalized matrix H and dangling node indicator array d
    H_matrix, dangling_nodes = normalize_matrix_and_get_dangling(Z_matrix)
    
    # 5) Calculating the Stationary Vector
    # Total number of articles, in this case we can assume that be all 1s
    A_total = np.array([1] * number_of_nodes)
    # get article vector
    article_vec = A_total / number_of_nodes
    # initial pi vector: every node is 1 / number_of_nodes
    pi_vec = np.array([1 / number_of_nodes] * number_of_nodes)
    # reshape pi_vector and article_vector
    pi_vec = np.reshape(pi_vec, (number_of_nodes, 1))
    article_vec = np.reshape(article_vec, (number_of_nodes, 1))
    # start iterating, get final pi_vector and iteration counts
    pi_vec, count = iterate(pi_vec, article_vec, dangling_nodes, H_matrix)

    # 6) Calculationg the EigenFactor (EF) Score
    # calculating eigenfactor
    eigenfactor = (100 * (np.dot(H_matrix, pi_vec) / np.sum(np.dot(H_matrix, pi_vec))))
  

    # create dataframe with Journal numbers and Eigenfactor associated with it
    df = pd.DataFrame({'Journal': journal_keys[:len(eigenfactor)], 'Eigenfactor': eigenfactor.T[0]})
    # sort df by Eigenfactor
    df = df.sort_values(by=['Eigenfactor'], ascending=False)
    # rename index
    df.index.name = 'Journal'

    # report result
    print("a) Eigen factor scores for the Top 20 journals:" )
    print(df['Eigenfactor'].head(20))
    end = time.time()
    print("b) Total time taken to run the code: {0} Secs".format(end - start))
    print("c) Number of iterations that required convergence and get real answer: {0}".format(count))


if __name__ == "__main__":
    main()


# In[28]:


# a) Eigen factor scores for the Top 20 journals:

# 1)  4408    1.446274
# 2)  4801    1.410543
# 3)  6610    1.233689
# 4)  2056    0.678919
# 5)  6919    0.664298
# 6)  6667    0.633399
# 7)  4024    0.576042
# 8)  6523    0.480174
# 9)  8930    0.477196
# 10) 6857    0.439395
# 11) 5966    0.429420
# 12) 1995    0.385487
# 13) 1935    0.384908
# 14) 3480    0.379425
# 15) 4598    0.372269
# 16) 2880    0.329962
# 17) 3314    0.326856
# 18) 6569    0.319103
# 19) 5035    0.316170
# 20) 1212    0.311132

# b) Total time taken to run the code: 17.713326930999756 Secs
# c) Number of iterations that required convergence and get real answer: 17

