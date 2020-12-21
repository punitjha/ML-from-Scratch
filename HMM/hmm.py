# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:27:53 2020

@author: Punit
"""


import numpy as np


def log_joint(initial_probs, transition_probs, emission_probs, zs, ys):
  """
  Calculate the log join likelihood of observing a set of states {z_1, ..., z_T} and emissions
  Args:
    initial_probs: a numpy array with shape (S,).  initial_probs[i] is the initial probability of state Z_i.
    transition_probs: a numpy array with shape (S, S).  transition_probs[i,j] is the probability of transitioning from Z_j to Z_i.
    emission_probs: a numpy array with shape (N, S).  emission_probs[i,j] is the probability of emission Y_i given state Z_j.
    zs: a numpy int array with shape (T,).  zs[t] is the state (integer in [0,S)) at time t.
    ys: a numpy int array with shape (T,).  ys[t] is the state (integer in [0,N)) at time t.
  Returns:
    The log likelihood P(zs, ys)
  """
  pass


def max_single_step(previous_log_probs, transition_probs, emission_prob, y):
  """
  Args:
    previous_log_probs: a numpy array with shape (S,).  previous_log_probs[i] is the initial probability of state Z_i at time t-1.
    transition_probs: a numpy array with shape (S, S).  transition_probs[i,j] is the probability of transitioning from Z_j to Z_i.
    emission_prob: a numpy array with shape (N, S).  emission_probs[i,j] is the probability of emission Y_i given state Z_j.
    y: an integer in [0,N), denoting the emission observed
  Returns:
    A tuple (pz, path), where
      pz is a numpy array of shape (S,), where pz[i] stores the log joint probability of the most likely path so far that led to hidden state Z_i and emission y at time t.
      path is a numpy int array of shape (S,), where path[i] is the most likely state that preceded observing Z_i at time t.
  """
  pass


def viterbi(initial_probs, transition_probs, emission_probs, ys):
    """
    Args:
      initial_probs: a numpy array with shape (S,).  initial_probs[i] is the initial probability of state Z_i.
      transition_probs: a numpy array with shape (S, S).  transition_probs[i,j] is the probability of transitioning from Z_j to Z_i.
      emission_probs: a numpy array with shape (N, S).  emission_probs[i,j] is the probability of emission Y_i given state Z_j.
      ys: a numpy int array with shape (T,).  ys[t] is the state (integer in [0,N)) at time t.
    Returns:
      path is a numpy int array of shape (T), where path[t] is the most likely sequence of states
    """
  pass


import numpy as np

def log_joint(initial_probs, transition_probs, emission_probs, zs, ys):
    """
    Calculate the log join likelihood of observing a set of states {z_1, ..., z_T} and emissions
    Args:
      initial_probs: a numpy array with shape (S,).  initial_probs[i] is the initial probability of state Z_i.
      transition_probs: a numpy array with shape (S, S).  transition_probs[i,j] is the probability of transitioning from Z_j to Z_i.
      emission_probs: a numpy array with shape (N, S).  emission_probs[i,j] is the probability of emission Y_i given state Z_j.
      zs: a numpy int array with shape (T,).  zs[t] is the state (integer in [0,S)) at time t.
      ys: a numpy int array with shape (T,).  zs[t] is the state (integer in [0,N)) at time t.

      The log likelihood P(zs, ys)
    """
    prob = 0.0
    for t in range(zs.size):
        if t == 0:
            prob += np.log(initial_probs[zs[t]])
            prob += np.log(emission_probs[ys[t], zs[t]])
        else:
            prob += np.log(transition_probs[zs[t], zs[t-1]])
            prob += np.log(emission_probs[ys[t], zs[t]])
    return prob



def max_single_step(previous_log_probs, transition_probs, emission_prob, y):
    """
    Args:
      previous_log_probs: a numpy array with shape (S,).  previous_log_probs[i] is the initial probability of state Z_i at time t-1.
      transition_probs: a numpy array with shape (S, S).  transition_probs[i,j] is the probability of transitioning from Z_j to Z_i.
      emission_prob: a numpy array with shape (N, S).  emission_probs[i,j] is the probability of emission Y_i given state Z_j.
      y: an integer in [0,N), denoting the emission observed
    Returns:
      A tuple (pz, path), where
          pz is a numpy array of shape (S,), where pz[i] stores the log joint probability of the most likely path so far that led to hidden state Z_i and emission y at time t.
          path is a numpy int array of shape (S,), where path[i] is the most likely state that preceded observing Z_i at time t.
    """
    pz = np.zeros(previous_log_probs.size)
    path = np.zeros([pz.size], dtype=np.int32)
    for k in range(pz.size):
        res = previous_log_probs + np.log(transition_probs[k, :]) + np.log(emission_prob[y, k])
        path[k] = np.argmax(res)
        pz[k] = np.max(res)
    return pz, path


def viterbi(initial_probs, transition_probs, emission_probs, ys):
    """
    Args:
      initial_probs: a numpy array with shape (S,).  initial_probs[i] is the initial probability of state Z_i.
      transition_probs: a numpy array with shape (S, S).  transition_probs[i,j] is the probability of transitioning from Z_j to Z_i.
      emission_probs: a numpy array with shape (N, S).  emission_probs[i,j] is the probability of emission Y_i given state Z_j.
      ys: a numpy int array with shape (T,).  ys[t] is the state (integer in [0,N)) at time t.
    Returns:
      path is a numpy int array of shape (T), where path[t] is the most likely sequence of states
    """
    pz = np.zeros(initial_probs.size)
    path = np.zeros([pz.size, ys.size-1])
    for t in range(ys.size):
        if t == 0:
            pz = np.log(initial_probs) + np.log(emission_probs[ys[0], :])
        else:
            pz, path[:, t-1] = max_single_step(pz, transition_probs, emission_probs, ys[t])
    zs = np.zeros(ys.size, dtype=np.int64)
    print(pz, path)
    zs[zs.size-1] = np.argmax(pz)
    for i in range(zs.size-1, 1, -1):
        zs[i-1] = path[zs[i], i-1]
    return zs

# def log_joint(initial_probs, transition_probs, emission_probs, zs, ys):
#     """
#   Calculate the log join likelihood of observing a set of states {z_1, ..., z_T} and emissions
#   Args:
#     initial_probs: a numpy array with shape (S,).  initial_probs[i] is the initial probability of state Z_i.
#     transition_probs: a numpy array with shape (S, S).  transition_probs[i,j] is the probability of transitioning from Z_j to Z_i.
#     emission_probs: a numpy array with shape (N, S).  emission_probs[i,j] is the probability of emission Y_i given state Z_j.
#     zs: a numpy int array with shape (T,).  zs[t] is the state (integer in [0,S)) at time t. hidden
#     ys: a numpy int array with shape (T,).  ys[t] is the state (integer in [0,N)) at time t. observed
#   Returns:
#     The log likelihood P(zs, ys)
#   """
#     #print(initial_probs)
#     #print(transition_probs)
#     #print(emission_probs)
#     #print(zs)
#     #print(ys)
#     log_like=0.0
#     log_like+=np.log(initial_probs[zs[0]])
#     log_like+=np.log(emission_probs[ys[0],zs[0]])
#     for xx in range(1,len(ys)): # states observed--- N
#         #for yy in range(1,len(zs)): # states hidden --- S ---
#         log_like+=np.log(transition_probs[zs[xx],zs[xx-1]])
#         log_like+=np.log(emission_probs[ys[xx],zs[xx]])
#     return log_like
    
  


# def max_single_step(previous_log_probs, transition_probs, emission_prob, y):
#     """
#   Args:
#     previous_log_probs: a numpy array with shape (S,).  previous_log_probs[i] is the initial probability of state Z_i at time t-1.
#     transition_probs: a numpy array with shape (S, S).  transition_probs[i,j] is the probability of transitioning from Z_j to Z_i.
#     emission_prob: a numpy array with shape (N, S).  emission_probs[i,j] is the probability of emission Y_i given state Z_j.
#     y: an integer in [0,N), denoting the emission observed
#   Returns:
#     A tuple (pz, path), where
#       pz is a numpy array of shape (S,), where pz[i] stores the log joint probability of the most likely path so far that led to hidden state Z_i and emission y at time t.
#       path is a numpy int array of shape (S,), where path[i] is the most likely state that preceded observing Z_i at time t.
#   """
#     pz=np.zeros((len(previous_log_probs)))
#     path=np.zeros((len(previous_log_probs)))
#     for s in range(len(previous_log_probs)):
#         ll,p=max((previous_log_probs[k]+np.log(transition_probs[s,k])+np.log(emission_prob[y,s]),k) for k in range(len(previous_log_probs)))
#         #pz[s]=previous_log_probs[ll]+np.log(transition_probs[ll,s])+np.log(emission_prob[y,s])
#         pz[s]=ll
#         path[s]=p
#     #print("path",path)
#     #print("log likly hood",pz)
#     return (pz, path)

# def viterbi(initial_probs, transition_probs, emission_probs, ys):
#     """
#     Args:
#       initial_probs: a numpy array with shape (S,).  initial_probs[i] is the initial probability of state Z_i.
#       transition_probs: a numpy array with shape (S, S).  transition_probs[i,j] is the probability of transitioning from Z_j to Z_i.
#       emission_probs: a numpy array with shape (N, S).  emission_probs[i,j] is the probability of emission Y_i given state Z_j.
#       ys: a numpy int array with shape (T,).  ys[t] is the state (integer in [0,N)) at time t.
#     Returns:
#       path is a numpy int array of shape (T), where path[t] is the most likely sequence of states
#     """
#     #print(ys)
#     #my_ys=set(ys)
#     #print("this is my_ys",my_ys)
#     #trellis = np.zeros((len(initial_probs), len(O)))
#     test=[]
#     all_paths=np.zeros((len(initial_probs),len(ys)))
#     all_probs=np.zeros((len(initial_probs),len(ys)))
#     test.append(0)
#     for x in range(1,len(ys)):
#         pz, path=max_single_step(initial_probs, transition_probs, emission_probs, ys[x])
#         initial_probs=pz
#         all_paths[:,x]=path
#         all_probs[:,x]=pz
#         #print(path)
#         #print(pz)
#         #print(path[np.argmax(pz)])
#         test.append(path[np.argmax(pz)])
#     print(all_paths)
#     print(all_probs)
#     best_path=[]
#     for x in range(-1, -(len(ys)+1), -1):
#         best_path.append(np.argmax(all_probs[:,x]))
#     max_path=(np.asarray(best_path))
#     test=np.asarray(test)
#     print("test",test)
#     return test























