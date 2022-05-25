
import math
from black import main
from cv2 import pencilSketch
import numpy as np
import random

negative_infinite = -1000

n_label = 4 # 4种状态，0: B, 1: M, 2: E, 3: S
n_char = 65536 # 65536种观测值，65536个字

def norm(arr):
    '''归一化概率数组'''
    s = sum(arr)
    for i in range(len(arr)):
        arr[i] /= s

def log_norm(arr):
    '''
    对数归一化概率数组
    '''
    s = sum(arr) # sum of arr
    sum_log = math.log(s) # log of sum
    for i in range(len(arr)):
        if arr[i] == 0:
            arr[i] = negative_infinite
        else: 
            arr[i] = math.log(arr[i]) - sum_log


def MLE(train_data_path): # 0: B, 1: M, 2: E, 3: S
    '''
    最大似然估计训练 HMM 模型，返回模型参数pi、A、B
    '''
    # initial parameters
    pi = [0] * n_label # shape: (4)
    A = [[0] * n_label for _ in range(n_label)] # shape: (4x4)
    B = [[0] * n_char for _ in range(n_label)] # shape: (4x65536)
    
    # get training data
    f = open(train_data_path, encoding='utf8')
    data = f.read()
    tokens = data.split('  ')
    f.close()
    
    # train
    last_l = 2 # 上一个token的结尾(2: E)
    for token in tokens:
        token = token.strip()
        token_len = len(token)
        if token_len == 0:
            continue
        if token_len == 1: # 3: S
            pi[3] += 1
            A[last_l][3] += 1 # 2: E -> 3: S
            B[3][ord(token)] += 1 # 3: S -> token
            last_l = 3
            continue
        # 更新初始状态概率 0: B, 1: M, 2: E
        pi[0] += 1
        # pi[1] += (token_len - 2)
        # pi[2] += 1
        # 更新状态转移概率矩阵
        A[last_l][0] += 1
        last_l = 2
        if token_len == 2:
            A[0][2] += 1 # 0: B -> 2: E
        else: # token_len > 3
            A[0][1] += 1 # 0: B -> 1: M
            A[1][1] += (token_len - 3) # 1: M -> 1: M
            A[1][2] += 1 # 1: M -> 2: E
        # 更新发射概率矩阵
        B[0][ord(token[0])] += 1 # 0: B -> token[0]
        for i in range(1, token_len-1): # 1: M -> token[1:token_len-1]
            B[1][ord(token[i])] += 1
        B[2][ord(token[token_len-1])] += 1 # 2: E -> token[-1]
    
    # 归一化
    log_norm(pi)
    #norm(pi)
    for i in range(n_label):
        log_norm(A[i])
        log_norm(B[i])
        #norm(A[i])
        #norm(B[i])
    return pi, A, B

def list_write(f, arr):
    for x in arr:
        f.write(str(x))
        f.write(' ')
    f.write('\n')


# 保存训练阈值参数
def save_param(pi, A, B):
    f_pi = open('threshold_pi.txt', mode='w')
    list_write(f_pi, pi)
    f_pi.close()
    f_A = open('threshold_A.txt', mode='w')
    for r in A:
        list_write(f_A, r)
    f_A.close()
    f_B = open('threshold_B.txt', mode='w')
    for r in B:
        list_write(f_B, r)
    f_B.close()



def run_main(train_data_path):
    pi, A, B = MLE(train_data_path)
    save_param(pi, A, B)
    return pi, A, B



if __name__ == '__main__':
    train_data_path="pku_training.utf8"
    # 打印函数执行耗时  在ipython或jupyter测试即可
    # %time pi, A, B = MLE()
    # %time save_param(pi, A, B)
    
    #训练获取参数
    run_main(train_data_path)

    
    
    
