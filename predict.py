
from train import *
# 维特比算法
#参考：https://blog.csdn.net/qq_37150711/article/details/107241829
def viterbi(pi, A, B, O):
    T = len(O)
    delta = [[0] * n_label for _ in range(T)] # delta[t][i]: t 时刻，到达状态 i，并输出观测序列 O[1:t], shape: (Tx4)
    pre = [[0] * n_label for _ in range(T)] # pre[t][i]: t 时刻，状态 i 的前一个状态，shape: (Tx4)
    # 初始化 delta[0]
    for i in range(n_label):
        delta[0][i] = pi[i] + B[i][ord(O[0])] # pi[i] * B[i][ord(O[0])]
    # 递推计算 delta[1:]
    for t in range(1, T):
        for i in range(n_label):
            delta[t][i] = delta[t-1][0] + A[0][i]
            for j in range(1, n_label): # 搜索最大的 delta[t][i]
                tj = delta[t-1][j] + A[j][i]
                if tj > delta[t][i]:
                    delta[t][i] = tj
                    pre[t][i] = j
            delta[t][i] += B[i][ord(O[t])]
    # 解码：回溯查找最优路径
    decode = [-1] * T
    decode[T-1] = np.argmax(delta[T-1])
    last_l = decode[-1]
    for t in range(T-2, -1, -1):
        last_l = pre[t+1][last_l]
        decode[t] = last_l
    return decode



# hmm模型参考：https://blog.csdn.net/qq_24819773/article/details/94008344
def segment(sentence, decode):
    N = len(sentence)
    i = 0
    while i < N:  #B/M/E/S
        if decode[i] == 0 or decode[i] == 1:  # B
            j = i+1
            while j < N:
                if decode[j] == 2:
                    break
                j += 1
            print(sentence[i:j+1], "|", end=' ')
            i = j+1
        elif decode[i] == 3 or decode[i] == 2:    # S
            print(sentence[i:i+1], "|", end=' ')
            i += 1
        else:
            print('Error:', i, decode[i])
            i += 1

def main(test_file):
    f = open("test.txt", encoding='utf-8')
    data = f.read()
    pi, A, B=run_main("pku_training.utf8")
    decode = viterbi(pi, A, B, data)
    segment(data, decode)



if __name__ == '__main__':
    main("test.txt")
