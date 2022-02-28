'''
# Dynamic Time Warping
# 논문 : Dynamic Time Warping Review
# time-series similarity measure which minimizes the effects of shifting and distortion in time by allowing "elastic" transformation of time series
# in order to detect similar shapes with different phases.
# O(MN)

# 자신과 같은 시간 index를 가진 요소와 비교 뿐만 아니라, 그 주변의 다른 요소와도 비교를 해서 더 비슷한 요소를 자신의 비교쌍으로 놓는다면 어느정도 극복되지 않을까?
#
'''



from math import *
import numpy as np
import csv
import sys
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt



def read_csv(wave_idx):
    csv_file_path = "./data/ITRoom_labeled_data_for_tsfresh.csv"
    f = open(csv_file_path,'r')
    rdr = csv.reader(f)
    wave = {}
    value = []
    for line in rdr:
        if line[0] == str(wave_idx):
            value.append(float(line[2]))
    wave[wave_idx] = value
    f.close()
    return value



def DTW(A, B, d=lambda x, y: abs(x - y)):
    # 비용 행렬 초기화
    A, B = np.array(A), np.array(B)
    M, N = len(A), len(B)
    cost = sys.maxsize * np.ones((M, N))
    # in python3 sys.maxint changed to max.size
    # 첫번째 로우,컬럼 채우기
    cost[0, 0] = d(A[0], B[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(A[i], B[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(A[0], B[j])
    # 나머지 행렬 채우기
    for i in range(1, M):
        for j in range(1, N):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(A[i], B[j])

    # 최적 경로 구하기
    n, m = N - 1, M - 1
    path = []

    while (m, n) != (0, 0):
        path.append((m, n))
        m, n = min((m - 1, n), (m, n - 1), (m - 1, n - 1), key=lambda x: cost[x[0], x[1]])

    path.append((0, 0))
    return cost[-1, -1], path


def main():

    A = read_csv(1)
    B = read_csv(3)


    #cost, path = DTW(A, B)
    #print(path)
    #print('Total Distance is ', cost)

    '''
    using tslearn library : dtw(x,y)= min sqrt(sigma(D(x,y)^2))
    '''
    optimal_path,dtw_score = dtw_path(A,B)
    print(len(optimal_path))
    print(optimal_path)
    print(dtw_score)
    plt.figure()
    plt.title("Dynamic Time Warping")
    plt.plot(A,color="blue",label='Wave1')
    plt.plot(B,color="orange",label='Wave3')
    plt.xlim(0,300)
    for i in range(len(optimal_path)):
        plt.plot( [optimal_path[i][0], optimal_path[i][1]], [A[optimal_path[i][0]], B[optimal_path[i][1]]], color="grey",linestyle='--',linewidth="0.5")
    plt.legend()
    plt.show()

    interpol_B = []
    for i in range(len(A)):
        for j in range(len(optimal_path)):
            if i == optimal_path[j][0]:
                interpol_B.append(B[optimal_path[j][1]])
                break

    plt.figure()
    plt.title("After Resample")
    plt.plot(A,color="blue",label='Wave1')
    plt.plot(interpol_B,color="orange",label='RESAMPLE Wave3')
    plt.xlim(0,300)
    plt.show()
if __name__ == '__main__':
    main()

