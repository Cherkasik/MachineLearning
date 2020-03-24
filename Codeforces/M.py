k1, k2 = map(int, input().split())
n = int(input())
variance = { i: { j: 0 for j in range(k2) } for i in range(k1) }
sumRow = [0 for _ in range(k1)]
sumColumn = [0 for _ in range(k2)]
for _ in range(n):
    x1, x2 = map(int, input().split())
    variance[x1 - 1][x2 - 1] += 1
    sumRow[x1 -1] += 1
    sumColumn[x2 - 1] += 1
hi2 = 0
for i in range(k1):
    for j in range(k2):
        if sumRow[i] != 0 and sumColumn[j] != 0:
            theoryVariance = sumRow[i] * sumColumn[j] / n
            hi2 += ((variance[i][j] - theoryVariance) ** 2) / theoryVariance
print(hi2)
