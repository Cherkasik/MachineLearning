import math

n = int(input())
X = [list(map(int, input().split())) for _ in range(n)]
if n != 0:
    x1_avg = sum(row[0] for row in X) / n
    x2_avg = sum(row[1] for row in X) / n
    diff = 0
    x1_diff2 = 0
    x2_diff2 = 0
    for i in range(n):
        x1_diff = X[i][0] - x1_avg
        x2_diff = X[i][1] - x2_avg
        diff += x1_diff * x2_diff
        x1_diff2 += x1_diff * x1_diff
        x2_diff2 += x2_diff * x2_diff
    if (x1_diff2 * x2_diff2 != 0):
        print(diff / math.sqrt(x1_diff2 * x2_diff2))
    else:
        print(0)
else:
    print(0)
