import math

kx, ky = map(int, input().split())
n = int(input())
p = { i: { j: 0 for j in range(ky) } for i in range(kx) }
sumX = [0 for i in range(kx)]
for i in range(n):
    x, y = map(int, input().split())
    p[x - 1][y - 1] += 1
    sumX[x - 1] += 1
h = 0
for i in range(kx):
    if sumX[i] != 0:
        for j in range(ky):
            if p[i][j] != 0:
                h -= (p[i][j] / n) * math.log(p[i][j] / sumX[i])
print(h)