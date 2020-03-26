import math

kx, ky = map(int, input().split())
n = int(input())
p = [{} for _ in range(kx)]
sumX = [0] * kx
for i in range(n):
    x, y = map(int, input().split())
    sumX[x - 1] += 1
    p[x - 1][y - 1] = (p[x - 1].get(y - 1) or 0) + 1
h = 0
for i in range(kx):
    curX = sumX[i]
    curP = curX / n
    curH = 0
    for x in p[i].values():
        if x != 0:
            curH -= (x / curX) * math.log(x / curX)
    h += curH * curP
print(h)