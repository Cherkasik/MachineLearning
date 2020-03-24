k = int(input())
n = int(input())
values = [list(map(int, input().split())) for _ in range(n)]
M = [0 for _ in range(k)]
M2 = [0 for _ in range(k)]
count = [0 for _ in range(k)]
for i in range(n):
    x = values[i][0]
    y = values[i][1]
    M[x - 1] += y
    M2[x - 1] += (y ** 2)
    count[x - 1] += 1
D = [0 for _ in range(k)]
for i in range(k):
    if count[i] == 0:
        D[i] = 0
    else:
        D[i] = (M2[i] / count[i] - (M[i] / count[i]) ** 2) * count[i] / n
print(sum(D))