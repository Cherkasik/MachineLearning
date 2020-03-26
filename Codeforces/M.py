k1, k2 = map(int, input().split())
n = int(input())

k1Count = [0] * k1
k2Count = [0] * k2
variance = [{} for _ in range(k1)]
for _ in range(n):
    x1, x2 = map(int, input().split())
    k1Count[x1 - 1] += 1
    k2Count[x2 - 1] += 1
    variance[x1 - 1][x2 - 1] = (variance[x1 - 1].get(x2 - 1) or 0) + 1
k1Sum = sum(k1Count)
k2Sum = sum(k2Count)
hi2 = (k1Sum * k2Sum) / n
for i in range(k1):
    for key, value in variance[i].items():
        cur = (k1Count[i] * k2Count[key]) / n
        hi2 += ((value - cur) ** 2) / cur - cur
print(hi2)
