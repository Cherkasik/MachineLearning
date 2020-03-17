import math

n = int(input())
X = [list(map(int, input().split())) for _ in range(n)]
x1 = zip([row[0] for row in X], [i for i in range(n)])
x2 = zip([row[1] for row in X], [i for i in range(n)])
x1_sorted = sorted(x1, key=lambda tup: tup[0])
x2_sorted = sorted(x2, key=lambda tup: tup[0])
x1_sorted = zip([row[0] for row in x1_sorted], [row[1] for row in x1_sorted], [i for i in range(n)])
x2_sorted = zip([row[0] for row in x2_sorted], [row[1] for row in x2_sorted], [i for i in range(n)])
x1_sorted = sorted(x1_sorted, key=lambda tup: tup[1])
x2_sorted = sorted(x2_sorted, key=lambda tup: tup[1])
if n != 0 and n != 1:
    diff = 0
    for i in range(n):
        diff += (x1_sorted[i][2] - x2_sorted[i][2]) ** 2
    print(1 - (6 * diff / (n * (n - 1) * (n + 1))))
else:
    print(1)
