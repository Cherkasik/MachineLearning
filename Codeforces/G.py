import math


m = int(input())
n = 2 ** m
net = []
for i in range(n):
    value = int(input())
    if value == 1:
        condition = []
        b = 0.5
        for x in range(m):
            value = 1 if (i & (1 << x)) != 0 else -1
            condition.append(value)
            if value == 1:
                b -= 1
        condition.append(b)
        net.append(condition)
if len(net) > 0:
    condition = []
    for _ in range(len(net)):
        condition.append(1)
    condition.append(-0.5)
    net.append(condition)
    print(2)
    print(str(len(net) - 1) + " 1")
else:
    print(1)
    print(1)
    condition = []
    for _ in range(m):
        condition.append(1)
    condition.append(-m - 0.5)
    net.append(condition)
for elem in net:
    print(' '.join(str(x) for x in elem))