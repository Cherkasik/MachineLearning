import math

n, m = map(int, input().split())
values = [list(map(int, input().split())) for _ in range(n)]
q = list(map(int, input().split()))
functionName = input()
kernelName = input()
windowType = input()
hk = int(input())
result = 0.0

distance = []
dist = 0.0
if (functionName == 'manhattan'):
    for k in range(n):
        dist = 0.0
        for i in range(m):
            dist += math.fabs(values[k][i] - q[i])
        distance.append(dist)
elif (functionName == 'euclidean'):
    for k in range(n):
        dist = 0.0
        for i in range(m):
            dist += (values[k][i] - q[i])**2
        distance.append(math.sqrt(dist))
elif (functionName == 'chebyshev'):
    for k in range(n):
        dist = 0.0
        for i in range(m):
            dist = max(math.fabs(values[k][i] - q[i]), dist)
        distance.append(dist)
else:
    distance = [0.0 for _ in range(n)]
copy = sorted(distance)
if windowType == 'fixed':
    windowWidth = hk
else:
    windowWidth = copy[hk]

kernelValues = []
if kernelName == 'uniform':
    for i in range(n):
        if windowWidth == 0 or math.fabs(distance[i] / windowWidth) >= 1:
            kernelValues.append(0.0)
            continue
        kernelValues.append(0.5)
elif kernelName == 'triangular':
    for i in range(n):
        if windowWidth == 0 or math.fabs(distance[i] / windowWidth) >= 1:
            kernelValues.append(0.0)
            continue
        kernelValues.append(1 - math.fabs(distance[i] / windowWidth))
elif kernelName == 'epanechnikov':
    for i in range(n):
        if windowWidth == 0 or math.fabs(distance[i] / windowWidth) >= 1:
            kernelValues.append(0.0)
            continue
        kernelValues.append(1 - (distance[i] / windowWidth)**2)
elif kernelName == 'quartic':
    for i in range(n):
        if windowWidth == 0 or math.fabs(distance[i] / windowWidth) >= 1:
            kernelValues.append(0.0)
            continue
        kernelValues.append((1 - (distance[i] / windowWidth)**2)**2)
elif kernelName == 'triweight':
    for i in range(n):
        if windowWidth == 0 or math.fabs(distance[i] / windowWidth) >= 1:
            kernelValues.append(0.0)
            continue
        kernelValues.append((1 - (distance[i] / windowWidth)**2)**3)
elif kernelName == 'tricube':
    for i in range(n):
        if windowWidth == 0 or math.fabs(distance[i] / windowWidth) >= 1:
            kernelValues.append(0.0)
            continue
        kernelValues.append((1 - math.fabs(distance[i] / windowWidth)**3)**3)
elif kernelName == 'gaussian':
    for i in range(n):
        if windowWidth == 0:
            kernelValues.append(0.0)
            continue
        kernelValues.append(math.exp(-0.5 * (distance[i] / windowWidth)**2))
elif kernelName == 'cosine':
    for i in range(n):
        if windowWidth == 0 or math.fabs(distance[i] / windowWidth) >= 1:
            kernelValues.append(0.0)
            continue
        kernelValues.append(math.cos(0.5 * math.pi * distance[i] / windowWidth))
elif kernelName == 'logistic':
    for i in range(n):
        if windowWidth == 0:
            kernelValues.append(0.0)
            continue
        kernelValues.append(1.0 / (math.exp(distance[i] / windowWidth) + 2 + math.exp(-distance[i] / windowWidth)))
elif kernelName == 'sigmoid':
    for i in range(n):
        if windowWidth == 0:
            kernelValues.append(0.0)
            continue
        kernelValues.append(1.0 / (math.exp(distance[i] / windowWidth) + math.exp(-distance[i] / windowWidth)))
else:
    kernelValues = [0.0 for _ in range(n)]

numerator = 0
denominator = sum(kernelValues)
for i in range(n):
    numerator += values[i][m] * kernelValues[i]
result = 0
if denominator > 0:
    result = numerator / denominator
else:
    count = len(list(filter(lambda x: x == 0, distance)))
    indices = [i for i, x in enumerate(distance) if x == 0]
    if windowWidth == 0 and count > 0:
        for elem in indices:
            result += values[elem][m] / count
    else:
        for i in range(n):
            result += values[i][m] / n
print(result)