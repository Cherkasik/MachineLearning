k = int(input())
n = int(input())
values = [list(map(int, input().split())) for _ in range(n)]
values = sorted(values)
allDistances = 0
classes = [[] for _ in range(k)]
inside = 0
for i in range(n):
    allDistances += 2 * values[i][0] * (2 * (i + 1) - n - 1)
    classes[values[i][1] - 1].append(values[i][0])
for elem in classes:
    leng = len(elem)
    for i in range(leng):
        inside += 2 * elem[i] * (2 * (i + 1) - leng - 1)
between = allDistances - inside
print(inside)
print(between)
