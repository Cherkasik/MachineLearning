n, m, k = map(int, input().split())
c = list(map(int, input().split()))
classes = [[] for _ in range(m)]
result = [[] for _ in range(k)]
for i in range(n):
    classes[c[i] - 1].append(i + 1)
current_position = 0
for i in range(m):
    for elem in classes[i]:
        result[current_position % k].append(elem)
        current_position += 1
for elem in result:
    print(len(elem), str(elem)[1:-1].replace(',', ''))
