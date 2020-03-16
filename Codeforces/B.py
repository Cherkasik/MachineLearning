k = int(input())
values = [list(map(int, input().split())) for _ in range(k)]
precision = []
recall = []
macroPrecision = 0
macroRecall = 0
macroF = 0
microF = 0
microPrecision = 0
all = sum([sum(row) for row in values])
for i in range(k):
    column = sum(row[i] for row in values)
    row = sum(values[i])
    curPrecision = 0
    curRecall = 0
    if column != 0:
        curPrecision = values[i][i] / column
    if row != 0:
        curRecall = values[i][i] / row
    precision.append(curPrecision)
    recall.append(curRecall)
for i in range(k):
    macroPrecision += precision[i] * sum(values[i]) / all
    macroRecall += recall[i] * sum(values[i]) / all
if (macroPrecision + macroRecall != 0):
    macroF = 2 * (macroPrecision * macroRecall) / (macroPrecision + macroRecall)
for i in range(k):
    if (precision[i] + recall[i] != 0):
        microPrecision += (2 * precision[i] * recall[i]) / (precision[i] + recall[i]) * sum(values[i]) / all
print(macroF)
print(microPrecision)
