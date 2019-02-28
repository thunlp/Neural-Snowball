import sys

res_file = open(sys.argv[1])

total_res = [0, 0, 0, 0, 0, 0]
total = 0

for line in res_file.readlines():
    line = line.rstrip()
    if len(line) > 0:
        total += 1
        res = line.split()
        for i in range(6):
            total_res[i] += float(res[i])

for i in range(6):
    print(float(total_res[i]) / float(total))

print('----')

print((2 * total_res[1] * total_res[2] / (total_res[1] + total_res[2])))
print((2 * total_res[4] * total_res[5] / (total_res[4] + total_res[5])))
