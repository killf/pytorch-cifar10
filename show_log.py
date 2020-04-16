import matplotlib.pyplot as plt
import os

log_file = "acc.txt"

if not os.path.exists(log_file):
    print(f"{log_file} does not exists.")
    exit(-1)

epoch, acc = [], []
with open(log_file, "r", encoding="utf8") as f:
    for line in f.readlines():
        line = line.strip()
        if len(line) == 0:
            continue

        ss = line.split('\t')
        epoch.append(int(ss[0]))
        acc.append(float(ss[1]) * 100)

plt.plot(epoch, acc)
plt.xlabel("epoch")
plt.ylabel("acc")
plt.ylim(0, 100)
plt.show()
