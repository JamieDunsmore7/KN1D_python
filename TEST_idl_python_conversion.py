import numpy as np
import sys

count = 0
while True:
    count += 1
    print('count:   ', count)
    if count < 10:
        continue
    print('should this be printed?')
    break
