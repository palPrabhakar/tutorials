"""
Batcher's Merge Exchange
"""
import argparse
import numpy as np
from functools import partial

def comparator(arr, i, j):
    if arr[i] > arr[j]:  # Tuple unpacking
        arr[i], arr[j] = arr[j], arr[i]

def is_sorted(arr):
    return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))

def gen_network(n):
    if n == 1:
        return []

    network = []
    t = int(np.ceil(np.log2(n)))
    p = 2**(t-1)
    while p > 0:
        q = 2**(t-1)
        r = 0
        d = p
        while True:
            for i in range(0, n - d):
                if i & p == r:
                    network.append(partial(comparator, i=i, j=i+d))

            if q != p:
                d = q - p
                q = int(q/2)
                r = p
            else:
                break
        p = int(np.floor(p/2))

    return network

def test(arr, network):
    for cmp in network:
        cmp(arr)

    if is_sorted(arr):
        print("Sorted!")
    else:
        print("Not Sorted!")

def main():
    args = parser.parse_args()
    n = args.n
    if n == 0:
        print("required: n > 0")
        exit()

    network = gen_network(n)
    arr = np.random.rand(n)
    test(arr, network)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a simple n-element sorting"
        "network using the principle of insertions")
    parser.add_argument("n", type=int, help="Enter the size sorting-network")
    main()
