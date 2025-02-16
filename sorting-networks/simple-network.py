import argparse
import numpy as np
from functools import partial


def comparator(arr, i, j):
    if arr[i] > arr[j]:  # Tuple unpacking
        arr[i], arr[j] = arr[j], arr[i]


def sorting_network_9():
    network = []
    network.append(partial(comparator, i=0, j=1))
    network.append(partial(comparator, i=3, j=4))
    network.append(partial(comparator, i=6, j=7))
    network.append(partial(comparator, i=1, j=2))
    network.append(partial(comparator, i=4, j=5))
    network.append(partial(comparator, i=7, j=8))
    network.append(partial(comparator, i=0, j=1))
    network.append(partial(comparator, i=3, j=4))
    network.append(partial(comparator, i=6, j=7))
    network.append(partial(comparator, i=0, j=3))
    network.append(partial(comparator, i=3, j=6))
    network.append(partial(comparator, i=0, j=3))
    network.append(partial(comparator, i=1, j=4))
    network.append(partial(comparator, i=4, j=7))
    network.append(partial(comparator, i=1, j=4))
    network.append(partial(comparator, i=2, j=5))
    network.append(partial(comparator, i=5, j=8))
    network.append(partial(comparator, i=2, j=5))
    network.append(partial(comparator, i=1, j=3))
    network.append(partial(comparator, i=5, j=7))
    network.append(partial(comparator, i=2, j=6))
    network.append(partial(comparator, i=4, j=6))
    network.append(partial(comparator, i=2, j=4))
    network.append(partial(comparator, i=2, j=3))
    network.append(partial(comparator, i=5, j=6))
    return network


def get_optimal_network(n):
    if n == 9:
        return sorting_network_9()
    else:
        exit(1)


def gen_network(n):
    if n == 1:
        return []

    network = gen_network(n-1)
    for i in range(n-1, 0, -1):
        network.append(partial(comparator, i=i-1, j=i))
    return network


def is_sorted(arr):
    return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))


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

    if n == 9:
        network = get_optimal_network(n)
    else:
        network = gen_network(n)
    arr = np.random.rand(n)
    test(arr, network)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a simple n-element sorting"
        "network using the principle of insertions")
    parser.add_argument("n", type=int, help="Enter the size sorting-network")
    main()
