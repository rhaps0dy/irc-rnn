import os
import pickle
import shutil

def makedirs_(path):
    if not os.path.exists(path):
        os.makedirs(path)

class CircularBufferQueueIterator:
    def __init__(self, queue):
        self.queue = queue
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.position == len(self.queue):
            raise StopIteration
        e = self.queue[self.position]
        self.position += 1
        return e

class CircularBufferQueue:
    def __init__(self, max_elements):
        self.n_elem = 0
        self.start = 0
        self.data = [None]*max_elements

    def __getitem__(self, index):
        if index >= self.n_elem or index < 0:
            raise IndexError("Queue index %d out of range" % index)
        return self.data.__getitem__((self.start + index) % len(self.data))

    def __setitem__(self, index, item):
        if index >= self.n_elem or index < 0:
            raise IndexError("Queue index %d out of range" % index)
        return self.data.__setitem__((self.start + index) % len(self.data), item)

    def __iter__(self):
        return CircularBufferQueueIterator(self)

    def full(self):
        return self.n_elem == len(self.data)

    def empty(self):
        return len(self) == 0

    def __len__(self):
        return self.n_elem

    def put(self, elem):
        if self.full():
            raise IndexError("Queue is full")
        self.n_elem += 1
        self[self.n_elem-1] = elem

    def get(self):
        if self.empty():
            raise IndexError("Queue is empty")
        a = self[0]
        self.n_elem -= 1
        self.start = (self.start + 1) % len(self.data)
        return a

def pkl_f_n(directory, number):
    return os.path.join(directory, "%03d.pkl" % number)

def load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def dump(data, fname):
    with open(fname, 'wb') as f:
        return pickle.dump(data, f)

def load_f_n(directory, number):
    return load(pkl_f_n(directory, number))

def dump_f_n(data, directory, number):
    return dump(data, pkl_f_n(directory, number))

def external_merge(inp_d, out_d, interval, len_list, key):
    """Recursively merge files in the interval"""
    start, end = interval
    if end < start + 2:
        for i in range(start, end):
            shutil.copy2(pkl_f_n(inp_d, i), pkl_f_n(out_d, i))
        return

    m = (start+end) // 2
    external_merge(inp_d, out_d, (start, m), len_list, key)
    external_merge(inp_d, out_d, (m, end), len_list, key)
    for i in range(start, end):
        os.remove(pkl_f_n(inp_d, i))
        shutil.move(pkl_f_n(out_d, i), pkl_f_n(inp_d, i))

    total_len = sum(len_list[i] for i in range(start, end))
    def out_file_len(i):
        n = end - start
        residual = 1 if i-start < total_len % n else 0
        return total_len // n + residual

    f_i = start
    f_j = m
    f_k = start
    list_i = load_f_n(inp_d, f_i)
    list_j = load_f_n(inp_d, f_j)
    list_k = []
    i = j = 0
    while f_i < m and f_j < end:
        k_len = out_file_len(f_k)
        while i < len(list_i) and j < len(list_j) and len(list_k) < k_len:
            if key(list_i[i]) < key(list_j[j]):
                list_k.append(list_i[i])
                i += 1
            else:
                list_k.append(list_j[j])
                j += 1
        if len(list_k) == k_len:
            dump_f_n(list_k, out_d, f_k)
            f_k += 1
            list_k = []
        if i == len(list_i):
            f_i += 1
            i = 0
            if f_i != m:
                list_i = load_f_n(inp_d, f_i)
        if j == len(list_j):
            f_j += 1
            j = 0
            if f_j != end:
                list_j = load_f_n(inp_d, f_j)

    if f_i == m:
        while f_j < end:
            k_len = out_file_len(f_k)
            while j < len(list_j) and len(list_k) < k_len:
                list_k.append(list_j[j])
                j += 1
            if len(list_k) == k_len:
                dump_f_n(list_k, out_d, f_k)
                f_k += 1
                list_k = []
            if j == len(list_j):
                f_j += 1
                j = 0
                if f_j != end:
                    list_j = load_f_n(inp_d, f_j)
    if f_j == end:
        while f_i < m:
            k_len = out_file_len(f_k)
            while i < len(list_i) and len(list_k) < k_len:
                list_k.append(list_i[i])
                i += 1
            if len(list_k) == k_len:
                dump_f_n(list_k, out_d, f_k)
                f_k += 1
                list_k = []
            if i == len(list_i):
                f_i += 1
                i = 0
                if f_i != m:
                    list_i = load_f_n(inp_d, f_i)

def external_sort(inp_d, sorted_d, n_files, key=lambda x: x):
    """ This function will overwrite the files in inp_d with partially sorted
    ones. Sort elements of the list of the #n_files files found in inp_d and
    store the output in sorted_d."""
    len_list = []
    for i in range(n_files):
        l = load_f_n(inp_d, i)
        l.sort(key=key)
        dump_f_n(l, inp_d, i)
        len_list.append(len(l))
    external_merge(inp_d, sorted_d, (0, n_files), len_list, key)

