import unittest
import tempfile
from random import randint
import shutil

import utils

class TestExternalSort(unittest.TestCase):
    def setUp(self):
        self.in_d = tempfile.mkdtemp()
        self.out_d = tempfile.mkdtemp()

        n_items = 0
        self.n_files = 10
        self.l = []

        for i in range(self.n_files):
            n = randint(10, 30)
            n_items += n
            l = list(randint(0, 1000000) for _ in range(n))
            self.l += l
            utils.dump_f_n(l, self.in_d, i)

        # Division rounding up
        self.len_first_file = (n_items + self.n_files - 1) // self.n_files

    def tearDown(self):
        shutil.rmtree(self.in_d)
        shutil.rmtree(self.out_d)

    def _check_sort(self):
        len_prev = self.len_first_file
        all_l = []
        for i in range(self.n_files):
            l = utils.load_f_n(self.out_d, i)
            # Force the file length to decrease, so we get all the files
            # being more or less the same length
            self.assertGreaterEqual(len_prev, len(l))
            len_prev = len(l)
            all_l += l
        self.assertEqual(self.l, all_l)

    def test_external_sort(self):
        """Test the output of the external sort is correct"""
        utils.external_sort(self.in_d, self.out_d, self.n_files)
        self.l.sort()
        self._check_sort()

    def test_external_sort_key(self):
        """Test correctness when using a key function"""
        utils.external_sort(self.in_d, self.out_d, self.n_files, key=lambda x: -x)
        self.l.sort(key=lambda x: -x)
        self._check_sort()

if __name__ == '__main__':
    unittest.main()
