from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
sys.path.append("../BGlib/")


class TestImport(unittest.TestCase):

    def test_basic(self):
        import BGlib as bgees
        print(bgees.__version__)
        self.assertTrue(True)
