# Your task is to split the chocolate bar of given dimension n x m into small squares.
# Each square is of size 1x1 and unbreakable.
# Implement a function that will return minimum number of breaks needed.
#
# For example if you are given a chocolate bar of size 2 x 1 you can split it to single squares in just one break, but for size 3 x 1 you must do two breaks.
#
# If input data is invalid you should return 0 (as in no breaks are needed if we do not have any chocolate to split). Input will always be a non-negative integer.
#
# Specification
# break_chocoloate(n, m)
#
# provide smallest amount of breaks to achieve all unbreakable squares
#
# Parameters
# n: Integer
# m: Integer
# Return Value
# Integer - The minimum amount of breaks
# Constraints
# 0 ≤ n ≤ 10
# 0 ≤ m ≤ 10
# Examples
#
# n	m	Return Value
# 5	5	"24"
# 7	4	"27"

import unittest


def break_chocolate(n,m):
    if n <= 1 and  m <= 1:
        return 0
    else:
        return (n*m-1)

# TESTS


# Note: the class must be called Test
class Test(unittest.TestCase):

    def test_should_handle_a_single_square_of_chocolate(self):
        self.assertEqual(break_chocolate(1, 1), 0)

    def test_should_handle_a_bigger_square_of_chocolate(self):
        self.assertEqual(break_chocolate(5, 5), 24)
