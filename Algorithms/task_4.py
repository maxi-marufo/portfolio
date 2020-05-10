# Given a non-negative integer, return an array / a list of the individual digits in order.
#
# Specification
# digitize(n)
#
# separate multiple digit numbers into an array
#
# Parameters
# n: Integer - Number to be converted
# Return Value
# Array (of Integers) - Array of separated single digit integers
# Examples
#
# n	Return Value
# 123	[1,2,3]
# 8675309	[8,6,7,5,3,0,9]


def digitize(n):
    return [int(ch) for ch in str(n)]
