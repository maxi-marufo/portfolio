# Your task is to create a function that will take an integer and return the result of the look-and-say function on that integer. This should be a general function that takes as input any positive integer, and returns an integer; inputs are not limited to the sequence which starts with "1".
#
# Conway's Look-and-say sequence goes like this:
#
# Start with a number.
# Look at the number, and group consecutive digits together.
# For each digit group, say the number of digits, then the digit itself.
# Sample inputs and outputs:
#
# 1 is read as "one 1" => 11
# 11 is read as "two 1s" => 21
# 21 is read as "one 2, then one 1" => 1211
# 9000 is read as "one 9, then 3 0s" => 1930
# 222222222222 is read as "twelve 2s" => 122

import unittest


def look_say(number):

    digits = [i for i in str(number)]

    prev_digit = digits[0]
    splits = 0
    counts = [1]
    unique_digits = [prev_digit]

    if len(digits) == 1:
        out = "1" + str(prev_digit)
    else:
        for digit in digits[1:]:
            if digit == prev_digit:
                counts[splits] += 1
            else:
                splits += 1
                counts.append(1)
                unique_digits.append(digit)
            prev_digit = digit
        out = ""
    if splits == 0:
        out = str(counts[0])+str(unique_digits[0])
    else:
        for i in range(splits+1):
            str_to_append = str(counts[i])+str(unique_digits[i])
            out = out + str_to_append

    return int(out)

# TESTS


class Test(unittest.TestCase):
    def test_look_say_should_say_hello(self):
        self.assertEqual(look_say(0), 10)
        self.assertEqual(look_say(11), 21)
        self.assertEqual(look_say(12), 1112)
