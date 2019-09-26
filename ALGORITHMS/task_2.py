# Overview
#
# Resistors are electrical components marked with colorful stripes/bands to indicate both their resistance value in ohms and how tight a tolerance that value has. Remembering these bands is tricky, so let's write a function that will take a string identifying the resistor's ohms and return a string containing a resistor's band colors.
#
# The resistor color codes
#
# You can see this Wikipedia page for a colorful chart, but the basic resistor color codes are:
#
# color:	black	brown	red	orange	yellow	green	blue	violet	gray	white
# value:	0	1	2	3	4	5	6	7	8	9
# All resistors have at least three bands, with the first and second bands indicating the first two digits of the ohms value, and the third indicating the power of ten to multiply them by, for example a resistor with a value of 47 ohms, which equals 47 * 10^0 ohms, would have the three bands "yellow violet black".
#
# Most resistors also have a fourth band indicating tolerance -- in an electronics kit like yours, the tolerance will always be 5%, which is indicated by a gold band. So in your kit, the 47-ohm resistor in the above paragraph would have the four bands "yellow violet black gold".
#
# Your mission
#
# Your function will receive a string containing the ohms value you need, followed by a space and the word ohms. The way an ohms value is formatted depends on the magnitude of the value:
#
# For resistors less than 1,000 ohms, the ohms value is just formatted as the plain number. For example, with the 47-ohm resistor above, your function would receive the string "47 ohms", and return the string "yellow violet black gold".
#
# For resistors greater than or equal to 1000 ohms, but less than 1,000,000 ohms, the ohms value is divided by 1,000, and has a lower-case "k" after it. For example, if your function received the string "4.7k ohms", it would need to return the string "yellow violet red gold".
#
# For resistors of 1,000,000 ohms or greater, the ohms value is divided by 1,000,000, and has an upper-case "M" after it. For example, if your function received the string "1M ohms", it would need to return the string "brown black green gold".
#
# Test case resistor values will all be between 10 ohms and 990M ohms.
#
# More examples, featuring some common resistor values from your kit
#
# "10 ohms"        "brown black black gold"
# "100 ohms"       "brown black brown gold"
# "220 ohms"       "red red brown gold"
# "330 ohms"       "orange orange brown gold"
# "470 ohms"       "yellow violet brown gold"
# "680 ohms"       "blue gray brown gold"
# "1k ohms"        "brown black red gold"
# "10k ohms"       "brown black orange gold"
# "22k ohms"       "red red orange gold"
# "47k ohms"       "yellow violet orange gold"
# "100k ohms"      "brown black yellow gold"
# "330k ohms"      "orange orange yellow gold"
# "2M ohms"        "red black green gold"
# Specification
#
# encode_resistor_colors(ohmsString)
#
# Converts ohms to a color band
#
# Parameters
# ohmsString: String - Ohm properties to be parsed and encoded into color bands
# Return Value
# String - Sequence of color bands representing ohm properties
# Constraints
# 10 ohms ≤ Ω ≤ 990M ohms

import test


def encode_resistor_colors(ohms_string):
    # your code here

    color_list = ["black",
                  "brown",
                  "red",
                  "orange",
                  "yellow",
                  "green",
                  "blue",
                  "violet",
                  "gray",
                  "white"
                  ]

    if "k" in ohms_string:
        case = "less_1M"
        value = float(ohms_string.split("k")[0])
    elif "M" in ohms_string:
        case = "more_1M"
        value = float(ohms_string.split("M")[0])
    else:
        case = "less_1k"
        value = float(ohms_string.split()[0])

    hundreds = int(value/100)
    tens = int((value - hundreds*100)/10)
    units = int(value - hundreds*100 - tens*10)
    decimals = int((value*10 - units*10))

    if case == "less_1k":
        if value < 100:
            third_band = "black"
            second_band = color_list[units]
            first_band = color_list[tens]
        else:
            third_band = "brown"
            second_band = color_list[tens]
            first_band = color_list[hundreds]
    elif case == "less_1M":
        if value < 10:
            third_band = "red"
            second_band = color_list[decimals]
            first_band = color_list[units]
        elif value < 100:
            third_band = "orange"
            second_band = color_list[units]
            first_band = color_list[tens]
        else:
            third_band = "yellow"
            second_band = color_list[tens]
            first_band = color_list[hundreds]
    elif case == "more_1M":
        if value < 10:
            third_band = "green"
            second_band = color_list[decimals]
            first_band = color_list[units]
        elif value < 100:
            third_band = "blue"
            second_band = color_list[units]
            first_band = color_list[tens]
        else:
            third_band = "violet"
            second_band = color_list[tens]
            first_band = color_list[hundreds]

    fourth_band = "gold"

    out_string = first_band + " " + second_band + " " + third_band + " " + fourth_band

    return out_string


# TESTS

test.describe("Basic tests")
test.it("Some common resistor values")
test.assert_equals(encode_resistor_colors("10 ohms"), "brown black black gold")
test.assert_equals(encode_resistor_colors("47 ohms"), "yellow violet black gold")
test.assert_equals(encode_resistor_colors("100 ohms"), "brown black brown gold")
test.assert_equals(encode_resistor_colors("220 ohms"), "red red brown gold")
test.assert_equals(encode_resistor_colors("330 ohms"), "orange orange brown gold")
test.assert_equals(encode_resistor_colors("470 ohms"), "yellow violet brown gold")
test.assert_equals(encode_resistor_colors("680 ohms"), "blue gray brown gold")
test.assert_equals(encode_resistor_colors("1k ohms"), "brown black red gold")
test.assert_equals(encode_resistor_colors("4.7k ohms"), "yellow violet red gold")
test.assert_equals(encode_resistor_colors("10k ohms"), "brown black orange gold")
test.assert_equals(encode_resistor_colors("22k ohms"), "red red orange gold")
test.assert_equals(encode_resistor_colors("47k ohms"), "yellow violet orange gold")
test.assert_equals(encode_resistor_colors("100k ohms"), "brown black yellow gold")
test.assert_equals(encode_resistor_colors("330k ohms"), "orange orange yellow gold")
test.assert_equals(encode_resistor_colors("1M ohms"), "brown black green gold")
test.assert_equals(encode_resistor_colors("2M ohms"), "red black green gold")


test.assert_equals(encode_resistor_colors("8.6k ohms"), "gray blue red gold")
