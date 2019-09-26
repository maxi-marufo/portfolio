# Task
# In this challenge you parse RGB colors represented by strings. The formats are primarily used in HTML and CSS. Your task is to implement a function which takes a color as a string and returns the parsed color as a map (see Examples).
#
# Input
#
# The input string color represents one of the following:
#
# 1. 6-digit hexadecimal
#
# "#RRGGBB" - Each pair of digits represents a value of the channel in hexadecimal: 00 to FF
#
# 2. 3-digit hexadecimal
#
# "#RGB" - Each digit represents a value 0 to F which translates to 2-digit hexadecimal: 0->00, 1->11, 2->22, and so on.
#
# 3. Preset color name
#
# You have to use the predefined map PRESET_COLORS (Ruby, Python, JavaScript), presetColors (Haskell), or preset-colors (Clojure). The keys are the names of preset colors in lower-case and the values are the corresponding colors in 6-digit hexadecimal (same as 1. "#RRGGBB").
#
# Specification
# parse_html_color(color)
#
# Takes a String that represents a color and returns the parsed color as a map.
#
# Parameters
# color: String - A color represented by color name, 3-digit hexadecimal or 6-digit hexadecimal
# Return Value
# dict - A set of numerical RGB values
# Examples
#
# color	Return Value
# "#80FFA0"	{"r":128,"g":255,"b":160}
# "#3B7"	{"r":51,"g":187,"b":119}
# "LimeGreen"	{"r":50,"g":205,"b":50}

import test


def parse_html_color(color):
    color_type = checkStringType(color)
    if color_type == "6-digit":
        r, g, b = convert6digitHex(color)
    elif color_type == "3-digit":
        r, g, b = convert3digitHex(color)
    elif color_type == "preset":
        r, g, b = convert6digitHex(PRESET_COLORS[color.lower()])
    else:
        print ("ERROR checking color string type")
    return {"r": r, "g": g, "b": b}


def checkStringType(color):
    if (color[0] == "#"):
        if len(color) == 7:
            return "6-digit"
        elif len(color) == 4:
            return "3-digit"
    else:
        if color.lower() in PRESET_COLORS:
            return "preset"
    return "ERROR"


def convert6digitHex(color):
    r = int(color[1], 16)*16+int(color[2], 16)
    g = int(color[3], 16)*16+int(color[4], 16)
    b = int(color[5], 16)*16+int(color[6], 16)
    return (r, g, b)


def convert3digitHex(color):
    r = int(color[1], 16)*16+int(color[1], 16)
    g = int(color[2], 16)*16+int(color[2], 16)
    b = int(color[3], 16)*16+int(color[3], 16)
    return (r, g, b)


# TESTS

test.describe('Example tests')
test.assert_equals(parse_html_color('#80FFA0'),   {'r': 128, 'g': 255, 'b': 160})
test.assert_equals(parse_html_color('#3B7'),      {'r': 51,  'g': 187, 'b': 119})
test.assert_equals(parse_html_color('LimeGreen'), {'r': 50,  'g': 205, 'b': 50 })
