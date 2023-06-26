def rgb_to_ilda_color(hex_color):
    if type(hex_color) == "string":
        rgb_color = hex_to_bit_color(hex_color)
    else:
        rgb_color = rgb_to_bit_color(hex_color)
    color = 0
    if rgb_color[0] == 1 and rgb_color[1] == 0 and rgb_color[2] == 0:  # RED
        color = 5
    elif rgb_color[0] == 1 and rgb_color[1] == 1 and rgb_color[2] == 0:  # YELLOW
        color = 15
    elif rgb_color[0] == 0 and rgb_color[1] == 1 and rgb_color[2] == 0:  # GREEN
        color = 25
    elif rgb_color[0] == 0 and rgb_color[1] == 1 and rgb_color[2] == 1:  # CYAN
        color = 34
    elif rgb_color[0] == 0 and rgb_color[1] == 0 and rgb_color[2] == 1:  # BLUE
        color = 42
    elif rgb_color[0] == 1 and rgb_color[1] == 0 and rgb_color[2] == 1:  # MAGENTA
        color = 52
    elif rgb_color[0] == 1 and rgb_color[1] == 1 and rgb_color[2] == 1:  # WHITE
        color = 63
    return color


def rgb_to_bit_color(color_rbg):
    r1 = 1 if color_rbg[0] > 128 else 0
    g1 = 1 if color_rbg[1] > 128 else 0
    b1 = 1 if color_rbg[2] > 128 else 0
    return [r1, g1, b1]


def rgb_to_7_color(color_rbg):
    r1 = 255 if color_rbg[0] > 128 else 0
    g1 = 255 if color_rbg[1] > 128 else 0
    b1 = 255 if color_rbg[2] > 128 else 0
    return [r1, g1, b1]


def hex_to_bit_color(color):
    hex_color = color.replace('#', '')
    r = int(hex_color.substring(0, 2), 16)
    g = int(hex_color.substring(2, 4), 16)
    b = int(hex_color.substring(4, 6), 16)
    r1 = 1 if r > 128 else 0
    g1 = 1 if g > 128 else 0
    b1 = 1 if b > 128 else 0
    return [r1, g1, b1]
