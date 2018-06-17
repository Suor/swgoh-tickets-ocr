import sys
from itertools import groupby
import re

from funcy import re_find, silent, ignore, select_keys, walk_values
from funcy import lzip, lmap, collecting, lpartition_by, join, first, last, split_by, pairwise
import cv2 as cv
import numpy as np
from PIL import Image, ImageFilter
from pytesseract import image_to_string


def main():
    # Read known names
    with open('dict.txt') as f:
        names = [l.strip() for l in f.readlines()]

    for filename in sys.argv[1:]:
        # Recognize text
        im = Image.open(filename)
        block = crop_right(im)
        ndim = np.asarray(block)

        nick_color = np.array([242, 242, 242])
        numbers_color = np.array([153, 252, 252])

        nick_mask = cv.inRange(ndim, nick_color - 65, nick_color + 65)
        numbers_mask = cv.inRange(ndim, numbers_color - 60, numbers_color + 60)
        mask = nick_mask + numbers_mask

        text = image_to_string(mask, lang='rus+eng')
        print(text)

        # Text recognition
        total_s, ticket_s = re.split(r'30\s*000', text, 2)
        total = silent(int)(re.sub(r'\D', '', re_find(r'[\d\s]+\D*$', total_s)))

        ticket_pairs = re.findall(r'(\w[\w\d ]*)\W+(\d+)?', ticket_s)
        tickets = {name.strip(): n for name, n in ticket_pairs}
        tickets = select_keys(lambda k: len(k) > 1 and not k.isdigit() and k != 'ДАЛЕЕ', tickets)
        tickets = walk_values(ignore(ValueError, default=0)(int), tickets)

        result = {'total': total, 'tickets': tickets}
        print(result)

        # W, H = im.size
        # region = im.crop((0.7 * W, 0, W, H))
        # region.show()
        # region = im
        # boxes = image_to_boxes(region, lang='rus+eng', output_type='dict')
        # labels = 'char', 'left', 'top', 'right', 'bottom'
        # chars = lzip(*[boxes[l] for l in labels])
        # words = [''.join(c[0] for c in word) for word in partition_by_borders(is_adjacent, chars)]

        # print(chars)
        # print(filename, words)


def show(im):
    if isinstance(im, Image.Image):
        im.show()
    else:
        Image.fromarray(im).show()


def crop_right(im):
    W, H = im.size
    region = im.crop((0.7 * W, 0, W, H))
    return region

# def select_colors(im)

def is_adjacent(char1, char2):
    _, left1, top1, right1, bottom1 = char1
    _, left2, top2, right2, bottom2 = char2
    return (abs(top1 - top2) <= 2 or abs(bottom1 == bottom2) <= 2) and abs(left2 - right1) <= 8


def partition_by_borders(f, seq):
    """Lazily partition when f(prev, val) is false."""
    for fit, items in groupby(pairwise(seq), lambda pair: f(*pair)):
        if fit:
            items = list(items)
            yield lmap(0, items) + [items[-1][-1]]



def edges(im):
    W, H = im.size
    im_edges = im.filter(ImageFilter.FIND_EDGES)

    # Find common vertical borders for several heights
    heights = [int(H * ratio) for ratio in (0.4, 0.6, 0.85)]
    candidates = ({i for i, val in enumerate(xblues(im_edges, h)) if val > 200} for h in heights)
    h_edges = sorted(set.intersection(*candidates))

    # Find left and right borders around 80% into image width
    ls, rs = split_by(lambda x: x < 0.8 * W, h_edges)
    left, right = last(ls), first(rs)

    # Find top and bottom borders of a box
    vs = list(yblues(im_edges, left))

    # return [(len(v), v[0]>200) for v in lpartition_by(lambda x: x > 200, ]


def xblues(im, y):
    "Returns an iterator of blue values for given y"
    for x in range(im.width):
        r, g, b = im.getpixel((x, y))
        yield b


def yblues(im, x):
    "Returns an iterator of blue values for given x"
    for y in range(im.height):
        r, g, b = im.getpixel((x, y))
        yield b


if __name__ == '__main__':
    main()

# print(image_to_boxes(Image.open('single1.png'), lang='rus+eng'))
# print(image_to_boxes(Image.open('single1.png'), lang='rus+eng', output_type='dict'))
# im = Image.open('full1.png')

# print(edges(im))
# print([i for i, val in enumerate(blues(im_edges)) if val > 200])

# print(['%dx%d' % (int(sum(xs) / len(xs)), len(xs))
#       for xs in lpartition_by(lambda x: x // 50, blues(im_edges))])
# W, H = im.size
# middle_y = H // 2

# for x in range(W-1, 0, -1):
#     print(im.getpixel((x, middle_y)))

# blues = [b for r, g, b in]


# print(image_to_string(im, lang='rus+eng'))
# print(image_to_boxes(im, lang='rus+eng', output_type='dict'))

