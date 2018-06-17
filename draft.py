import sys

from funcy import collecting, lpartition_by, join, first, last, split_by
from PIL import Image, ImageFilter
from pytesseract import image_to_string, image_to_boxes


def main():
    for filename in sys.argv[1:]:
        im = Image.open(filename)
        print(filename, edges(im))



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

