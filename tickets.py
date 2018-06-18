import sys
from itertools import groupby
import re

from funcy import re_find, silent, ignore, memoize
import cv2 as cv
import numpy as np
from PIL import Image, ImageFilter
from pytesseract import image_to_string
from Levenshtein import distance


def main():
    for filename in sys.argv[1:]:
        print(filename)
        text = ocr_tickets(filename)
        print(text)
        result = parse_tickets(text)
        print(result)


# Text handling part

def parse_tickets(text):
    result = {'tickets': {}, 'warnings': []}

    # Parse total number of tickets
    total_s, ticket_s = re.split(r'30\s*000', text, 2)
    total = silent(int)(re.sub(r'\D', '', re_find(r'[\d\s]+\D*$', total_s)))

    # Parse (nick, ticket count) pairs
    known_names = read_names()
    ticket_pairs = re.findall(r'(\w[\w\d ]*)\W+(\d+)?', ticket_s)
    tickets = {}
    for name, count in ticket_pairs:
        name = name.strip()
        if len(name) <= 1 or name.isdigit() or name == 'ДАЛЕЕ':
            continue
        count = silent(int)(count)

        # Check if name is known or similar enough to known one
        guess, warning = guess_name(name)
        if guess:
            result['tickets'][guess] = count
        if warning:
            result['warnings'].append(warning)

    return result


def guess_name(name):
    known_names = read_names()

    if name in known_names:
        return name, None
    else:
        distances = [distance(n, name) for n in known_names]
        min_dist = min(distances)
        if min_dist >= len(name):
            return None, 'Gave up on "%s"' % name
        if min_dist > 2:
            return name, 'No match for "%s"' % name

        candidates = [n for n, dist in zip(known_names, distances) if dist == min_dist]
        if len(candidates) > 1:
            return None, 'Multiple matches for "%s": %s' % (name, ', '.join(candidates))
        return candidates[0], "Recognized %s as %s" % (name, candidates[0])


@memoize
def read_names():
    with open('dict.txt') as f:
        return {l.strip() for l in f.readlines()}


# OCR part

def ocr_tickets(filename):
    image = Image.open(filename)
    block = crop_right(image)
    np_image = np.asarray(block)

    mask = select_colors(np_image)
    show(mask)
    return image_to_string(mask, lang='rus+eng')


def crop_right(im):
    W, H = im.size
    region = im.crop((0.7 * W, 0, W, H))
    return region


def select_colors(np_image):
    nick_color = np.array([242, 242, 242])
    numbers_color = np.array([153, 252, 252])

    nick_mask = cv.inRange(np_image, nick_color - 65, nick_color + 65)
    numbers_mask = cv.inRange(np_image, numbers_color - 60, numbers_color + 60)
    return nick_mask + numbers_mask


def show(im):
    """Show PIL or numpy image. Debug function"""
    if isinstance(im, Image.Image):
        im.show()
    else:
        Image.fromarray(im).show()



if __name__ == '__main__':
    main()
