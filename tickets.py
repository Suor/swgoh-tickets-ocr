import sys
from itertools import groupby
import re

from funcy import re_find, silent, ignore, memoize, ldistinct, some, lmap, walk_keys, interleave
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
    ticket_pairs = re.findall(r'(\w[\w\d ]*)\W+((?:\w )?[\doOоО]+)?', ticket_s)
    print(ticket_pairs)
    tickets = {}
    for name, count in ticket_pairs:
        # Check if name is known or similar enough to known one
        name, warning = parse_name(name)
        if warning:
            result['warnings'].append(warning)
        if name is None:
            continue

        # Parse number part
        count = parse_number(count)
        if count is None:
            result['warnings'].append('No count for %s' % name)

        if count is not None:
            result['tickets'][name] = count

    return result


def parse_name(text):
    # Names to try:
    #   - full with no trailing nor dup spaces
    #   - one with single junk cleared from both ends
    #   - any part longer than 1 char
    name_parts = text.strip().split()
    full_name = ' '.join(name_parts)  # No trailing and dup spaces
    clean_name = re.sub(r'^\w\s|\s\w$', '', full_name)
    names = ldistinct([full_name, clean_name] + [n for n in name_parts if len(n) > 1])

    # Use "ocr normalized" version if raw one fails
    names = ldistinct(interleave(names, map(ocr_normalize, names)))

    versions = lmap(guess_name, names)
    return some(versions) or versions[0]


def guess_name(name):
    known_names = read_names()

    if len(name) <= 1 or name.isdigit() or name == 'ДАЛЕЕ':
        return None, None

    if name in known_names:
        return known_names[name], None

    distances = [distance(n, name) for n in known_names]
    min_dist = min(distances)
    if min_dist >= len(name):
        return None, 'Gave up on %s' % name
    if min_dist > 2:
        return name, 'No match for %s' % name

    candidates = [n for n, dist in zip(known_names, distances) if dist == min_dist]
    if len(candidates) > 1:
        return None, 'Multiple matches for %s: %s' % (name, ', '.join(candidates))
    guess = known_names[candidates[0]]
    return guess, "Recognized %s as %s" % (name, guess)


def parse_number(s):
    # Replace all sorts of Os, english and russian
    s = re.sub(r'[oOоО]', '0', s)
    numbers = re.findall(r'\d+', s)

    # Filter out single digits as noise
    if len(numbers) > 1:
        numbers = [s for s in numbers if len(s) > 1]

    if len(numbers) > 1:
        return None

    return silent(int)(numbers[0])


@memoize
def read_names():
    with open('dict.txt') as f:
        names = {l.strip(): l.strip() for l in f.readlines()}
        return {**walk_keys(ocr_normalize, names), **names}


# Russian to english, 0 to O
OCR_NORMALIZE_TABLE = str.maketrans('0КЕНХВАРОСМТукенгхаросмт', 'OKEHXBAPOCMTyKeHrxapocmT')

def ocr_normalize(s):
    return s.translate(OCR_NORMALIZE_TABLE).upper()


# OCR part

def ocr_tickets(filename):
    image = Image.open(filename)
    block = crop_right(image)
    np_image = np.asarray(block)

    mask = select_colors(np_image)
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
