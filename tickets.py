import sys
from itertools import groupby
import re

from funcy import re_find, silent, ignore, memoize, ldistinct, first, lmap, walk_keys, interleave
import cv2 as cv
import numpy as np
from PIL import Image, ImageFilter
from pytesseract import image_to_string
from Levenshtein import distance


__all__ = ['parse_file']


def main():
    for filename in sys.argv[1:]:
        print(parse_file(filename))


def parse_file(filename):
    # print(filename)
    text = ocr_tickets(filename)
    # print(text)
    return parse_tickets(text)


# Text handling part

def parse_tickets(text):
    result = {'tickets': {}, 'warnings': []}

    # Parse total number of tickets
    total_s, ticket_s = re.split(r'30\s*000', text, 2)
    result['total'] = silent(int)(re.sub(r'\D', '', re_find(r'[\d\s]+\D*$', total_s)))

    # Parse (nick, ticket count) pairs
    known_names = read_names()
    ticket_pairs = re.findall(r'(\w[\w\d \|]*)\W+((?:\w )?[\doOоО]+\b)?', ticket_s)
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
    return first((guess, warning) for guess, warning in versions if not warning) \
        or first((guess, warning) for guess, warning in versions if guess) \
        or (None, None)


def guess_name(name):
    known_names = read_names()

    if len(name) <= 1 or name.isdigit() or 'ДАЛЕЕ' in name:
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
    return known_names[candidates[0]], None


def parse_number(s):
    # Replace all sorts of Os, english and russian
    s = re.sub(r'[oOоО]', '0', s)
    numbers = re.findall(r'\d+', s)

    # Filter out single digits as noise
    if len(numbers) > 1:
        numbers = [s for s in numbers if len(s) > 1]

    if len(numbers) != 1:
        return None

    return silent(int)(numbers[0])


@memoize
def read_names():
    with open('dict.txt') as f:
        names = {l.strip(): l.strip() for l in f.readlines()}
        return {**walk_keys(ocr_normalize, names), **names}


# Russian to english, 0 to O
OCR_NORMALIZE_TABLE = str.maketrans('0КЕНХВАРОСМТукенгхапросмт|', 'OKEHXBAPOCMTyKeHrxanpocmTl')

def ocr_normalize(s):
    return s.translate(OCR_NORMALIZE_TABLE).upper()


# OCR part

def ocr_tickets(filename):
    image = Image.open(filename)
    block = crop_right(image)
    np_image = np.asarray(block)

    mask = select_colors(np_image)

    # Thiken text for better recognition, works sporadically
    # kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    # mask = cv.dilate(mask, kernel, iterations=1)
    # mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

    # Invert to get black on white, which works better for whatever reason
    mask = cv.bitwise_not(mask)

    # show(mask)
    return image_to_string(mask, lang='rus+eng')


def crop_right(im):
    W, H = im.size
    region = im.crop((0.73 * W, 0, W, H))
    return region


def select_colors(np_image):
    nick_color = np.array([242, 242, 242])
    numbers_color = np.array([153, 252, 252])

    nick_mask = cv.inRange(np_image, nick_color - 74, nick_color + 74)
    numbers_mask = cv.inRange(np_image, numbers_color - 80, numbers_color + 80)
    return nick_mask + numbers_mask


def show(im):
    """Show PIL or numpy image. Debug function"""
    _image(im).show()

def save(im, filename):
    """Save PIL or numpy image. Debug function"""
    _image(im).save(filename)

def _image(im):
    return im if isinstance(im, Image.Image) else Image.fromarray(im)


if __name__ == '__main__':
    main()
