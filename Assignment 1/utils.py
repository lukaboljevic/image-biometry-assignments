import os
from PIL import Image
import numpy as np


def read_dataset(image_size):
    """ Read the images from ./awe, convert them to grayscale and resize them """
    images = []
    classes = []
    for root, dirs, files in os.walk("./awe"):
        for dir in dirs:
            # the current dir = class
            current_dir_path = f"./awe/{dir}/"
            for file_name in os.listdir(current_dir_path):
                if file_name.endswith(".json"):
                    continue

                # Convert to grayscale and resize
                img_path = current_dir_path + file_name
                img = Image.open(img_path)
                img = img.convert("L")  # convert to grayscale
                img = img.resize(image_size)

                images.append(img)
                classes.append(int(dir))  # class of this image
        break

    return images, classes


def apply_histogram(lbp_image, block_size, bins_size):
    """
    A function that splits up the LBP image into blocks of fixed
    size, calculates the histogram for each block, and then
    concatenates all of them to obtain the final feature vector
    for the original image.
    """
    # I won't complicate now, with checking if these values are
    # okay; I will assume the appropriate block size is passed
    block_width, block_height = block_size
    height, width = lbp_image.shape

    vector = np.array([])

    for x in range(0, height, block_height):
        for y in range(0, width, block_width):
            values = []
            for i in range(block_height):
                for j in range(block_width):
                    values.append(lbp_image[x + i][y + j])
            
            h = np.histogram(values, bins=range(bins_size))[0]
            vector = np.append(vector, h)
    
    return vector


def cyclic_shift_to_smallest(binary_string):
    """
    Given a string of 1s and 0s, that actually represents a number, 
    return the smallest possible (decimal) number that can be obtained
    by cyclic shifting the original string/number

    Cyclic shifting to the smallest possible number is equivalent to 
    having the maximum number of most significant bits being all 
    consecutive 0s, but sounded like it needed more thinking, so...

    Cyclic shifting is used to achieve rotational invariance for LBP.
    """
    numbers = []
    for _ in range(len(binary_string)):
        binary_string = binary_string[1:] + binary_string[0]
        numbers.append(binary_string)

    decimal_numbers = list(map(lambda x: int(x, 2), numbers))
    return min(decimal_numbers)


def count_transitions(binary_string):
    """ 
    Counts the number of 0-1 and 1-0 transitions for
    the purpose of uniform LBP
    """
    length = len(binary_string)
    total = 0

    for i in range(length - 1):
        bits = binary_string[i:i+2]
        if bits == "01" or bits == "10":
            total += 1
    
    final_bits = binary_string[length-1] + binary_string[0]
    if final_bits == "01" or final_bits == "10":
        total += 1

    return total


def generate_neighbor_steps(R):
    """ 
    Generate the moves that need to be performed to locate 
    all neighbors of a pixel during LBP.

    Moves are created starting to the right of the center
    in a counter-clockwise notion (like in the original paper).
    """
    neighbors = []

    # Start to the right of the center
    x = R
    # start from -1 and not 0 because of the immediate +1 3 lines below.
    y = -1
    # this makes the code slightly more intuitive and understandable in my opinion

    # Go up to the top right corner
    while y < R:
        y += 1
        neighbors.append((x, y))

    # Then from the top right corner, go to the top left corner
    while x > -R:
        x -= 1
        neighbors.append((x, y))

    # From top left, go to bottom left
    while y > -R:
        y -= 1
        neighbors.append((x, y))

    # From bottom left, go to bottom right
    while x < R:
        x += 1
        neighbors.append((x, y))

    # From bottom right, go up to the center (starting point)
    while y < -1:
        y += 1
        neighbors.append((x, y))

    return neighbors


def inside(x, y, width, height):
    """ 
    Return True if position (x, y) is inside of a rectangle
    with dimensions width and height (more specifically, return
    True if pixel at (x, y) is inside of the image)
    """
    if x < 0 or x >= width or y < 0 or y >= height:
        return False
    return True
