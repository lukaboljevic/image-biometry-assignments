import os
from PIL import Image


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

    numbers = map(lambda x: int(x, 2), numbers)
    return min(numbers)


def generate_neighbor_indices(R):
    """ 
    Generate the moves that are to be performed to locate neighbors 
    of a pixel during LBP 
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
