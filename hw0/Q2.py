import sys
from PIL import Image

# Load argument from bash.
IMAGE_FILE_PATH = sys.argv[1]

# Load image from file.
originalImage = Image.open(IMAGE_FILE_PATH)
# Get width and height of image.
width, height = originalImage.size
# print 'width = %d, height = %d' %(width, height)

# New image with the same size and 'RBG' format.
outputImage = Image.new('RGB', originalImage.size)

for i in range(0, width):
    for j in range(0, height):
        # Get RGB for each pixel.
        r, g, b = originalImage.getpixel((i, j))
        # Put (RGB) // 2 for each pixel.
        outputImage.putpixel((i, j), (r // 2, g // 2, b // 2))
        
outputImage.save('Q2.png')