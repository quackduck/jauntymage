import sys
from PIL import Image
import math
import time

# Check if the command line argument is provided
if len(sys.argv) < 2:
    print("Please provide the path to an image file as a command line argument.")
    sys.exit(1)

# Get the image file path from the command line argument
image_path = sys.argv[1]

# Open the image
try:
    image = Image.open(image_path)
except IOError:
    print("Unable to open image file:", image_path)
    sys.exit(1)

# Get the RGB pixel values
# image = image.resize((256, 256))
image = image.resize((64, 64))
pixels = image.load()
# image.show()
width, height = image.size


# # make a copy of the image
image_copy = image.copy()
pixels_copy = image_copy.load()

# loop through the image pixels
for x in range(width):
    for y in range(height):
        # get the pixel value at (x, y)
        pixel = pixels[x, y]

        radius = 1
        neighbors = []
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if x+i >= 0 and x+i < width and y+j >= 0 and y+j < height:
                    if i == 0 and j == 0:
                        continue
                    if i ** 2 + j ** 2 <= radius ** 2:
                        neighbors.append(pixels[x+i, y+j])

        # get the average of the neighbors
        err_red = 0
        err_green = 0
        err_blue = 0
        for neighbor in neighbors:
            err_red += (neighbor[0] - pixel[0]) ** 2
            err_green += (neighbor[1] - pixel[1]) ** 2
            err_blue += (neighbor[2] - pixel[2]) ** 2
        err_red /= len(neighbors)
        err_green /= len(neighbors)
        err_blue /= len(neighbors)
        err_green, err_red, err_blue = err_green ** 0.5, err_red ** 0.5, err_blue ** 0.5
        err = (round(err_red), round(err_green), round(err_blue))

        # set the pixel value at (x, y)
        pixels_copy[x, y] = err

# multiply the image such that the max value is 255
max_pixel = 0
for x in range(width):
    for y in range(height):
        pixel = pixels_copy[x, y]
        for i in range(3):
            if pixel[i] > max_pixel:
                max_pixel = pixel[i]
print(max_pixel)
for x in range(width):
    for y in range(height):
        pixel = pixels_copy[x, y]
        for i in range(3):
            pixels_copy[x, y] = (pixel[0] * 255 // max_pixel, pixel[1] * 255 // max_pixel, pixel[2] * 255 // max_pixel)

# find new max
max_pixel = 0
for x in range(width):
    for y in range(height):
        pixel = pixels_copy[x, y]
        for i in range(3):
            if pixel[i] > max_pixel:
                max_pixel = pixel[i]
print(max_pixel)

def activation(x):
    # if x < 0.01:
    #     return 0
    return x ** 2
    # return 0
    # return 1 / (1 + math.exp(-22*(x-0.75)))

def waveform(x, mode=0):
    # square wave with fourier series
    # return np.sin(x) + np.sin(x/4) # + np.sin(3*x)/3
    coeffs = [1, 0.65, 0.61, 0.15, 0.09, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01] if mode == 0 else [1, 0.61, 0.10, 0.24, 0.11, 0.09, 0, 0.02, 0, 0.00, 0.09, 0.01] # if mode == 1 else [1, 0.2, 0.1, 0.05, 0.01, 0.01, 0.01, 0.01]
    # coeffs = [1, 0.33, 0.33, 0.33] if mode == 0 else [1, 0.0, 0.0, 0.0, 0.0, 1]
    # coeffs = [1]
    result = np.sin(x)
    for i in range(1, len(coeffs)):
        result += coeffs[i] * np.sin((i+1)*x)
    return result
        


# write the image to a file
image_copy.save("out.png")


# make sound using the derivative of the image
import simpleaudio as sa
import numpy as np

# calculate the duration of the sound
sample_rate = 8000 # 44100  # Hz
beat_length = 0.4 # seconds
interval = 0.0 # seconds
print()

pentatonic_scale = [440, 493.88, 554.37, 659.25, 739.99] # [830.61, 987.77, 1108.73, 1318.51, 1479.98, 1661.22, 1975.53, 2217.46, 2637.02, 2959.96, 3322.44, 3951.07, 4434.92, 5274.04, 5919.91, 6644.88, 7902.13]

a_minor_penta = [440, 523.25, 587.33, 659.25, 783.99]

a_minor_blues = [440, 523.25, 587.33, 622.25, 659.25, 783.99]

# scale_list = [pentatonic_scale, a_minor_penta]
scale = pentatonic_scale

# resize to height 8
# image_copy = image_copy.resize((image_copy.size[0], 4))
# image_copy = image_copy.resize((image_copy.size[0], 8))
image_copy = image_copy.resize((image_copy.size[0], len(scale)))

pixels_copy = image_copy.load()
width, height = image_copy.size

all_samples = np.zeros(0)

# i = 0
# t = np.linspace(0, 1, 1 * sample_rate, False)

# center_freq = 440/2

# print("magnitude2 max", np.max(magnitude2))
# exit(2)
for x in range(width):

    range_map1 = {range(height//2): 0.5, 
                  range(height//2, height): 0.5}
    range_map2 = {range(height//4): 10.5, 
                 range(height//4, 3*height//4): 0.25, 
                 range(3*height//4,height): 0.125}
    
    range_map = range_map1 # if x % 32 != 0 else range_map2

    for r in range_map.keys():
        samples = np.zeros(round(sample_rate * beat_length * range_map[r]))
        t = np.arange(samples.size) / sample_rate
        for y in r:
            pixel = pixels_copy[x, y]
            # freq = center_freq * 2 ** ((y - height/2)/8)
            freq = scale[y]
            # freq = scale_list[x%2][y]
            mag = ((pixel[0]**2 + pixel[1]**2 + pixel[2]**2) / (255**2) / 3) ** 0.5
            print(freq, mag, activation(mag))
            np.add(samples, activation(mag)*waveform(t * 2 * math.pi * freq, x%2), out=samples, casting="unsafe")

            np.add(samples, 2*activation(mag)*waveform(t * 2 * math.pi * scale[97*y % len(scale)] / 4, (x+1)%2), out=samples, casting="unsafe") # base
        
        all_samples = np.append(all_samples, samples)

all_samples *= 32767 / max(abs(all_samples))
all_samples = all_samples.astype(np.int16)
play_obj = sa.play_buffer(all_samples, 1, 2, sample_rate)
play_obj.wait_done()
