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
    if x < 0.05:
        return 0
    return x ** 0.5
    # return 0
    # return 1 / (1 + math.exp(-22*(x-0.75)))

def waveform(x):
    # square wave with fourier series
    # return np.sin(x) + np.sin(x/4) # + np.sin(3*x)/3
    coeffs = [1, 0.65, 0.61, 0.15, 0.09, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01]
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
beat_length = 0.5 # seconds
interval = 0.2 # seconds
print()

# resize to height 8
image_copy = image_copy.resize((image_copy.size[0], 8))
pixels_copy = image_copy.load()
width, height = image_copy.size

# i = 0
# t = np.linspace(0, 1, 1 * sample_rate, False)

center_freq = 440

# print("magnitude2 max", np.max(magnitude2))
# exit(2)
for x in range(width):
    samples = np.zeros(round(sample_rate * beat_length))
    t = np.arange(samples.size) / sample_rate
    for y in range(height):
        # sound
        # get the pixel value at (x, y)
        pixel = pixels_copy[x, y]
        # m = magnitude2[y, x]
        # p = phase[y, x]
        # print(p)
        # samples[i] = math.sin(i/sample_rate * 2 * math.pi * 293.665 * (1 + 0* p/2))
        # samples[i] += math.sin(i/sample_rate * 2 * math.pi * 440 * (1 + 0* p/2))
        # samples[i] += math.sin(i/sample_rate * 2 * math.pi * 1000 * (1 + 0* p/2))
        # add this phase to the full sound
        # print(p**0.5)
        # freq = wavy_x(1 - ((x-width/2)**2 + (y-height/2)**2)/((width**2 + height**2)/4)) * 440

        # frequency is the distance from the origin
        # freq = ((x-width/2)**2 + (y-height/2)**2)**0.5 * 30

        freq = 440 * 2 ** ((y - height/2)/8)
        mag = ((pixel[0]**2 + pixel[1]**2 + pixel[2]**2) / (255**2) / 3) ** 0.5

        print(freq, mag, activation(mag))

        np.add(samples, activation(mag)*waveform(t * 2 * math.pi * freq), out=samples, casting="unsafe")

    samples *= 32767 / max(abs(samples))
    samples = samples.astype(np.int16)
    play_obj = sa.play_buffer(samples, 1, 2, sample_rate)
    play_obj.wait_done()
    # wait for interval
    time.sleep(interval)
