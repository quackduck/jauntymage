import sys
from PIL import Image
import math

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
image = image.resize((256, 256))
pixels = image.load()
# image.show()
width, height = image.size

# have all values be in the range 0 to 1 by changing the image mode to 'F'
# image = image.convert('F')


# make a copy of the image
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
        # print(err_green, len(neighbors))
        err_red /= len(neighbors)
        err_green /= len(neighbors)
        err_blue /= len(neighbors)
        # err_green, err_red, err_blue = err_green ** 0.5, err_red ** 0.5, err_blue ** 0.5
        # err_red //= len(neighbors) # * 10
        # err_green //= len(neighbors) # * 10
        # err_blue //= len(neighbors) # * 10
        err = (round(err_red), round(err_green), round(err_blue))

        # set the pixel value at (x, y)
        pixels_copy[x, y] = err

        # get the RGB values
        # red = pixel[0] // 2
        # green = pixel[1] // 2
        # blue = pixel[2] // 2



        # set the pixel value at (x, y)
        # pixels_copy[x, y] = (red, green, blue)

# save the image
image_copy.save("out.png")

# # make another copy. this time to do a fourier transform
# image_copy = image.copy()
# pixels_copy = image_copy.load()

# # fft
import numpy as np
from scipy import fftpack

# rescale to 256 x 256
# image_copy = image_copy.resize((256, 256))

# convert to grayscale
image_copy = image_copy.convert('L')
pixels_copy = image_copy.load()

# convert to numpy array
image_array = np.array(image_copy)

# get shape
height, width = image_array.shape
print(height, width)

# do the fft
fft2 = fftpack.fft2(image_array)
fft2 = fftpack.fftshift(fft2)

# get the magnitude
magnitude = np.abs(fft2)
print(magnitude.shape)
print(width, height)

# # get the phase
phase = np.angle(fft2)

# # get the real part
# real = np.real(fft2)

# # get the imaginary part
# imaginary = np.imag(fft2)

magnitude2 =  magnitude / np.max(magnitude)

print("max:", np.max(magnitude2))

# max_mag = 0

# now write the magnitude to the image
for x in range(width):
    for y in range(height):
        # get the pixel value at (x, y)
        # pixel = pixels_copy[x, y]

        # get the magnitude
        # mag = 

        # mag = mag / (width * height)

        # if mag > max_mag:
        #     max_mag = mag



        # set the pixel value at (x, y)
        pixels_copy[x, y] = round(magnitude2[x, y]**0.5 * 255) # (round(mag), round(mag), round(mag))

# print(max_mag)
        
def wavy_x(x):
    return math.sin(10*x)/10 + x

# save the image
image_copy.save("out-fft.png")

# make sound using the derivative of the image
import simpleaudio as sa

# calculate the duration of the sound
sample_rate = 8000 # 44100  # Hz
# duration = 1.0 # seconds
# T = duration
# N = sample_rate * T
# print(N)

print()

phase = phase - np.min(phase)
phase = phase / np.max(phase)

# make sound using the phase of the fft
samples = np.zeros(sample_rate * 8)
i = 0
# t = np.linspace(0, 1, 1 * sample_rate, False)
t = np.arange(samples.size) / sample_rate

# print("magnitude2 max", np.max(magnitude2))
# exit(2)
for x in range(width):
    for y in range(height):
        # sound
        # get the pixel value at (x, y)
        # pixel = pixels_copy[x, y]
        m = magnitude2[y, x]
        p = phase[y, x]
        # print(p)
        # samples[i] = math.sin(i/sample_rate * 2 * math.pi * 293.665 * (1 + 0* p/2))
        # samples[i] += math.sin(i/sample_rate * 2 * math.pi * 440 * (1 + 0* p/2))
        # samples[i] += math.sin(i/sample_rate * 2 * math.pi * 1000 * (1 + 0* p/2))
        # add this phase to the full sound
        print(m)
        np.add(samples, m*np.sin(t * 2 * math.pi * 440 * wavy_x(1 - ((x-width/2)**2 + (y-height/2)**2)/((width**2 + height**2)/4)) + 2*math.pi*(0.0+1.0*p)), out=samples, casting="unsafe")
        # i += 1

# find the top 5 magnitudes and their phases
# top_magnitudes = np.zeros(5)
# top_phases = np.zeros(5)
# for x in range(width):
#     for y in range(height):
#         # get the magnitude
#         m = magnitude2[y, x]
#         p = phase[y, x]

#         for i in range(5):
#             if m > top_magnitudes[i]:
#                 top_magnitudes[i] = m
#                 top_phases[i] = p
#                 break

# for i in range(5):
#     print(top_magnitudes[i], top_phases[i])
#     np.add(samples, np.sin(t * 2 * math.pi * 440 * (0.0 + 1.0*top_magnitudes[i]) + 2*math.pi*top_phases[i]), out=samples, casting="unsafe")



print(samples, samples.size, max(samples), min(samples))

# convert samples to 16-bit values
samples *= 32767 / max(abs(samples))
# samples = np.array(samples)
print(samples, samples.size, max(samples), min(samples))
# samples *= 32767 # / np.max(np.abs(samples))
samples = samples.astype(np.int16)

print(samples, samples.size, max(samples), min(samples))

# start playback
play_obj = sa.play_buffer(samples, 1, 2, sample_rate)

# wait for playback to finish before exiting
play_obj.wait_done()

exit(2)

# write the phase this time
for x in range(width):
    for y in range(height):
        # get the pixel value at (x, y)
        # pixel = pixels_copy[x, y]

        # get the magnitude
        ph = phase[y, x]

        ph = (ph + np.pi) / (2 * np.pi) * 255

        print(ph)

        # set the pixel value at (x, y)
        pixels_copy[x, y] = round(ph) # (round(mag), round(mag), round(mag))

# save the image
image_copy.save("out-phase.png")

# now reconstruct the image
# do the ifft
fft2 = fftpack.ifftshift(fft2)
ifft2 = fftpack.ifft2(fft2)

# get the magnitude
magnitude = np.abs(ifft2)
print(magnitude.shape)
print(width, height)

# now write the magnitude to the image
for x in range(width):
    for y in range(height):
        # get the pixel value at (x, y)
        # pixel = pixels_copy[x, y]

        # get the magnitude
        mag = magnitude[y, x]
        # print(mag)

        # set the pixel value at (x, y)
        pixels_copy[x, y] = round(mag) # (round(mag), round(mag), round(mag))
        
# save the image
image_copy.save("out-ifft.png")

