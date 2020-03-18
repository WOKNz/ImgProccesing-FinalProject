# Imports
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np


def dots_texture_2d(size_of_circle, size_of_bkg_square, grid):
	"""

	:param size_of_circle: Radius of the circle in px (must be less than half square side)
	:param size_of_bkg_square: Size of one side of square in px
	:param grid: tuple (m,n) stacked squares with circles inside
	:return: one tiff img of square with circle inside
	"""
	bkg_square = Image.new('L', (size_of_bkg_square, size_of_bkg_square), color=255)
	# bkg_square.save('test.tiff','TIFF')
	draw = ImageDraw.Draw(bkg_square)
	initial_point = size_of_bkg_square / 2 - size_of_circle  # point from where to draw circle that center will be in middle
	draw.ellipse((initial_point, initial_point, size_of_bkg_square - initial_point, size_of_bkg_square - initial_point), \
	             fill=0)
	# bkg_square.save('test.tiff', 'TIFF')

	# stacking the images
	total_width = size_of_bkg_square * grid[0]
	max_height = size_of_bkg_square * grid[1]

	stacked_img = Image.new('RGB', (total_width, max_height))

	x_offset = 0
	for i in range(grid[0]):
		for j in range(grid[1]):
			stacked_img.paste(bkg_square, (size_of_bkg_square * i, size_of_bkg_square * j))

	return stacked_img


def rotate_imgae_in_3d(img, omega, phi, kappa, shift):
	"""

	:param img: tiff image of texture
	:param omega: in radians
	:param phi: in radians
	:param kappa: in radians
	:param shift: tuple (dx,dy) from where?
	:return: rotated and shifted image
	"""
	# Implementation
	pass


# Test area

# Testing 2D unrotated texture
square_sizes = [50, 75, 150, 300, 600]
circle_sizes = [2, 5, 10, 25, 50]

fig, axes = plt.subplots(5, 5)
for i in range(len(square_sizes)):
	for j in range(len(circle_sizes)):
		axes[i, j].imshow(dots_texture_2d(circle_sizes[i], square_sizes[j], (4, 4)), cmap='gray')
fig.text(0.5, 0.015, 'Change in Initial Square Size', ha='center', va='center')
fig.text(0.021, 0.5, 'Change in Initial Circle Size', ha='center', va='center', rotation='vertical')
plt.tight_layout()
plt.savefig('CompareOfRamaters.png', dpi=300)
plt.show()
