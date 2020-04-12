# Imports
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import pandas as pd  # better printouts


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


def transfom(img_in, omega=None, phi=None, kappa=None, scale=None):
	"""

	:param img: PIL img
	:param omega: in degrees
	:param phi: in degrees
	:param kappa: in degrees
	:param scale: float
	:return: rotated and scaled image
	"""
	# Implementation
	img = cv2.cvtColor(np.array(img_in), cv2.COLOR_RGB2BGR)  # Convert to CV2 img type
	rows, cols = img.shape[:2]  # Max corner of the img

	r = R.from_euler('xyz', (omega, phi, kappa), degrees=True)  # Rotation Matrix
	# print(r.as_euler('xyz', degrees=True))
	# print(r.as_dcm())

	src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])  # Original Corners of the img
	rotation_mat = r.as_dcm() * 1  # Convert to np array
	points = np.hstack((src_points[:, 0].reshape(4, 1) - int(cols / 2), src_points[:, 1].reshape(4, 1) - int(rows / 2)))
	points = np.vstack((points.T, np.zeros((1, src_points.shape[0]))))
	dst_points = scale * np.dot(rotation_mat, points)

	# perspective on the image plane (corners)
	planeNormal = np.array([0, 0, 1])
	planePoint = np.array([0, 0, 0])
	rayPoint = np.array([0, 0, (img_in.width + img_in.height) * 2])  # Perspective point behind the plane of the img

	for i in range(4):  # Calculating the point on the img plane
		dst_points[:, i] = LinePlaneCollision(planeNormal, \
		                                      planePoint, \
		                                      dst_points[:, i] + np.array([0, 0, (img_in.width + img_in.height) * 2]), \
		                                      rayPoint)

	dst_points[0, :] = dst_points[0, :] + int(cols / 2)
	dst_points[1, :] = dst_points[1, :] + int(rows / 2)
	dst_points = np.float32(dst_points[0:2, :].T)
	# print(pd.DataFrame(dst_points))
	projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)  # Transformation matrix
	output = cv2.warpPerspective(img, projective_matrix, (cols, rows))  # Transformed image
	return Image.fromarray(output)


def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-12):
	ndotu = planeNormal.dot(rayDirection)
	if abs(ndotu) < epsilon:
		raise RuntimeError("no intersection or line is within plane")

	w = rayPoint - planePoint
	si = -planeNormal.dot(w) / ndotu
	Psi = w + si * rayDirection + planePoint
	return Psi


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
plt.savefig('CompareOfPamaters.png', dpi=300)
plt.show()

# # Testing 2D transformed texture
omegas = [5, 10, 20, 40, 80]
phis = [5, 10, 20, 40, 80]
# phis = [5, 10, 20, 40, 80]

fig, axes = plt.subplots(5, 5)
for i in range(len(omegas)):
	for j in range(len(phis)):
		rotated = transfom(dots_texture_2d(25, 75, (8, 8)), omegas[i], phis[j], 0, 0.8)
		axes[i, j].imshow(rotated, cmap='gray')
fig.text(0.5, 0.015, 'Change in Phi Angle', ha='center', va='center')
fig.text(0.021, 0.5, 'Change in Omega Angle', ha='center', va='center', rotation='vertical')
plt.tight_layout()
plt.savefig('CompareOfAngles.png', dpi=300)
plt.show()

Image.Image.show(transfom(dots_texture_2d(25, 75, (8, 8)), 35, 13, 8, 0.6))
