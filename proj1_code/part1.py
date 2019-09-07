#!/usr/bin/python3

import numpy as np
import math

def create_Gaussian_kernel(cutoff_frequency):
  """
  Returns a 2D Gaussian kernel using the specified filter size standard
  deviation and cutoff frequency.

  The kernel should have:
  - shape (k, k) where k = cutoff_frequency * 4 + 1
  - mean = k // 2 + 1
  - standard deviation = cutoff_frequency
  - values that sum to 1

  Args:
  - cutoff_frequency: an int controlling how much low frequency to leave in
    the image.
  Returns:
  - kernel: numpy nd-array of shape (k, k)

  HINT:
  - The 2D Gaussian kernel here can be calculated as the outer product of two
    vectors drawn from 1D Gaussian distributions.
  """

  ############################
  ### TODO: YOUR CODE HERE ###


  """
  # the general gaussian function
  # mean and std are calculated below
  def gaussian(x):
    # make sure it is symmetrical by starting with 1
    x += 1
    return np.exp(-np.power((x - mean) / std, 2) / 2) / (math.sqrt(2 * math.pi) * std)
  """

  gaussian = lambda x: np.exp(-np.power((x + 1 - mean) / std, 2) / 2) / (math.sqrt(2 * math.pi) * std)
  
  # preparing the parameters
  k = cutoff_frequency * 4 + 1
  mean = k // 2 + 1
  std = cutoff_frequency

  # generate the 1d gaussian array
  one_d_gaussian = np.fromfunction(gaussian, (k,))

  # validate the 1d gaussian by eyeballing, make sure it's symmetrical
  # print(one_d_gaussian)

  # compute the 2d array as the outer product of two 1d array
  kernel = np.outer(one_d_gaussian, one_d_gaussian)
  
  # normalize
  kernel /= np.sum(kernel)

  ### END OF STUDENT CODE ####
  ############################

  return kernel

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of shape (m, n, c)
  - filter: numpy nd-array of shape (k, j)
  Returns
  - filtered_image: numpy nd-array of shape (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using OpenCV or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###

  # 1. support grayscale and color images
  # 2. support arbitrarily-shaped filters with odd dimensions
  # 3. pad the input image with zeros or reflected image content
  # 4. return a filtered image which is the same resolution as the input image
  
  # padding
  padding_height = filter.shape[0] // 2
  padding_width = filter.shape[1] // 2
  padded_image = np.pad(image, ((padding_height, padding_height),(padding_width, padding_width), (0, 0)), "edge")

  """
  # padding manually
  padded_image = np.zeros((image.shape[0] + padding_height * 2, image.shape[1] + padding_width * 2, image.shape[2]))
  padded_image[padding_height:padding_height + image.shape[0], padding_width:padding_width + image.shape[1]] = image
  """

  # init the filtered image
  filtered_image = np.zeros((image.shape[0], image.shape[1], image.shape[2]))

  # reconstruct the filter to broadcast
  filter = filter.reshape(filter.shape[0], filter.shape[1], 1)

  # filtering
  for i in range(padding_height, filtered_image.shape[0] + padding_height):
    for j in range(padding_width, filtered_image.shape[1] + padding_width):
      filtered_box = filter * padded_image[i - padding_height:i + padding_height + 1, j - padding_width:j + padding_width + 1]
      # I used mean and I spent half an hour wondering why my image looks so dark
      filtered_pix = np.sum(filtered_box.reshape(filtered_box.shape[0]*filtered_box.shape[1],filtered_box.shape[2]), axis=0)
      filtered_image[i - padding_height][j - padding_width] = filtered_pix

  ### END OF STUDENT CODE ####
  ############################

  return filtered_image

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and a low-pass filter and creates a hybrid image. Returns
  the low frequency content of image1, the high frequency content of image 2,
  and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (x, y)
  Returns
  - low_frequencies: numpy nd-array of shape (m, n, c)
  - high_frequencies: numpy nd-array of shape (m, n, c)
  - hybrid_image: numpy nd-array of shape (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values of the hybrid image are between
    0 and 1. This is known as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]
  assert filter.shape[0] <= image1.shape[0]
  assert filter.shape[1] <= image1.shape[1]
  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###

  low_frequencies = my_imfilter(image1, filter)
  high_frequencies = image2 - my_imfilter(image2, filter)
  hybrid_image = np.clip(low_frequencies + high_frequencies, 0, 1)

  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image
