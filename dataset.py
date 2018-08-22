import numpy as np


def phase_img(image):
  """Return phase image for gs/rgb images."""
  """In the performed experiments, phase was calculated after the subtraction of images, so dPhase was calculated from dGS"""
  f = np.fft.rfft2(image, axes=(0,1))
  phase = np.angle(f)
  f = np.cos(phase) + 1j * np.sin(phase)
  image = np.fft.irfft2(f, axes=(0,1))
  return image
