import numpy as np
from .base import BaseActivation
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LU(BaseActivation):
	def __init__(self) -> None:
		pass

	def forward(self, Z):
		logger.debug(f"{self.__class__.__name__}: forward()")
		return Z

	def backward(self, Z):
		# TODO: check if it makes sense
		return Z
