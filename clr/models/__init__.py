"""Clusterwise Linear Regression models."""
from .base import BaseCLR
from .vnd_recursive import CLR_VND_Logistic, CLR_VND_Tree
from .vnd_iterative import CLR_VND_TreeIterative
from .clr_kipok import CLR_Kipok