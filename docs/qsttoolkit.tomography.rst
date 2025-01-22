================================
qsttoolkit.tomography Subpackage
================================

This subpackage contains the models for performing quantum state tomography.

Traditional QST
===============

Traditional quantum state tomography is implemented using Maximum Likelihood Estimation (MLE):

.. toctree::
   :maxdepth: 2

   qsttoolkit.tomography.tradqst.MLE

Deep Learning QST
=================

Four deep learning models are currently implemented using `TensorFlow <https://www.tensorflow.org/>`_: one for quantum state discrimination and three for quantum state tomography. Each model has a dedicated class:

.. toctree::
   :maxdepth: 2

   qsttoolkit.tomography.dlqst.CNN_classifier
   qsttoolkit.tomography.dlqst.GAN_reconstructor
   qsttoolkit.tomography.dlqst.multitask_reconstructor

Global QST Utility Functions
============================

Functions used in both traditional and deep learning methods for quantum state tomography.

.. automodule:: qsttoolkit.tomography.QST
   :members:
   :undoc-members:
   :show-inheritance:
