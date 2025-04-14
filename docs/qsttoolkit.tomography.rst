================================
qsttoolkit.tomography Subpackage
================================

This subpackage contains models for performing quantum state tomography. Each model is implemented in its own class, each inheriting basic performance analysis and plotting functions from a parent class `qsttoolkit.tomography.QST`.

The fundamental aim of QSTToolkit is to provide modular, 'drag-and-drop' functions for researching, testing and comparing quantum state tomography methods in different experimental situations. Experimentation with combinations of tomography components is encouraged - for example, using a generator model with a different density matrix parametrization and loss function.

Traditional QST
===============

Traditional quantum state tomography is implemented using Maximum Likelihood Estimation (MLE):

.. toctree::
   :maxdepth: 2

   qsttoolkit.tomography.tradqst.MLE_reconstructor

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
