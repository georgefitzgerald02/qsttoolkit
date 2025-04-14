==========================
qsttoolkit.data Subpackage
==========================

This subpackage contains functions and classes for generating QuTiP representations of specific optical quantum states. These states can be produced individually, in batches of specified size with randomized state parameters, or in specific preset datasets, intended to be standard datasets for modelling. States can be produced as pure states, or with some mixing applied to the density matrix. Measurement data is generated for direct photon occupation number measurement, or a Husimi Q function as a result of a phase space displace-and-measure technique [1]. Different sources of noise can be applied to the image data for the latter at customizable levels.

.. _ref1: references

Individual States
=================

.. automodule:: qsttoolkit.data.states
   :members:
   :undoc-members:
   :show-inheritance:

State Batches
=============

.. automodule:: qsttoolkit.data.state_batches
   :members:
   :undoc-members:
   :show-inheritance:

Datasets
========

.. automodule:: qsttoolkit.data.datasets
   :members:
   :undoc-members:
   :show-inheritance:

Noise
=====

.. automodule:: qsttoolkit.data.noise
   :members:
   :undoc-members:
   :show-inheritance:
