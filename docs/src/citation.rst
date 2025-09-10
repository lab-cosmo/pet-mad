##########
 Citation
##########

If you use PET-MAD in your research, please cite the appropriate papers:

***************
 PET-MAD Model
***************

For the main PET-MAD interatomic potential:

.. code:: bibtex

   @misc{PET-MAD-2025,
         title={PET-MAD, a universal interatomic potential for advanced materials modeling},
         author={Arslan Mazitov and Filippo Bigi and Matthias Kellner and Paolo Pegolo and Davide Tisi and Guillaume Fraux and Sergey Pozdnyakov and Philip Loche and Michele Ceriotti},
         year={2025},
         eprint={2503.14118},
         archivePrefix={arXiv},
         primaryClass={cond-mat.mtrl-sci},
         url={https://arxiv.org/abs/2503.14118}
   }

*******************
 PET-MAD-DOS Model
*******************

For the density of states model:

.. code:: bibtex

   @misc{PET-MAD-DOS-2025,
         title={A universal machine learning model for the electronic density of states},
         author={Wei Bin How and Pol Febrer and Sanggyu Chong and Arslan Mazitov and Filippo Bigi and Matthias Kellner and Sergey Pozdnyakov and Michele Ceriotti},
         year={2025},
         eprint={2508.17418},
         archivePrefix={arXiv},
         primaryClass={physics.chem-ph},
         url={https://arxiv.org/abs/2508.17418}
   }

**************
 Related Work
**************

If you use the uncertainty quantification features, please also consider
citing:

.. code:: bibtex

   @article{uncertainty_quantification,
         title={Uncertainty quantification in machine learning potentials},
         journal={Machine Learning: Science and Technology},
         year={2023},
         doi={10.1088/2632-2153/ad594a}
   }

   @article{ensemble_methods,
         title={Ensemble methods for machine learning potentials},
         journal={Machine Learning: Science and Technology},
         year={2023},
         doi={10.1088/2632-2153/ad805f}
   }

For non-conservative forces implementation:

.. code:: bibtex

   @misc{non_conservative_forces,
         title={Non-conservative forces in machine learning potentials},
         year={2024},
         eprint={2412.11569},
         archivePrefix={arXiv}
   }

***********************
 Software Dependencies
***********************

PET-MAD is built on several key software packages. If you use PET-MAD
extensively, please consider citing:

**Metatensor and Metatrain**:

.. code:: bibtex

   @software{metatensor,
         title={Metatensor: A library for sparse tensor operations},
         url={https://docs.metatensor.org}
   }

**ASE (Atomic Simulation Environment)**:

.. code:: bibtex

   @article{ase,
         title={The atomic simulation environment—a Python library for working with atoms},
         author={Larsen, Ask Hjorth and others},
         journal={Journal of Physics: Condensed Matter},
         volume={29},
         number={27},
         pages={273002},
         year={2017},
         publisher={IOP Publishing}
   }

**PyTorch**:

.. code:: bibtex

   @incollection{pytorch,
         title = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
         author = {Paszke, Adam and others},
         booktitle = {Advances in Neural Information Processing Systems 32},
         pages = {8024--8035},
         year = {2019},
         publisher = {Curran Associates, Inc.}
   }

*****************
 Acknowledgments
*****************

The development of PET-MAD was supported by:

-  The European Research Council (ERC) under the European Union's
   Horizon 2020 research and innovation programme
-  The Swiss National Science Foundation (SNSF)
-  EPFL and the Laboratory of Computational Science and Modeling (COSMO)

We thank the broader scientific community for providing datasets,
feedback, and computational resources that made this work possible.
