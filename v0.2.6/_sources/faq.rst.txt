.. _faq:

Frequently Asked Questions
==========================

If something doesn't work as you expect, please first update to the latest
version of the UPET package (and simulation engines like
``lammps-metatomic``) before spending hours debugging or reporting an
issue. Our codebase evolves quickly and chances are your issue has been
fixed in a recent update. If you still see the problem after updating,
please check the FAQs below, and open an `issue
<https://github.com/lab-cosmo/upet/issues>`_ or a `discussion
<https://github.com/lab-cosmo/upet/discussions>`_ if you didn't find the
answer.


**Q: What model should I use for my application?**

**A:**

- For molecular dynamics simulations, we recommend the **PET-MAD v1.5.0**
  models.
- For materials discovery tasks (convex hull energies, geometry
  optimization, phonons, etc.), we recommend the **PET-OAM** models.
- For accurate and fast simulations of biomolecules, we recommend the
  **PET-SPICE** models.
- If you want to fine-tune your own model, we recommend starting from the
  **PET-OMat** checkpoints and choosing an appropriate size (XS to XL)
  for your needs.
- In any case, start from the smaller models (XS or S) to benchmark your
  application, then scale up if you need more accuracy.


**Q: The model is slow for my application. What should I do?**

**A:**

- Make sure you run it on a GPU.
- Use an S or XS model.
- Simulate with LAMMPS (KOKKOS-GPU version).
- Use non-conservative forces and stresses, preferably with multiple
  time-stepping. See `this example
  <https://atomistic-cookbook.org/examples/pet-mad-nc/pet-mad-nc.html>`_
  for details.
- Still too slow? Check out `FlashMD
  <https://github.com/lab-cosmo/flashmd>`_ for a further 30x boost.


**Q: My MD ran out of memory. How do I fix that?**

**A:**

- Reduce the model size (XS models are the least memory-intensive).
- Reduce the structure size.
- Try LAMMPS (KOKKOS-GPU version) and run with multiple MPI tasks to
  enable domain decomposition.
- As a last resort, use non-conservative forces and stresses.
- If you hit a weird bug when running more than 65535 atoms on GPU, this
  is not an out-of-memory bug but a PyTorch bug. You can work around it
  by adding ``torch.backends.cuda.enable_mem_efficient_sdp(False)``
  (after importing ``torch``) near the top of your script.


**Q: The model is not fully equivariant. Should I worry?**

**A:** Although our models are unconstrained, they are explicitly trained
for equivariance, and the equivariance error is, in the vast majority of
cases, one to two orders of magnitude smaller than the machine-learning
error with respect to the target electronic structure method. Hence:

- Read `this paper
  <https://iopscience.iop.org/article/10.1088/2632-2153/ad86a0>`_, which
  shows that the impact of non-equivariance on observables is often
  negligible. Proceed to the next two points **only** if you believe you
  are seeing effects due to non-equivariance.
- For MD, activate random frame averaging (a tutorial is coming).
- For geometry optimization, use a symmetrized calculator (see the
  ``rotational_average_order`` parameter in the ASE calculator,
  documented at :ref:`usage_ase`).


**Q: The XL models are huge!**

**A:** There are two aspects to this:

- The number of parameters is large, but they are used in a very sparse
  way and the evaluation cost is comparable to — and often lower than —
  that of other large models in the field.
- The listed cutoff radius may be large, but the cutoff strategy is
  adaptive: the model prunes the neighbor list internally. The effective
  cutoff for the vast majority of atomic environments in materials ends
  up being between 4 and 7 Å in practice.


**Q: I'm fine-tuning UPET models. Any other FAQs I should know about?**

**A:** Yes, please also see the `metatrain FAQs
<https://docs.metatensor.org/metatrain/latest/faq.html>`_.


Known issues
------------

**Simulation blows up with lammps-metatomic 2025.9.10.mta2**

While running UPET models with the version above, we observe that
simulations blow up. This is most likely a bug introduced in a recent PR
in ``lammps-metatomic``. Please update to the latest version of
``lammps-metatomic`` (``2025.9.10.mta3`` or later) to fix this.
