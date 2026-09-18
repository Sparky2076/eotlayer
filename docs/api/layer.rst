EOTLayer
========

.. py:module:: eotlayer
   :synopsis: Differentiable balanced entropic optimal transport in PyTorch.

.. py:class:: EOTLAYER_CONSTRUCTOR_FROM_SOURCE

   A ``torch.nn.Module`` for balanced entropic optimal transport. Call the layer
   with ``layer(cost, source, target)`` to compute a plan or regularized value.

   :param float epsilon: Finite positive entropic regularization, in cost units.
   :param str mode: One of ``"unrolled"``, ``"implicit"``, or ``"envelope"``.
   :param iter_num: Positive integer, required only for ``unrolled`` mode.
   :param forward_tol: Positive normalized marginal tolerance. Defaults to
      ``1e-10`` in implicit and envelope modes.
   :param forward_iter_max: Positive forward iteration cap. Defaults to ``10000``
      in implicit and envelope modes.
   :param epsilon_scaling_start: Optional positive continuation start value,
      at least ``epsilon``. Available in implicit and envelope modes.
   :param epsilon_scaling_decay: Geometric continuation decay in ``(0, 1)``.
      Requires a start value; defaults to ``0.5`` when continuation is enabled.
   :param backward_solver: Implicit backend name. Defaults to ``"dense"``.
   :param backward_tol: Positive implicit adjoint relative tolerance. Defaults
      to ``1e-8``; Dense additionally applies its dtype-dependent floor.
   :param backward_iter_max: Positive iterative backward cap. Defaults to
      ``1000``. Dense uses direct solves and does not use this cap.

   Mode-specific arguments must be omitted where they do not apply. In
   particular, envelope mode rejects backward solver options, and unrolled mode
   rejects forward tolerance, continuation, and backward solver options.

Constructor options by mode
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Parameter
     - Unrolled
     - Implicit
     - Envelope
   * - ``epsilon`` and ``mode``
     - Required
     - Required
     - Required
   * - ``iter_num``
     - Required
     - Rejected
     - Rejected
   * - ``forward_tol``, ``forward_iter_max``
     - Rejected
     - Optional
     - Optional
   * - ``epsilon_scaling_start``, ``epsilon_scaling_decay``
     - Rejected
     - Optional
     - Optional
   * - ``backward_solver``, ``backward_tol``, ``backward_iter_max``
     - Rejected
     - Optional
     - Rejected

Calling the layer
-----------------

.. py:method:: EOTLayer.forward(cost, source, target)

   Solve one balanced problem or a batch. Calling ``layer(...)`` is the normal
   PyTorch entry point and delegates to this method.

   :param cost: Finite dense strided tensor of shape ``(m, n)`` or ``(B, m, n)``.
   :param source: Strictly positive source weights, shape ``(m,)`` or, for batched
      costs, ``(B, m)``.
   :param target: Strictly positive target weights, shape ``(n,)`` or, for batched
      costs, ``(B, n)``.
   :returns: A dense plan matching the cost shape in unrolled/implicit mode, or
      a scalar ``()`` / vector ``(B,)`` in envelope mode.
   :raises ValueError: Invalid shapes, nonfinite inputs, nonpositive weights,
      unequal total masses, or inconsistent dtype/device.
   :raises TypeError: Unsupported floating-point dtype.
   :raises RuntimeError: A closed layer, failure to converge, nonfinite results,
      or an unsupported backend execution path.

All dimensions must be nonempty. All three inputs must have the same device and
either ``torch.float32`` or ``torch.float64`` dtype. Shared and batched marginals
may be mixed. No automatic mass normalization or dtype/device conversion occurs.

For balanced mass validation, the implementation allows a discrepancy of at most
``100 * finfo(dtype).eps * max(source_mass, target_mass)``. This is a
floating-point consistency allowance, not an unbalanced-transport model.

Diagnostics and configuration
-----------------------------

.. py:property:: EOTLayer.diagnostics

   Tuple of independent per-item dictionary snapshots for the latest forward
   attempt, in batch order. Read it again after backward to obtain updated
   measurements. See :doc:`diagnostics` for every field.

.. py:property:: EOTLayer.epsilon

   Configured positive regularization scalar.

.. py:property:: EOTLayer.mode

   Selected differentiation mode.

The read-only properties ``iter_num``, ``forward_tol``, ``forward_iter_max``,
``epsilon_scaling_start``, ``epsilon_scaling_decay``, ``backward_solver``,
``backward_tol``, and ``backward_iter_max`` expose the resolved configuration.
Options not used by the selected mode return ``None``.

.. py:property:: EOTLayer.auto_diagnostics

   Experimental automatic-selection diagnostics keyed by internal problem slot.
   Returns an empty dictionary when automatic selection is not used. This is
   not needed for the canonical backends.

Resource lifetime
-----------------

.. py:property:: EOTLayer.closed

   Whether the layer or its implicit solver session has been closed.

.. py:method:: EOTLayer.close()

   Release owned solver resources and prevent future forward calls. Explicit
   closing is optional in normal use; Python object lifetime owns resources.
   Complete any pending backward work before closing.

Differentiation limits
----------------------

Implicit and envelope expose first-order custom backward rules. Double backward
is not supported by those modes. ``epsilon`` is scalar configuration, so no
gradient with respect to epsilon is returned. Cost and marginal tensors can
participate in normal first-order PyTorch differentiation.
