BGlib
=====

.. image:: https://github.com/pycroscopy/BGlib/workflows/build/badge.svg?branch=master
    :target: https://github.com/pycroscopy/BGlib/actions?query=workflow%3Abuild
    :alt: GitHub Actions

.. image:: https://img.shields.io/pypi/v/BGlib.svg
    :target: https://pypi.org/project/bglib/
    :alt: PyPI

.. image:: https://img.shields.io/pypi/l/sidpy.svg
    :target: https://pypi.org/project/sidpy/
    :alt: License

.. image:: http://pepy.tech/badge/BGlib
    :target: http://pepy.tech/project/BGlib
    :alt: Downloads

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4542239.svg
   :target: https://doi.org/10.5281/zenodo.4542239
   :alt: DOI

Band Excitation (BE) and General Mode (G-Mode) Scanning Probe Microscopy (SPM) data format translation, analysis and visualization codes.

MCP Server
----------

BGlib now includes a small MCP server for exposing a focused set of BE tools over stdio.

Install with the optional MCP dependency:

.. code-block:: bash

    pip install -e .[mcp]

Start the server:

.. code-block:: bash

    bglib-mcp

Exposed tools:

* ``LabViewPatcher(h5_path, force_patch=False)``
* ``projectLoop(vdc, amp_vec, phase_vec)``
* ``calc_switching_coef_vec(loop_coef_vec, nuc_threshold)``
* ``calculate_loop_centroid(vdc, loop_vals)``
* ``get_rotation_matrix(theta)``
