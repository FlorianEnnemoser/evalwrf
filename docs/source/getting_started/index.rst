.. _getting_started:

===============
Getting started
===============

Installation
-------------

.. grid:: 2
    :gutter: 4

    .. grid-item-card:: Working with development versions
        :class-card: install-card
        :padding: 3

        Development versions of evalwrf must be installed via `uv` and `pip`.
        Get the evalwrf `repository <https://github.com/FlorianEnnemoser/evalwrf>`__: 
        +++

        First pull the repository from github:

        .. code-block:: bash

            git clone https://github.com/FlorianEnnemoser/evalwrf.git

        Then, navigate to you folder and, with `uv <https://docs.astral.sh/uv>`__ installed in the current python enviromnet run the following:
        
        .. code-block:: bash

            uv sync
            uv pip install -e .

    .. grid-item-card:: Working with validated versions
        :class-card: install-card
        :padding: 3

        A validated evalwrf must be installed via pip:
        +++
        .. code-block:: bash

            pip install evalwrf

        or uv
            
        .. code-block:: bash

            uv pip install evalwrf
