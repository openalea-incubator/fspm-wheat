===================
README for FSPM-Wheat
===================

This is FSPM-Wheat, an example of model coupling for building a Functional Structural Plant Model of Wheat.


Prerequisites
=============

* To run the model: 
    * Python >= 2.7, http://www.python.org/
    * NumPy >= 1.7.2, http://www.numpy.org/
    * SciPy >= 0.12.1, http://www.scipy.org/
    * Pandas >= 0.14.0, http://pandas.pydata.org/
* To run the tools: Matplotlib >= 1.3.1, http://matplotlib.org/
* To build the documentation: Sphinx >= 1.1.3, http://sphinx-doc.org/
* To run the tests: Nose >= 1.3.0, http://nose.readthedocs.org/
* To get code coverage testing: Coverage >= 3.6b3, http://nedbatchelder.com/code/coverage/


Installing
==========

Use ``setup.py``::

   python setup.py install
   
To install in develop mode:: 
 
   python setup.py develop


Reading the docs
================

After installing::

   python setup.py build_sphinx

Then, direct your browser to ``_build/html/index.html``.


Testing
=======

To run the tests, use::

    nosetests


Contact
=======

Please send a mail to fspm-wheat@groupes.renater.fr.


Contributing
============

#. Check for open issues or open a fresh issue to start a discussion around a
   feature idea or a bug: https://sourcesup.renater.fr/tracker/?group_id=2736.
#. If you feel uncomfortable or uncertain about an issue or your changes, feel
   free to email fspm-wheat@groupes.renater.fr.
