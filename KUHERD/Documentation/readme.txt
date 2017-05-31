README for rebuilding the sphinx documentation

1) Change/Add/Remove the .rst files to reflect changed to the code
2) rebuild sphinxs via: "sphinx-build -b html . ./_build"
3) Find _build directory and run 'make html' and 'make latexpdf'
4) From the Documentation directory, copy in the generated PDF
5) Also, copy the html folder.


Commands for steps 4 & 5:
cp doc/_build/latex/KUHERD.pdf .
cp -r doc/_build/html .

