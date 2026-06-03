# Thesis Template

## Prerequisites

You want to have the following things installed if you intend to use this
template:
 * A LaTeX backend (e.g. [TeX Live](https://www.tug.org/texlive) or [MikTeX](https://miktex.org))
 * A comfortable text editor (e.g. [Notepad++](https://notepad-plus-plus.org)) or a LaTeX frontend (e.g. [TeXstudio](http://www.texstudio.org))
   * [Visual Studio Code](https://code.visualstudio.com/) also has a quite usable [LaTeX extension](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)
 * A bibliography manager (e.g. [JabRef](http://www.jabref.org))

You may also use a Docker-based workflow. See below for details.


## HowTo

See [the automatically created PDF file](https://git.imp.fu-berlin.de/agse/thesis-template/-/jobs/artifacts/main/raw/thesis.pdf?job=compile-thesis)
for more documentation on the "configuration options" of your document.

To create a PDF file manually (i.e. if you don't use a LaTeX frontend), you need to call:
```bash
latexmk -pdf thesis
```

### Alternative

You can also use a Docker image to compile the LaTeX source:

```bash
docker run --rm -it -v $(pwd):/workdir danteev/texlive latexmk -pdf thesis.tex

# For continuous builds
docker run --rm -it -v $(pwd):/workdir danteev/texlive latexmk -pdf -pvc thesis.tex
```
 
