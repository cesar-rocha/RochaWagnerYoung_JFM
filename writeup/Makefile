LATEX       = pdflatex
BIBTEX      = bibtex
BASH        = bash -c
ECHO        = echo
RM          = rm -rf
RM_TMP      = ${RM} $(foreach suff, ${TMP_SUFFS}, ${NAME}.${suff})

TMP_SUFFS   = aux bbl blg log dvi ps eps out nav snm toc
SUFF        = pdf

CHECK_RERUN = grep Rerun $*.log

NAME = RochaWagnerYoung
DOC_OUT = ${NAME}.${SUFF}

default: ${DOC_OUT}

%.pdf: %.tex %.bib
	${LATEX} $<
	( ${CHECK_RERUN} && ${LATEX} $< ) || echo "Done."
	${BIBTEX} $(<:.tex=.aux)
	${LATEX} $<
	( ${CHECK_RERUN} && ${LATEX} $< ) || echo "Done."

clean:
	${RM_TMP}

