.PHONY: directories

all: directories target
directories:
	-mkdir pdf

FILES := $(filter-out README, $(basename $(wildcard *.md)))
PDF := $(addsuffix .pdf, $(FILES))

target: $(PDF)

%.pdf: %.md
	pandoc -o "pdf/$@" "$<" --latex-engine=xelatex --wrap=preserve

.PHONY: clean

clean:
	-rm -rf pdf/
