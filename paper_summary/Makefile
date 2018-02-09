PDF_DIR=pdf/
.PHONY: directories

all: directories target
directories:
	-mkdir pdf

FILES := $(filter-out README, $(basename $(wildcard *.md)))
PDF := $(addprefix $(PDF_DIR), $(addsuffix .pdf, $(FILES)))

target: $(PDF)

$(PDF_DIR)%.pdf: %.md
	pandoc -o "$@" "$<" --latex-engine=xelatex --wrap=preserve

.PHONY: clean

clean:
	-rm -rf pdf/
