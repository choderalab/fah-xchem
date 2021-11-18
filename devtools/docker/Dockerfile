FROM continuumio/miniconda3

SHELL ["/bin/bash", "-c"]

RUN conda env create -n fah-xchem -f environment.yml \
 && source activate fah-xchem \
 && pip install .

RUN groupadd -g 999 fah-xchem \
 && useradd -m -r -u 999 -g fah-xchem fah-xchem

USER fah-xchem

ENV PATH /opt/conda/bin:$PATH
RUN echo "source activate fah-xchem" > ~/.bashrc

ENTRYPOINT /bin/bash -c "source activate fah-xchem && fah-xchem"
