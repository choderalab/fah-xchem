FROM continuumio/miniconda3

SHELL ["/bin/bash", "-c"]

COPY . fah-xchem

RUN conda env create -n fah-xchem -f fah-xchem/environment.yml \
 && source activate fah-xchem \
 && cd fah-xchem \
 && pip install . \
 && conda clean --all

RUN groupadd -g 999 fah-xchem \
 && useradd -m -r -u 999 -g fah-xchem fah-xchem

USER fah-xchem

ENV PATH /opt/conda/bin:$PATH
RUN echo "source activate fah-xchem" > ~/.bashrc

ENTRYPOINT /bin/bash -c "source activate fah-xchem && fah-xchem"
