#!/bin/bash

WORKDIR=/home/basti/brainbook2/data/freesurfer/228

python normalize.py --inputfolder ${WORKDIR}/REGISTERED/ \
--exportfolder ${WORKDIR}/CONCENTRATIONS/ --refroi  