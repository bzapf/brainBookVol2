#!/bin/bash

WORKDIR=/home/basti/brainbook2/data/freesurfer/228

python normalize.py --inputfolder ${WORKDIR}/REGISTERED/ \
--exportfolder ${WORKDIR}/CONCENTRATIONS/ \
--t1map ${WORKDIR}/T1MAPS/t1_dolfin_subdomains.mgz # --refroi --mask 