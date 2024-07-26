#!/bin/bash

rsync -aPu utils springfield:genome_embedding/cog_presence
rsync -Pu cog-presence-test.py springfield:genome_embedding/cog_presence

frink run --follow cog_presence.yaml
