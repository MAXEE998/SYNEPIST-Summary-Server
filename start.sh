#!/usr/bin/env bash
service nginx start
uwsgi --ini SYNEPIST.ini
