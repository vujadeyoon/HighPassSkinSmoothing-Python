#!/bin/bash
#
#
# Developer: vujadeyoon
# Email: vujadeyoon@gmail.com
#
#
readonly path_curr=$(pwd)
readonly path_parents=$(dirname "${path_curr}")
#
#
readonly repo_tag="${1:-vujadeyoon:latest}"
readonly git_maintainer="${2:-vujadeyoon}"
readonly git_mail="${3:-vujadeyoon@gamil.com}"
readonly path_git_proj="${4:-/home/dev/proj}"
readonly path_containerfile="${path_curr}/Containerfile"
#
#
sudo docker build --build-arg UID=$(id -u) \
                  --build-arg GID=$(id -g) \
                  --build-arg GIT_MAINTAINER=${git_maintainer} \
                  --build-arg GIT_EMAIL=${git_mail} \
                  --build-arg PATH_GIT_PROJ=${path_git_proj} \
                  -t ${repo_tag} -f ${path_containerfile} .
