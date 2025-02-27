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
readonly volume="${2:-$(pwd):/home/dev/proj}"
readonly port="${3:-10001:11001}"
readonly env="${4:-/home/dev}"
#
#
sudo docker run -it \
                --rm \
                --privileged \
                --runtime nvidia \
                --env DISPLAY=unix$DISPLAY \
                --ipc=host \
                --net=host \
                --volume /tmp/.X11-unix:/tmp/.X11-unix:ro \
                --volume ${HOME}/.cache/torch:/root/.cache/torch \
                --volume ${HOME}/.cache/huggingface:/root/.cache/huggingface \
                --volume /home/dev/NAS:/home/dev/NAS \
                --volume ${volume} \
                --publish ${port} \
                --env ${env} \
                ${repo_tag} /bin/bash
