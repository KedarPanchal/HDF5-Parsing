# Setting up this code on a non-Linux machine
- Make sure this git repository (and none of its subdirectories) is the current working directory
- Run `bash docker/run_docker.bash` (or `chmod +x` it first to make it executable) to build the docker image (if not already done) and start up a new container
  - This process takes a while--especially if you're building the image for the first time
  - If you don't have patience be sure not to delete/make a new docker container if you already have one for this image and instead just start a bash shell in an existing container as `catkin_make` take forever to run
  - Wait even after you enter the docker container, as it runs `roscore &` automatically
  - You can pass a command-line argument to the script (e.g. `bash docker/run_docker.bash mycontainer`) to give the container created a custom name
