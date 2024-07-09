# Setting up this code on a non-Linux machine
- Make sure this git repository (and none of its subdirectories) is the current working directory
- Run `bash docker/run_docker.bash` (or `chmod +x` it first to make it executable) to build the docker image (if not already done) and start up a new container
  - This process takes a while--especially if you're building the image for the first time
  - Wait even after you enter the docker container, as it runs `roscore &` automatically
  - You can pass a command-line argument to the script (e.g. `bash docker/run_docker.bash mycontainer`) to give the container created a custom name
- Once you enter the specific container instance for the first time run `bash setup_catkin.bash` (or `chmod +x` it first to make it executable) to import the `tf_bag` library