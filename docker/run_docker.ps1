# Powershell script because I'm a special snowflake
param(
    [string]$cname=$null
)

docker build -t "hdf5_parse" docker/

if  (!$cname) {
    docker run -v "${PSCommandPath}/:/usr/share/hdf5_parse" --name $cname -it "hdf5_parse"
} else {
    docker run -v "${PSCommandPath}/:/usr/share/hdf5_parse" -it "hdf5_parse"
}