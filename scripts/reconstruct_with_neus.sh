#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

function printhelp {
        echo "Usage: $(basename $0) [OPTIONS]"
        echo "Options:"
        echo "  --input    Path to the neus_data directory"
        exit 0
}

if [[ $# -eq 0 ]] ; then
    printhelp
    exit 0
fi

parse_iter=0
while [ $parse_iter -lt 100 ] ; do
        parse_iter=$((parse_iter+1))
        case "$1" in
                --input) neus_data_dir=$(realpath "$2") ; shift 2 ;;
                --help | -h) printhelp ; shift  ;;
                *) break ;;
        esac
done

if ! [[ -d "$neus_data_dir" ]]; then
        echo "$neus_data_dir is not a directory"
        exit 1
fi
echo "input dir is $neus_data_dir"

repo_root=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )"/.. &> /dev/null && pwd )

# assume NeuS is in the same parent directory
pushd "$repo_root"/../NeuS/
if ! [[ -f "$neus_data_dir/exp/womask_sphere/meshes/00300000.ply" ]]; then
        options=""
        if [[ -d "$neus_data_dir/exp/womask_sphere/checkpoints" ]]; then
                options="--is_continue"
        fi
        config_path=(${neus_data_dir}/w*mask.conf)
        config_path=${config_path[0]}
        python exp_runner.py --mode train --conf $config_path --case "dummy" $options
        if [[ -f "$neus_data_dir/exp/womask_sphere/meshes/00300000.ply" ]]; then
                python exp_runner.py --mode validate_mesh --conf $config_path --case "dummy" --is_continue 
        fi
fi
popd
