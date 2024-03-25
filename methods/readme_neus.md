# Running NeuS on the Objects With Lighting dataset

This describes how to generate the meshes for the objects in the dataset with NeuS.
Note that the dataset includes the meshes and running NeuS is optional.

## Prerequisites

These instructions assume a working installation of NeuS next to this repo in the same parent directory

## Data conversion

Use the `scripts/create_neus_data.py` script to convert a scene to a format that can be used with NeuS.

```bash
python scripts/create_neus_data.py dataset/OBJECT/test neus_data
```

## Running NeuS

To start the reconstruction process use the script `scripts/reconstruct_with_neus.sh`.

```bash
./scripts/reconstruct_with_neus.sh --input neus_data
```

The output mesh in this example will be written to `neus_data/exp/womask_sphere/meshes/00300000.ply`.

