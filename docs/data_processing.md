# Data processing

To process the data after capturing the data needs to be stored in a specific layout in the `source_data` directory.

1. Create a new folder for each captured object

2. Inside the object folder create a folder for each environment and name them `train`, `valid` or `test`.
   Inside the environment folders create the following folder structure.

   | Folder | Description |
   | --- | --- |
   | `images`  | This directory stores the images that sample the hemisphere around the object taken with the DSLR camera.
   | `env`    | This folder contains the exposure sequence captured with the 360 camera that corresponds to the environment under which the images in `images` were taken.
   | `test1` | This contains the exposure sequence for a test image taken with the DSLR on a tripod. There should be exactly 7 images in this folder. The center image will be used for COLMAP. |
   | `test1_env` | Stores the exposure sequence taken with the 360 camera that corresponds to the environment for `test1`. |
   | `test2` | See `test1` |
   | `test2_env` | See `test1_env` |
   | `test3` | See `test1` |
   | `test3_env` | See `test1_env` |

3. (Optional) Create a mask image for environments captured with an ND-filter.
   The mask image is a file with name `nd-filter_images_mask.png` in the same folder as the dual fish eye images.
   Create a file `nd-filter` with the format `f-stop Rxxxxx1-Rxxxxx2`. `f-stop` is the f-stop reduction of the nd filter used.
   The value for the filter is 17.091 throughout the dataset.
   `Rxxxxx1-Rxxxxx2` is the consecutive range of images for which the ND-filter was applied. Example: `9 R0011134-R0011137`.
   

4. Create the exposure.txt.

5. Annotate object keypoints.

6. Create the intermediate files for reconstruction.
   ```bash
   # inside the script directory
   python create_intermediate_files.py ../source_data/object ../intermediate_data
   ```

7. (Optional. This step has been replaced with our custom stitching code.) 
   Run the stitcher.
   We use the RICOH Theta Stitcher with wine.
   To stitch all images in a batch run the `stitch_cmd.sh` files inside each environment subdir.
   ```bash
   # inside intermediate_data/object/{train,valid,test}
   source stitch_cmd.sh
   ```
   The command opens the gui for the stitcher. Set a fixed distance of 1 meter and keep all other settings.

8. Run the reconstruction with colmap.

9. Define a coarse object bounding box.
   Use `bounding_box_tool.py` to create the `object_bounding_box.txt`.
   ```bash
   # inside the script folder
   python bounding_box_tool.py ../intermediate_data/object/train
   ```
   The `object_bounding_box.txt` is created in the corresponding dir in the `source_data` folder and can usually be copied afterwards to the other environments since we try to place the object on the marker board roughly with the same orientation and position.
   
10. Compute transformations between the environments.

11. Create the dataset files.