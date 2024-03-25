# Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

# This script is based on an original implementation by True Price.
# Get and update functions added by Benjamin Ummenhofer

import sys
import sqlite3
import numpy as np
from enum import IntEnum

class TwoViewConfig(IntEnum):
    UNDEFINED=0
    DEGENERATE=1
    CALIBRATED=2          # Essential matrix
    UNCALIBRATED=3        # Fundamental matrix
    PLANAR=4              # Homography, planar scene with baseline
    PANORAMIC=5           # Homography, pure rotation without baseline
    PLANAR_OR_PANORAMIC=6 # Homography, planar or panoramic
    WATERMARK=7           # Pure 2D translation in image borders
    MULTIPLE=8            # Multi-model configuration, i.e. the inlier matches result from multiple
                          # individual, non-degenerate configurations.  

class CameraModel(IntEnum):
    SIMPLE_PINHOLE=0
    PINHOLE=1
    SIMPLE_RADIAL=2
    RADIAL=3
    OPENCV=4
    OPENCV_FISHEYE=5
    FULL_OPENCV=6
    FOV=7
    SIMPLE_RADIAL_FISHEYE=8
    RADIAL_FISHEYE=9
    THIN_PRISM_FISHEYE=10


IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def array_to_blob_or_none(array):
    if array is None:
        return None
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def get_cameras(self):
        ans = list(self.execute(
            "SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras"))
        result = {}
        for row in ans:
            camera_id = row[0]
            values = {}
            values['model'] = row[1]
            values['width'] = row[2]
            values['height'] = row[3]
            values['params'] = blob_to_array(row[4], np.float64)
            values['prior_focal_length'] = row[5]
            result[camera_id] = values
        return result

    def update_camera(self, camera_id, model, width, height, params, prior_focal_length=False):
        self.execute(
            f"UPDATE cameras SET model={model}, width={width}, height={height}, params=?, prior_focal_length={prior_focal_length} WHERE camera_id={camera_id}", (array_to_blob(params),))

    def add_image(self, name, camera_id,
                  prior_q=np.full(4, np.NaN), prior_t=np.full(3, np.NaN), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def get_images(self):
        ans = list(self.execute(
            "SELECT image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz FROM images"))
        result = {}
        for row in ans:
            image_id = row[0]
            values = {}
            values['name'] = row[1] 
            values['camera_id'] = row[2]
            values['prior_q'] = None if row[3] is None else np.array(row[3:6])
            values['prior_t'] = None if row[6] is None else np.array(row[6:9])
            result[image_id] = values
        return result
    
    def update_image(self, image_id, name, camera_id, prior_q=None, prior_t=None):
        prior_q = np.full(4, np.NaN) if prior_q is None else prior_q
        prior_t = np.full(3, np.NaN) if prior_t is None else prior_t
        self.execute(
            f"UPDATE images SET name=?, camera_id=?, prior_qw=?, prior_qx=?, prior_qy=?, prior_qz=?, prior_tx=?, prior_ty=?, prior_tz=? WHERE image_id={image_id}",
            (name, camera_id, prior_q[0], prior_q[1], prior_q[2], prior_q[3], prior_t[0], prior_t[1], prior_t[2]))

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def get_keypoints(self, image_id):
        ans = list(self.execute(
            f"SELECT cols, data FROM keypoints WHERE image_id={image_id}"))
        keypoints = None
        if ans:
            cols, data = ans[0]
            if data is None:
                return np.empty((0,cols), dtype=np.float32)
            keypoints = blob_to_array(data, np.float32, (-1,cols))
        return keypoints

    def update_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            f"UPDATE keypoints SET rows={keypoints.shape[0]}, cols={keypoints.shape[1]}, data=? WHERE image_id={image_id}", (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def get_descriptors(self, image_id):
        ans = list(self.execute(
            "SELECT cols, data FROM descriptors WHERE image_id={}".format(image_id)))
        descriptors = None
        if ans:
            cols, data = ans[0]
            if data is None:
                return np.empty((0,cols), dtype=np.uint8)
            descriptors = blob_to_array(data, np.uint8, (-1,cols))
        return descriptors

    def update_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            f"UPDATE descriptors SET rows={descriptors.shape[0]}, cols={descriptors.shape[1]}, data=? WHERE image_id={image_id}", (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))
    
    def get_matches(self, image_id1, image_id2):
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        ans = list(self.execute(
            f"SELECT cols, data FROM matches WHERE pair_id={pair_id}"))
        matches = None
        if ans:
            cols, data = ans[0]
            if data is None:
                return np.empty((0,cols), dtype=np.uint32)
            matches = blob_to_array(data, np.uint32, (-1,cols))
        if image_id1 > image_id2 and matches is not None:
            matches = matches[:,::-1]
        return matches

    def update_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            f"UPDATE matches SET rows={matches.shape[0]}, cols={matches.shape[1]}, data=? WHERE pair_id={pair_id}",
            (array_to_blob(matches),))


    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3),
                              qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                              tvec=np.zeros(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H),
             array_to_blob(qvec), array_to_blob(tvec)))

    def get_two_view_geometry(self, image_id1, image_id2):
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        
        ans = list(self.execute(
            f"SELECT cols, data, config, F, E, H, qvec, tvec FROM two_view_geometries WHERE pair_id={pair_id}"))
        result = {}
        if ans:
            cols, data, config, F, E, H, qvec, tvec = ans[0]
            result['matches'] = None if data is None else blob_to_array(data, np.uint32, (-1,cols))
            result['config'] = config
            result['F'] = None if F is None else blob_to_array(F, np.float64, (3,3))
            result['E'] = None if E is None else  blob_to_array(E, np.float64, (3,3))
            result['H'] = None if H is None else  blob_to_array(H, np.float64, (3,3))
            result['qvec'] = None if qvec is None else  blob_to_array(H, np.float64, )
            result['tvec'] = None if tvec is None else blob_to_array(H, np.float64, )
        return result

    def update_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3),
                              qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                              tvec=np.zeros(3), config=2):
        if matches is not None:
            assert(len(matches.shape) == 2)
            assert(matches.shape[1] == 2)
            matches_shape = matches.shape
        else:
            matches_shape = (0,2)

        if matches is not None and image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = None if matches is None else np.asarray(matches, np.uint32)
        F = None if F is None else np.asarray(F, dtype=np.float64)
        E = None if E is None else np.asarray(E, dtype=np.float64)
        H = None if H is None else np.asarray(H, dtype=np.float64)
        qvec = None if qvec is None else np.asarray(qvec, dtype=np.float64)
        tvec = None if tvec is None else np.asarray(tvec, dtype=np.float64)
        self.execute(
            f"UPDATE two_view_geometries SET rows={matches_shape[0]}, cols={matches_shape[1]}, data=?, config={int(config)}, F=?, E=?, H=?, qvec=?, tvec=? WHERE pair_id={pair_id}", 
             (array_to_blob_or_none(matches), array_to_blob_or_none(F), array_to_blob_or_none(E), array_to_blob_or_none(H),
             array_to_blob_or_none(qvec), array_to_blob_or_none(tvec)))


def example_usage():
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()

    if os.path.exists(args.database_path):
        print("ERROR: database path already exists -- will not modify it.")
        return

    # Open the database.

    db = COLMAPDatabase.connect(args.database_path)

    # For convenience, try creating all the tables upfront.

    db.create_tables()

    # Create dummy cameras.

    model1, width1, height1, params1 = \
        0, 1024, 768, np.array((1024., 512., 384.))
    model2, width2, height2, params2 = \
        2, 1024, 768, np.array((1024., 512., 384., 0.1))

    camera_id1 = db.add_camera(model1, width1, height1, params1)
    camera_id2 = db.add_camera(model2, width2, height2, params2)

    # Create dummy images.

    image_id1 = db.add_image("image1.png", camera_id1)
    image_id2 = db.add_image("image2.png", camera_id1)
    image_id3 = db.add_image("image3.png", camera_id2)
    image_id4 = db.add_image("image4.png", camera_id2)

    # Create dummy keypoints.
    #
    # Note that COLMAP supports:
    #      - 2D keypoints: (x, y)
    #      - 4D keypoints: (x, y, theta, scale)
    #      - 6D affine keypoints: (x, y, a_11, a_12, a_21, a_22)

    num_keypoints = 1000
    keypoints1 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints2 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints3 = np.random.rand(num_keypoints, 2) * (width2, height2)
    keypoints4 = np.random.rand(num_keypoints, 2) * (width2, height2)

    db.add_keypoints(image_id1, keypoints1)
    db.add_keypoints(image_id2, keypoints2)
    db.add_keypoints(image_id3, keypoints3)
    db.add_keypoints(image_id4, keypoints4)

    # Create dummy matches.

    M = 50
    matches12 = np.random.randint(num_keypoints, size=(M, 2))
    matches23 = np.random.randint(num_keypoints, size=(M, 2))
    matches34 = np.random.randint(num_keypoints, size=(M, 2))

    db.add_matches(image_id1, image_id2, matches12)
    db.add_matches(image_id2, image_id3, matches23)
    db.add_matches(image_id3, image_id4, matches34)

    # Commit the data to the file.

    db.commit()

    # Read and check cameras.

    rows = db.execute("SELECT * FROM cameras")

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id1
    assert model == model1 and width == width1 and height == height1
    assert np.allclose(params, params1)

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id2
    assert model == model2 and width == width2 and height == height2
    assert np.allclose(params, params2)

    # Read and check keypoints.

    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM keypoints"))

    assert np.allclose(keypoints[image_id1], keypoints1)
    assert np.allclose(keypoints[image_id2], keypoints2)
    assert np.allclose(keypoints[image_id3], keypoints3)
    assert np.allclose(keypoints[image_id4], keypoints4)

    # Read and check matches.

    pair_ids = [image_ids_to_pair_id(*pair) for pair in
                ((image_id1, image_id2),
                 (image_id2, image_id3),
                 (image_id3, image_id4))]

    matches = dict(
        (pair_id_to_image_ids(pair_id),
         blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
    )

    assert np.all(matches[(image_id1, image_id2)] == matches12)
    assert np.all(matches[(image_id2, image_id3)] == matches23)
    assert np.all(matches[(image_id3, image_id4)] == matches34)

    # Clean up.

    db.close()

    if os.path.exists(args.database_path):
        os.remove(args.database_path)


if __name__ == "__main__":
    example_usage()
