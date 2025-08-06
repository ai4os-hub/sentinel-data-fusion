import tarfile
import zipfile
import io
import os
import datetime
import numpy as np
import cv2


def safe_extract_tar(tar: tarfile.TarFile, path: str = "."):
    members = tar.getmembers()
    for member in members:
        member_path = os.path.join(path, member.name)
        if not os.path.abspath(member_path).startswith(os.path.abspath(path) + os.sep):
            raise Exception(f"Path traversal detected in tar file: {member.name}")
    for member in members:
        tar.extract(member, path)


def safe_extract_zip(zf: zipfile.ZipFile, path: str = "."):
    names = zf.namelist()
    for name in names:
        member_path = os.path.join(path, name)
        if not os.path.abspath(member_path).startswith(os.path.abspath(path) + os.sep):
            raise Exception(f"Path traversal detected in zip file: {name}")
    for name in names:
        zf.extract(name, path)


def open_compressed(byte_stream, file_format, output_folder, file_path=None):
    """
    Extract and save a stream of bytes of a compressed file from memory.
    Parameters
    ----------
    byte_stream : BinaryIO
    file_format : str
        Compatible file formats: tarballs, zip files
    output_folder : str
        Folder to extract the stream
    Returns
    -------
    Folder name of the extracted files.
    """

    tar_extensions = [
        "tar",
        "bz2",
        "tb2",
        "tbz",
        "tbz2",
        "gz",
        "tgz",
        "lz",
        "lzma",
        "tlz",
        "xz",
        "txz",
        "Z",
        "tZ",
    ]

    if file_format in tar_extensions:
        with open(file_path, "wb") as f:
            f.write(byte_stream)
        tar = tarfile.open(file_path)
        safe_extract_tar(tar, output_folder)
        tar.close()
        os.remove(file_path)

    elif file_format == "zip":
        zf = zipfile.ZipFile(io.BytesIO(byte_stream))
        safe_extract_zip(zf, output_folder)
        zf.close()
    else:
        raise ValueError("Invalid file format for the compressed byte_stream")


def get_date(tile, satellite):

    if satellite == "SENTINEL-2":
        date = ((tile.split("_"))[2])[:8]
        date = datetime.datetime.strptime(date, "%Y%m%d")

    elif satellite == "SENTINEL-3":
        date = ((tile.split("_"))[7])[:8]
        date = datetime.datetime.strptime(date, "%Y%m%d")

    return date.strftime("%Y-%m-%d")


def data_resize(data_bands):

    max_res = np.amin(list(data_bands.keys()))
    m, n = data_bands[max_res].shape[:2]

    rs_bands = {}

    resolutions = [res for res in list(data_bands.keys()) if res != max_res]
    for res in resolutions:
        arr_bands = np.zeros((m, n, data_bands[res].shape[-1]))
        for i in range(data_bands[res].shape[-1]):
            arr_bands[:, :, i] = cv2.resize(
                data_bands[res][:, :, i], (n, m), interpolation=cv2.INTER_CUBIC
            )
        rs_bands[res] = arr_bands

    return rs_bands
