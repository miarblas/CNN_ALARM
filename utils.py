import h5py
import numpy as np

def _decode_array(x):
    """Robust decoding for arrays that may be bytes, fixed-width strings, or objects."""
    arr = np.array(x)
    if arr.dtype.kind in {"S", "O"}:
        def _to_str(v):
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="ignore")
            return str(v)
        return np.vectorize(_to_str)(arr).astype(str)
    return arr.astype(str)

def loadh5(filepath):
    """
    Reads an H5 dataset file (trainSet.h5 / testSet.h5).

    Expected keys:
      - fs
      - alarm
      - status (may be missing in hidden test set)
      - channels
      - waveform
    """
    data = {}
    with h5py.File(filepath, "r") as dataset:
        if "fs" in dataset:
            data["fs"] = dataset["fs"][:]
        if "waveform" in dataset:
            data["waveform"] = dataset["waveform"][:]

        if "alarm" in dataset:
            data["alarm"] = _decode_array(dataset["alarm"][:])
        if "status" in dataset:
            data["status"] = _decode_array(dataset["status"][:])
        if "channels" in dataset:
            data["channels"] = _decode_array(dataset["channels"][:])

    return data

if __name__ == "__main__":
    import os

    # Change this to the path of your H5 file
    h5_path = os.path.join("data", "trainSet.h5")

    # Load the data
    data = loadh5(h5_path)

    print("Keys in H5 file:", list(data.keys()))

    # Print shapes
    if "waveform" in data:
        print("waveform shape:", data["waveform"].shape)
        # print a small snippet
        print("waveform sample [0]:", data["waveform"][0, :5, :])  # first 5 samples, all channels

    if "fs" in data:
        print("fs shape:", np.shape(data["fs"]))
        print("fs sample:", data["fs"][:5])

    if "status" in data:
        print("status shape:", np.shape(data["status"]))
        print("status sample:", data["status"][:5])

    if "channels" in data:
        print("channels shape:", np.shape(data["channels"]))
        print("channels sample:", data["channels"][:5])

