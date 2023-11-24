# -*- coding: utf-8 -*-
"""
This sub-package holds classes and functions to define (microscope) systems.

Classes
-------
 - `dukit.shared.systems.System`
 - `dukit.shared.systems.MelbSystem`
 - `dukit.shared.systems.LVControl`
 - `dukit.shared.systems.PyControl`
 - `dukit.shared.systems.Zyla`
 - `dukit.shared.systems.CryoWidefield`
 - `dukit.shared.systems.LegacyCryoWidefield`
 - `dukit.shared.systems.Argus`
 - `dukit.shared.systems.LegacyArgus`
 - `dukit.shared.systems.PyCryoWidefield`

"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.shared.systems.System": True,
    "dukit.shared.systems.MelbSystem": True,
    "dukit.shared.systems.LVControl": True,
    "dukit.shared.systems.PyControl": True,
    "dukit.shared.systems.Zyla": True,
    "dukit.shared.systems.CryoWidefield": True,
    "dukit.shared.systems.LegacyCryoWidefield": True,
    "dukit.shared.systems.Argus": True,
    "dukit.shared.systems.LegacyArgus": True,
    "dukit.shared.systems.PyCryoWidefield": True,
}

# ============================================================================

import re
from math import radians
import numpy as np
import numpy.typing as npt

# ============================================================================

from dukit.shared.json2dict import json_to_dict
from dukit.shared.misc import dukit_warn

# ============================================================================


class System:
    """Abstract class defining what is expected for a system."""

    name: str = "Unknown System"
    """Name of the system."""

    _pixel_size: float | None = None
    """Specified pixel size (m) (float). 
    If None then obj mag etc. attributes are used to calc. pixel size instead.
    Either _pixel_size or those attributes must be set!
    """

    _sensor_pixel_pitch: float = -1.0
    """Pixel size (m) at camera
    """

    _obj_mag: float = -1.0
    """Magnification of objective lens (in general a float).
    """

    _obj_ref_focal_length: float = -1.0
    """Reference focal length for objective manufacturer (m). (float).
    200e-3 for Nikon or Leica.
    For Olympus use 180e-3, for Zeiss 165e-3.
    """

    _camera_tube_lens: float = -1.0
    """Tube lens that focuses collimated beam from sample onto the camera.
    A float length in metres.
    """

    _bias_mag: float | None = None
    """Bias magnetic field strength (T)."""
    _bias_theta: float | None = None
    """Bias magnetic field polar angle (deg)."""
    _bias_phi: float | None = None
    """Bias magnetic field azimuthal angle (deg)."""

    def __init__(
        self,
        pixel_size: float | None = None,
        sensor_pixel_pitch: float | None = None,
        obj_mag: float | None = None,
        obj_ref_focal_length: float | None = None,
        camera_tube_lens: float | None = None,
        bias_mag: float | None = None,
        bias_theta: float | None = None,
        bias_phi: float | None = None,
    ):
        """
        Initialize the `System`, optionally supply known bias field and override
        the `System`-defined microscope distances/etc.

        Parameters
        ----------
        pixel_size: float | None, default=None
            Define pixel size manually. If set, below 4 params are disregarded.
            Either pixel_size or the below 4 must be defined either in init
            or hard-coded into System (sub-)class. Given in metres.
        sensor_pixel_pitch: float | None, default=None
            Effective pixel 'size' at camera sensor. Given in metres.
        obj_mag: float | None, default=None
            Magnification of objective used. Usually you will supply an int.
        obj_ref_focal_length: float | None, default=None
            Reference focal length from objective manufacturer (in metres).
            200e-3 for Nikon or Leica. For Olympus use 180e-3, for Zeiss 165e-3.
        camera_tube_lens: float | None, default=None
            Tube lens length (m) used to focus light onto camera.
        bias_mag: float | None, default=None
            Magnitude of bias field (if known, else None) in Teslas.
        bias_theta: float | None, default=None
            Polar angle (deg) of bias field.
        bias_phi: float | None, default=None
            Azimuthal angle (deg) of bias field.
        """
        for att, param in zip(
            [
                "_pixel_size",
                "_sensor_pixel_pitch",
                "_obj_mag",
                "_obj_ref_focal_length",
                "_camera_tube_lens",
                "_bias_mag",
                "_bias_theta",
                "_bias_phi",
            ],
            [
                pixel_size,
                sensor_pixel_pitch,
                obj_mag,
                obj_ref_focal_length,
                camera_tube_lens,
                bias_mag,
                bias_theta,
                bias_phi,
            ],
        ):
            if param is not None:
                setattr(self, att, param)
            # check we have sufficient info to calculate pixel size.
            if self._pixel_size is not None and np.all(
                [
                    i < 0
                    for i in [
                        self._sensor_pixel_pitch,
                        self._obj_mag,
                        self._obj_ref_focal_length,
                        self._camera_tube_lens,
                    ]
                ]
            ):
                raise ValueError(
                    "System must have a pixel_size defined, or all of: "
                    + "sensor_pixel_pitch, obj_mag, obj_ref_focal_length, "
                    + "camera_tube_lens."
                )

    def read_image(
        self, filepath: str, ignore_ref: bool, norm: str = "div"
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """
        Method that must be defined to read raw data in from filepath.

        Parameters
        ----------
        filepath : str or Pathlib etc. object
            Path to measurement file
        ignore_ref : bool
            Ignore any reference measurements. (i.e. no-RF lock-in)
        norm : str
            Normalisation method. "div", "sub" or "true_sub". (latter for T1 datasets)

        Returns
        -------
        sig : np array, 3D
            Format: [sweep values, y, x]. Not cropped etc.
        ref : np array, 3D
            Format: [sweep values, y, x]. Not cropped etc.
        sig_norm : np array, 3D
            Format: [sweep values, y, x]. Not cropped etc.

        Notes
        -----
        if norm == "sub":
            sig_norm = 1 + (sig - ref) / (sig + ref)
        elif norm == "div":
            sig_norm = sig / ref
        elif norm == "true_sub":
            sig_norm = (sig - ref) / np.nanmax(sig - ref, axis=0)
        """
        raise NotImplementedError

    def get_hardware_binning(self, filepath: str) -> int:
        """
        Method that must be defined to define the camera binning from metadata

        Arguments
        ---------
        filepath : str or Pathlib etc. object
            Path to measurement file
        """
        raise NotImplementedError

    def read_sweep_arr(self, filepath: str) -> npt.NDArray[np.float64]:
        """
        Method that must be defined to read sweep_arr in from filepath.

        Arguments
        ---------
        filepath : str or Pathlib etc. object
            Path to measurement file

        Returns
        -------
        sweep_arr : np array, 1D
            List of sweep value, either freq (MHz) or taus (s).
        """
        raise NotImplementedError

    def get_raw_pixel_size(self, filepath: str) -> float:
        """
        Get raw (from camera, without additional binning) pixel size.

        Arguments
        ---------
        filepath : str or Pathlib etc. object
            Path to measurement file
        """

        hardware_binning = self.get_hardware_binning(filepath)

        if self._pixel_size is not None:
            return hardware_binning * self._pixel_size

        f_obj = self._obj_ref_focal_length / self._obj_mag

        camera_pixel_size = self._sensor_pixel_pitch * f_obj / self._camera_tube_lens

        return hardware_binning * camera_pixel_size

    def get_bias_field(
        self, filepath: str
    ) -> tuple[bool, tuple[float | None, float | None, float | None]]:
        """
        Method to get magnet bias field from experiment metadata,
        i.e. if set with programmable electromagnet. Default: False, (None, None, None).

        Arguments
        ---------
        filepath : str or Pathlib etc. object
            Path to measurement file

        Returns
        -------
        bias_on : bool
            Was programmable bias field used?
        bias_field : tuple
            Tuple representing vector bias field
                (B_mag (Tesla), B_theta (rad), B_phi (rad))
        """
        return False, (
            self._bias_mag,
            radians(self._bias_theta) if self._bias_theta is not None else None,
            radians(self._bias_phi) if self._bias_phi is not None else None,
        )


# ============================================================================


class MelbSystem(System):
    """Some shared methods for melbourne systems."""

    name = "Unknown Melbourne System"

    @staticmethod
    def _chop_into_sig_ref(
        img: npt.NDArray[np.float64], used_ref: bool, norm: str
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        if norm not in ["div", "sub", "true_sub"]:
            raise ValueError("bad norm option, use one of ['sub', 'div', 'true_sub']")
        # now chop up into sig, ref & normalise
        if used_ref:
            sig = img[::2, :, :]
            ref = img[1::2, :, :]
            if norm == "sub":
                sig_norm = 1 + (sig - ref) / (sig + ref)
            elif norm == "div":
                sig_norm = sig / ref
            else:
                sig_norm = (sig - ref) / np.nanmax(sig - ref, axis=0)
        else:
            sig = img
            ref = img
            sig_norm = sig / np.nanmax(sig, axis=0)
        return sig, ref, sig_norm


class LVControl(MelbSystem):
    """Older Labview-based control software, save formats etc."""

    name = "Unknown Labview controlled System"

    # TODO test this one
    def read_image(
        self, filepath: str, ignore_ref: bool = False, norm: str = "div"
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        # read from disk
        with open(filepath, mode="r", encoding="utf-8") as fid:
            ret: npt.NDArray[np.float32] = np.fromfile(fid, dtype=np.float32)
            raw_data: npt.NDArray[np.float32] = ret[2:]
            prod: int = int(np.prod(ret[:2]))
            sweep_len: int = int(np.prod(raw_data.shape) // prod)

        # reshape into something more useable
        img, used_ref = self._reshape_raw(raw_data, sweep_len, filepath, ignore_ref)

        # now chop up into sig, ref & normalise
        return self._chop_into_sig_ref(img, used_ref, norm)

    def read_sweep_arr(self, filepath: str) -> npt.NDArray[np.float64]:
        with open(
            filepath + "_metaSpool.txt",
            "r",
            encoding="utf-8",
        ) as fid:
            sweep_str = fid.readline().rstrip().split("\t")
        return np.array([float(i) for i in sweep_str], dtype=np.float64)

    def get_hardware_binning(self, filepath: str) -> int:
        metadata = self._read_metadata(filepath)
        return int(metadata["Binning"])

    def _read_metadata(self, filepath: str) -> dict:
        """
        Reads metaspool text file into a metadata dictionary.
        Filepath argument is the filepath of the (binary) dataset.
        """
        with open(filepath + "_metaSpool.txt", "r", encoding="utf-8") as fid:
            # skip the sweep list (freqs or taus)
            _ = fid.readline().rstrip().split("\t")
            # ok now read the metadata
            rest_str = fid.read()
            # any text except newlines and tabs, followed by a colon and a space
            # then any text except newlines and tabs
            # (match the two text regions)
            matches = re.findall(
                # r"^([a-zA-Z0-9_ /+()#-]+):([a-zA-Z0-9_ /+()#-]+)",
                r"([^\t\n]+):\s([^\t\n]+)",
                rest_str,
                re.MULTILINE,
            )

            def fail_float(a):
                try:
                    return float(a)
                except ValueError:
                    if a == "FALSE":
                        return False
                    if a == "TRUE":
                        return True
                    return a

            metadata = {a: fail_float(b) for (a, b) in matches}
        return metadata

    def _reshape_raw(
        self,
        raw_data: npt.NDArray[np.float32],
        sweep_len: int,
        filepath: str,
        ignore_ref: bool,
    ) -> tuple[npt.NDArray[np.float64], bool]:
        """
        Reshapes raw data into more useful shape, according to image size in metadata.
        Unimelb-specific data reshaping procedure (relies upon metadata)

        Arguments
        ---------
        raw_data : np array, 1D (unshaped)
            Raw unshaped data read from binary file
        sweep_len : int
            Length of sweep list
        filepath : str or Pathlib etc. object
            Path to measurement file
        ignore_ref : bool
            Ignore any reference measurements. (i.e. no-RF lock-in)

        Returns
        -------
        image : np array, 3D
            Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and
            has not been rebinned. Unwanted sweep values not removed.
        used_ref : bool
            Was a reference (e.g. no-MW measurement) loaded?
        """

        metadata = self._read_metadata(filepath)

        # NOTE AOIHeight and AOIWidth are saved by labview the opposite
        #  of what you'd expect
        # -> LV rotates to give image as you'd expect standing in lab
        # -> thus, we ensure all images in this processing code matches LV orientation
        try:
            if not ignore_ref:
                image = np.reshape(
                    raw_data,
                    [
                        sweep_len,
                        int(metadata["AOIHeight"]),
                        int(metadata["AOIWidth"]),
                    ],
                )
                used_ref = False  # if we succeed here there's no ref
            else:
                raise ValueError
        except ValueError:
            # if the ref is used then there's 2* the number of sweeps
            # i.e. auto-detect reference existence
            if ignore_ref:
                used_ref = False
                image = np.reshape(
                    raw_data,
                    [
                        2 * sweep_len,
                        int(metadata["AOIHeight"]),
                        int(metadata["AOIWidth"]),
                    ],
                )[
                    ::2
                ]  # hmmm disregard ref -> use every second element.
            else:
                image = np.reshape(
                    raw_data,
                    [
                        sweep_len,
                        int(metadata["AOIHeight"]),
                        int(metadata["AOIWidth"]),
                    ],
                )
                used_ref = True
        # Transpose the dataset to get the correct x and y orientations ([y, x])
        # will work for non-square images
        return image.transpose([0, 2, 1]).astype(np.float64), used_ref

    def get_bias_field(
        self, filepath: str
    ) -> tuple[bool, tuple[float | None, float | None, float | None]]:
        """get bias on (bool) and field as tuple (mag (G), theta (rad), phi (rad))"""
        metadata = self._read_metadata(filepath)

        key_ars = [
            ["Field Strength (G)"],
            ["Theta (deg)"],
            ["Phi (def)", "Phi (deg)"],
        ]
        bias_field = []
        for arr in key_ars:
            found = False
            for key in arr:
                if key in metadata:
                    bias_field.append(metadata[key])
                    found = True
            if not found:
                return False, (None, None, None)
        if len(bias_field) != 3:
            dukit_warn(
                f"Found {len(bias_field)} bias field params in metadata, "
                + "this shouldn't happen (expected 3)."
            )
            return False, (None, None, None)
        onoff_str = metadata.get("Mag on/off", "")
        bias_on = onoff_str == " TRUE"
        return bias_on, (
            bias_field[0] * 1e-4,  # convert G to T
            radians(bias_field[1]),
            radians(bias_field[2]),
        )


# ============================================================================


class PyControl(MelbSystem):
    """Newer systems, using python control software -> new API"""

    name = "Unknown Python controlled System"

    def read_image(
        self, filepath: str, ignore_ref: bool = False, norm: str = "div"
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        if norm not in ["div", "sub", "true_sub"]:
            raise ValueError("bad norm option, use one of ['sub', 'div', 'true_sub']")

        image = np.load(filepath + ".npy")
        if ignore_ref and self._read_metadata(filepath)["Measurement"]["ref_bool"]:
            return self._chop_into_sig_ref(image[::2], False, norm)
        if not self._read_metadata(filepath)["Measurement"]["ref_bool"]:
            return self._chop_into_sig_ref(image, False, norm)
        return self._chop_into_sig_ref(image, True, norm)

    def read_sweep_arr(self, filepath: str) -> npt.NDArray[np.float64]:
        return np.ndarray(
            json_to_dict(filepath + ".json")["freq_list"],
            dtype=np.float64,
        )  # TODO name won't work for tau sweeps

    def get_hardware_binning(self, filepath: str) -> int:
        metadata = self._read_metadata(filepath)

        binning = metadata["Devices"]["camera"]["bin"]
        if binning[0] != binning[1]:
            raise ValueError("dukit not setup to handle anisotropic binning.")
        return int(binning[0])

    def _read_metadata(self, filepath: str):
        """
        Reads metaspool text file into a metadata dictionary.
        Filepath argument is the filepath of the (binary) dataset.
        """
        return json_to_dict(filepath + "_metadata.json", hook="dd")

    def get_bias_field(
        self, filepath: str
    ) -> tuple[bool, tuple[float | None, float | None, float | None]]:
        """get bias on (bool) and field as tuple (mag (G), theta (rad), phi (rad))"""
        metadata = self._read_metadata(filepath)
        keys = [
            "bnorm",  # update with units?
            "theta",
            "phi",
        ]
        bias_field = []
        for key in keys:
            if key in metadata["Devices"]["null"]:
                bias_field.append(metadata["Devices"]["null"][key])
            else:
                return False, (None, None, None)
        if len(bias_field) != 3:
            dukit_warn(
                f"Found {len(bias_field)} bias field params in metadata, "
                + "this shouldn't happen (expected 3)."
            )
            return False, (None, None, None)

        return True, (
            bias_field[0] / 1000,  # convert mT to T
            radians(bias_field[1]),
            radians(bias_field[2]),
        )


# ============================================================================


class Zyla(LVControl):
    """
    Specific system details for the Zyla QDM.
    """

    name = "Zyla"
    _obj_mag = 4
    _obj_ref_focal_length = 200e-3
    _camera_tube_lens = 300e-3
    _sensor_pixel_pitch = 6.5e-6


class CryoWidefield(LVControl):
    """
    Specific system details for Cryogenic (Attocube) widefield QDM.
    """

    name = "Cryo Widefield"
    _pixel_size = 59.6e-9


class LegacyCryoWidefield(LVControl):
    """
    Specific system details for cryogenic (Attocube) widefield QDM.
    - Legacy binning version
    """

    name = "Legacy Cryo Widefield"
    _pixel_size = 59.6e-9

    def determine_binning(self, filepath: str):
        """silly old binning convention -> change when labview updated to new binning"""
        bin_conversion = [1, 2, 3, 4, 8]
        metadata = self._read_metadata(filepath)
        return bin_conversion[int(metadata["Binning"])]


class Argus(LVControl):
    """
    Specific system details for Argus room-temperature widefield QDM.
    """

    name = "Argus"
    _obj_mag = 40
    _obj_ref_focal_length = 200e-3
    _camera_tube_lens = 300e-3
    _sensor_pixel_pitch = 6.5e-6


class LegacyArgus(LVControl):
    """
    System for Argus with old binning convention
    """

    name = "Legacy Argus"
    _obj_mag = 40
    _obj_ref_focal_length = 200e-3
    _camera_tube_lens = 300e-3
    _sensor_pixel_pitch = 6.5e-6

    def determine_binning(self, filepath: str):
        """Silly old binning convention -> change when labview updated to new binning"""
        bin_conversion = [1, 2, 3, 4, 8]
        metadata = self._read_metadata(filepath)
        return int(bin_conversion[metadata["Binning"]])


class PyCryoWidefield(PyControl):
    """
    Specific system details for Cryogenic (Attocube) widefield QDM.
    """

    name = "Py Cryo Widefield"
    _pixel_size = 59.6e-9


# ============================================================================
