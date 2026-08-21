import logging
from typing import override

from numpy import floating
from numpy.typing import NDArray
from pytsmod import ola, phase_vocoder, wsola

from third_party.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class OLA(BaseAdapter):
    def __init__(self, rate: float, win_size: int = 1024, win_type: str = 'hann',overlap_ratio: float = 0.5):
        """
        Initialize the OLA with the given parameters.

        Args:
            rate: The rate of time stretching. (c.g. 1.0 = no change, 2.0 = double speed)
            win_size: The size of the window. Must be one of [128, 256, 512, 1024, 2048, 4096, 8192].
            win_type: The type of the window. Must be 'hann' or 'sin'.
            overlap_ratio: The overlap ratio between frames. Must be between 0.0 and 1.0.

        Raises:
            ValueError: If the input parameters are invalid.
        """
        if rate < 0.0:
            raise ValueError("rate must be non-negative")
        if not win_size in [128, 256, 512, 1024, 2048, 4096, 8192]:
            raise ValueError("win_size must be one of [128, 256, 512, 1024, 2048, 4096, 8192]")
        if not 0.0 < overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must be between 0.0 and 1.0")
        if win_type not in ['hann', 'sin']:
            raise ValueError("win_type must be 'hann' or 'sin'")

        self._rate: float = rate
        self._win_size: int = win_size
        self._win_type: str = win_type
        self._overlap_ratio: float = overlap_ratio
        self._synthetic_hop_size: int = int(win_size * overlap_ratio)

    @override
    def __call__(self, x: NDArray[floating]) -> NDArray[floating]:
        """
        Apply OLA algorithm to the input using pytsmod.

        Args:
            input (NDArray[floating]): The input audio signal.
                - shape must be 2D  e.g. (channels, samples)
                - dtype must be floating point
                - values must be in range [-1.0, 1.0]

        Returns:
            NDArray[floating]: The time-stretched audio signal.
            - dtype: same as input

        Raises:
            Exception: If an error occurs during the OLA algorithm.
        """
        try:
            return ola(
                x=x,
                s=1./self._rate,
                win_size=self._win_size,
                win_type=self._win_type,
                syn_hop_size=self._synthetic_hop_size,
            ).astype(x.dtype)
        except Exception as e:
            logger.error(f"Error in pytsmod.ola: {e}")
            raise

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def win_size(self) -> int:
        return self._win_size

    @property
    def win_type(self) -> str:
        return self._win_type

    @property
    def overlap_ratio(self) -> float:
        return self._overlap_ratio

    @property
    def synthetic_hop_size(self) -> int:
        return self._synthetic_hop_size


class WSOLA(OLA):
    def __init__(self, rate: float, win_size: int = 1024, win_type: str = 'hann', overlap_ratio: float = 0.5, tolerance_ratio: float = 0.25):
        """
        Initialize the WSOLA with the given parameters.

        Args:
            rate: The rate of time stretching. (c.g. 1.0 = no change, 2.0 = double speed)
            win_size: The size of the window. Must be one of [128, 256, 512, 1024, 2048, 4096, 8192].
            win_type: The type of the window. Must be 'hann' or 'sin'.
            overlap_ratio: The overlap ratio between frames. Must be between 0.0 and 1.0.
            tolerance_ratio: The tolerance ratio for WSOLA. Must be between 0.0 and 0.5.

        Raises:
            ValueError: If the input parameters are invalid.
        """
        super().__init__(rate, win_size, win_type, overlap_ratio)

        if not 0 < tolerance_ratio < 0.5:
            raise ValueError("tolerance_ratio must be between 0 and 0.5")

        self._tolerance_ratio: float = tolerance_ratio
        self._tolerance: int = int(self._win_size * self._tolerance_ratio)

    @override
    def __call__(self, x: NDArray[floating]) -> NDArray[floating]:
        """
        Apply WSOLA algorithm to the input using pytsmod.

        Args:
            input (NDArray[floating]): The input audio signal.
                - shape must be 2D  e.g. (channels, samples)
                - dtype must be floating point
                - values must be in range [-1.0, 1.0]

        Returns:
            NDArray[floating]: The time-stretched audio signal.
            - dtype: same as input

        Raises:
            Exception: If an error occurs during the WSOLA algorithm.
        """
        try:
            return wsola(
                x=x,
                s=1./self._rate,
                win_size=self._win_size,
                win_type=self._win_type,
                syn_hop_size=self._synthetic_hop_size,
                tolerance=self._tolerance,
            ).astype(x.dtype)
        except Exception as e:
            logger.error(f"Error in pytsmod.wsola: {e}")
            raise

    @property
    def tolerance_ratio(self) -> float:
        return self._tolerance_ratio

    @property
    def tolerance(self) -> int:
        return self._tolerance


class PV(OLA):
    def __init__(self, rate: float, win_size: int = 1024, win_type: str = 'hann', overlap_ratio: float = 0.5, phase_locking: bool = False):
        """
        Initialize the WSOLA with the given parameters.

        Args:
            rate: The rate of time stretching. (c.g. 1.0 = no change, 2.0 = double speed)
            win_size: The size of the window. Must be one of [128, 256, 512, 1024, 2048, 4096, 8192].
            win_type: The type of the window. Must be 'hann' or 'sin'.
            overlap_ratio: The overlap ratio between frames. Must be between 0.0 and 1.0.
            phase_locking: Whether to use phase locking.

        Raises:
            ValueError: If the input parameters are invalid.
        """
        super().__init__(rate, win_size, win_type, overlap_ratio)

        self._phase_locking: bool = phase_locking

    @override
    def __call__(self, x: NDArray[floating]) -> NDArray[floating]:
        """
        Apply Phase Vocoder algorithm to the input using pytsmod.

        Args:
            input (NDArray[floating]): The input audio signal.
                - shape must be 2D  e.g. (channels, samples)
                - dtype must be floating point
                - values must be in range [-1.0, 1.0]

        Returns:
            NDArray[floating]: The time-stretched audio signal.
                - dtype: same as input

        Raises:
            Exception: If an error occurs during the Phase Vocoder algorithm.
        """
        try:
            return phase_vocoder(
                x=x,
                s=1.0/self._rate,
                win_size=self._win_size,
                win_type=self._win_type,
                syn_hop_size=self._synthetic_hop_size,
                phase_lock=self._phase_locking
            ).astype(x.dtype)
        except Exception as e:
            logger.error(f"Error in pytsmod.phase_vocoder: {e}")
            raise

    @property
    def phase_locking(self) -> bool:
        return self._phase_locking
