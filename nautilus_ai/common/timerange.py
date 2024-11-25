import re
from datetime import datetime, timezone
from typing import Optional, Tuple
from typing_extensions import Self
from nautilus_ai.common.constants import DATETIME_PRINT_FORMAT
from nautilus_ai.common.logging import Logger
from nautilus_ai.exceptions import ConfigurationError

logger = Logger(__name__)


class TimeRange:
    """
    Object defining timerange inputs.
    - The `starttype` and `stoptype` determine if `startts` or `stopts` should be used.
    - If `starttype` or `stoptype` is `None`, the corresponding value is ignored.

    Attributes:
    -----------
    starttype : Optional[str]
        Type of start timestamp (e.g., "date").
    stoptype : Optional[str]
        Type of stop timestamp (e.g., "date").
    startts : int
        Start timestamp (UNIX epoch).
    stopts : int
        Stop timestamp (UNIX epoch).
    """

    def __init__(
        self,
        starttype: Optional[str] = None,
        stoptype: Optional[str] = None,
        startts: int = 0,
        stopts: int = 0,
    ):
        self.starttype: Optional[str] = starttype
        self.stoptype: Optional[str] = stoptype
        self.startts: int = startts
        self.stopts: int = stopts

    @property
    def startdt(self) -> Optional[datetime]:
        """
        Returns:
        --------
        Optional[datetime]
            Start datetime in UTC, or None if `startts` is not set.
        """
        return (
            datetime.fromtimestamp(self.startts, tz=timezone.utc)
            if self.startts
            else None
        )

    @property
    def stopdt(self) -> Optional[datetime]:
        """
        Returns:
        --------
        Optional[datetime]
            Stop datetime in UTC, or None if `stopts` is not set.
        """
        return (
            datetime.fromtimestamp(self.stopts, tz=timezone.utc)
            if self.stopts
            else None
        )

    @property
    def timerange_str(self) -> str:
        """
        Provides a string representation of the timerange in `yyyymmdd-yyyymmdd` format.

        Returns:
        --------
        str
            Timerange string with empty values for unbounded ranges.
        """
        start = self.startdt.strftime("%Y%m%d") if self.startdt else ""
        stop = self.stopdt.strftime("%Y%m%d") if self.stopdt else ""
        return f"{start}-{stop}"

    @property
    def start_fmt(self) -> str:
        """
        Returns a formatted string representation of the start date.

        Returns:
        --------
        str
            Formatted start date or "unbounded" if not set.
        """
        return (
            self.startdt.strftime(DATETIME_PRINT_FORMAT)
            if self.startdt
            else "unbounded"
        )

    @property
    def stop_fmt(self) -> str:
        """
        Returns a formatted string representation of the stop date.

        Returns:
        --------
        str
            Formatted stop date or "unbounded" if not set.
        """
        return (
            self.stopdt.strftime(DATETIME_PRINT_FORMAT) if self.stopdt else "unbounded"
        )

    def __eq__(self, other: Self) -> bool:
        """
        Check equality of two TimeRange objects.

        Parameters:
        -----------
        other : TimeRange
            Another TimeRange object.

        Returns:
        --------
        bool
            True if both objects are equal, False otherwise.
        """
        return (
            self.starttype == other.starttype
            and self.stoptype == other.stoptype
            and self.startts == other.startts
            and self.stopts == other.stopts
        )

    def subtract_start(self, seconds: int) -> None:
        """
        Subtracts a specified number of seconds from the start timestamp.

        Parameters:
        -----------
        seconds : int
            Number of seconds to subtract.

        Returns:
        --------
        None
        """
        if self.startts:
            self.startts -= seconds

    def adjust_start_if_necessary(
        self, timeframe_secs: int, startup_candles: int, min_date: datetime
    ) -> None:
        """
        Adjusts the start timestamp if the available data is insufficient for the required startup candles.

        Parameters:
        -----------
        timeframe_secs : int
            Timeframe in seconds.
        startup_candles : int
            Number of startup candles required.
        min_date : datetime
            Minimum available data date.

        Returns:
        --------
        None
        """
        if not self.starttype or (
            startup_candles and min_date.timestamp() >= self.startts
        ):
            logger.warning(
                "Adjusting start date by %s candles to account for startup time.",
                startup_candles,
            )
            self.startts = int(min_date.timestamp() + timeframe_secs * startup_candles)
            self.starttype = "date"

    @classmethod
    def parse_timerange(cls, text: Optional[str]) -> Self:
        """
        Parses a timerange string in various formats to create a TimeRange object.

        Parameters:
        -----------
        text : Optional[str]
            Timerange string (e.g., "20220101-20221231", "-20220101").

        Returns:
        --------
        TimeRange
            A TimeRange object based on the parsed input.

        Raises:
        -------
        ConfigurationError
            If the timerange format is invalid or the start date is after the stop date.
        """
        if not text:
            return cls()

        syntax = [
            (r"^-(\d{8})$", (None, "date")),
            (r"^(\d{8})-$", ("date", None)),
            (r"^(\d{8})-(\d{8})$", ("date", "date")),
            (r"^-(\d{10})$", (None, "date")),
            (r"^(\d{10})-$", ("date", None)),
            (r"^(\d{10})-(\d{10})$", ("date", "date")),
            (r"^-(\d{13})$", (None, "date")),
            (r"^(\d{13})-$", ("date", None)),
            (r"^(\d{13})-(\d{13})$", ("date", "date")),
        ]

        for regex, stype in syntax:
            match = re.match(regex, text)
            if match:
                groups = match.groups()
                start, stop = cls._parse_start_stop(groups, stype)
                if start > stop > 0:
                    raise ConfigurationError(
                        f'Start date is after stop date for timerange "{text}"'
                    )
                return cls(stype[0], stype[1], start, stop)

        raise ConfigurationError(f'Incorrect syntax for timerange "{text}"')

    @staticmethod
    def _parse_start_stop(
        groups: Tuple[str, ...], stype: Tuple[Optional[str], Optional[str]]
    ) -> Tuple[int, int]:
        """
        Parse start and stop timestamps from the regex groups.

        Parameters:
        -----------
        groups : Tuple[str, ...]
            Regex match groups.
        stype : Tuple[Optional[str], Optional[str]]
            Start and stop types (e.g., "date").

        Returns:
        --------
        Tuple[int, int]
            Parsed start and stop timestamps.
        """
        start, stop = 0, 0
        if stype[0]:
            start = TimeRange._convert_to_timestamp(groups[0], stype[0])
        if stype[1]:
            stop = TimeRange._convert_to_timestamp(groups[1], stype[1])
        return start, stop

    @staticmethod
    def _convert_to_timestamp(value: str, ttype: str) -> int:
        """
        Convert a date string or timestamp string to a UNIX epoch timestamp.

        Parameters:
        -----------
        value : str
            Date or timestamp string.
        ttype : str
            Type of the timestamp ("date").

        Returns:
        --------
        int
            Converted UNIX epoch timestamp.
        """
        if ttype == "date" and len(value) == 8:
            return int(
                datetime.strptime(value, "%Y%m%d")
                .replace(tzinfo=timezone.utc)
                .timestamp()
            )
        elif len(value) == 13:
            return int(value) // 1000
        return int(value)
