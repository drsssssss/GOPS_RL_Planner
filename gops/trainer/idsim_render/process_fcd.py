import os
from dataclasses import dataclass
from operator import attrgetter
from typing import Iterator, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class Vehicle:
    __slots__ = ("id", "x", "y", "angle", "type", "speed", "pos", "lane", "slope")
    id: str
    x: float
    y: float
    angle: float
    type: str
    speed: float
    pos: float
    lane: str
    slope: float

    @staticmethod
    def from_xml(el: ET.Element) -> "Vehicle":
        # assert el.tag == "vehicle" and len(el.attrib) == 9
        if el.tag == "vehicle" and len(el.attrib) == 9:
            return Vehicle(
                id=el.attrib["id"],
                x=float(el.attrib["x"]),
                y=float(el.attrib["y"]),
                angle=float(el.attrib["angle"]),
                type=el.attrib["type"],
                speed=float(el.attrib["speed"]),
                pos=float(el.attrib["pos"]),
                lane=el.attrib["lane"],
                slope=float(el.attrib["slope"]),
            )
        ## person
        elif len(el.attrib) == 8:
            return Vehicle(
                id=el.attrib["id"],
                x=float(el.attrib["x"]),
                y=float(el.attrib["y"]),
                angle=float(el.attrib["angle"]),
                type='person',
                speed=float(el.attrib["speed"]),
                pos=float(el.attrib["pos"]),
                lane=el.attrib["edge"],
                slope=float(el.attrib["slope"]),
            )
        else:
            raise ValueError(f"Invalid vehicle: {el.attrib}")


@dataclass
class Timestep:
    __slots__ = ("time", "vehicles")
    time: float
    vehicles: List[Vehicle]

    @staticmethod
    def from_xml(el: ET.Element) -> "Timestep":
        assert el.tag == "timestep" and len(el.attrib) == 1
        return Timestep(
            time=float(el.attrib["time"]),
            vehicles=[Vehicle.from_xml(vehicle) for vehicle in el],
        )


@dataclass(repr=False)
class FCDLog:
    __slots__ = ("timesteps", "step_length")
    timesteps: List[Timestep]
    step_length: float

    def at(self, time: float) -> Timestep:
        # NOTE: already verified that the step length is constant
        index = round(time / self.step_length)
        return self.timesteps[index]

    def count(self) -> int:
        return len(self.timesteps)

    def __repr__(self):
        return f"FCDLog(count={self.count()}, step_length={self.step_length})"

    @staticmethod
    def from_iterator(iterator: Iterator[Tuple[str, ET.Element]]) -> "FCDLog":
        timesteps = []
        for _, el in iterator:
            if el.tag == "timestep":
                timesteps.append(Timestep.from_xml(el))
                el.clear()
            elif el.tag == "fcd-export":
                assert (
                    len(el.attrib) == 1
                    and el.attrib["{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation"]
                    == "http://sumo.dlr.de/xsd/fcd_file.xsd"
                )
                el.clear()

        # Verify arithmetic progression
        time_getter = attrgetter("time")
        times = np.fromiter(map(time_getter, timesteps), dtype=np.float64, count=len(timesteps))
        assert np.allclose(times[1:] - times[:-1], times[1] - times[0]) and times[0] == 0.0
        step_length = times[1] - times[0]
        return FCDLog(
            timesteps=timesteps,
            step_length=step_length,
        )

    @staticmethod
    def from_file(path: os.PathLike) -> "FCDLog":
        return FCDLog.from_iterator(ET.iterparse(path, events=("end",)))


if __name__ == "__main__":
    fcd_file = "data/fcd.xml"
    fcd_log = FCDLog.from_file(fcd_file)
    print(fcd_log.at(5.0))
