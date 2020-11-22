# -*- coding: utf-8 -*-

from collections import Counter
import copy
from csv import DictReader
from dataclasses import dataclass, field
from operator import attrgetter

import numpy as np
from scipy.optimize import differential_evolution


VALUES = {
    "Base Metals": 68,
    "Condensed Alloy": 150,
    "Crystal Compound": 249,
    "Dark Compound": 286,
    "Fiber Composite": 74,
    "Gleaming Alloy": 269,
    "Glossy Compound": 138,
    "Heavy Metals": 208,
    "Lucent Compound": 141,
    "Lustering Alloy": 112,
    "Motley Compound": 167,
    "Noble Metals": 147,
    "Opulent Compound": 78,
    "Precious Alloy": 285,
    "Reactive Metals": 569,
    "Sheen Compound": 168,
    "Toxic Metals": 579,
}


@dataclass
class Planet:
    name: str
    resources: list = field(default_factory=list)

    def __repr__(self):
        return (
            f"Name: {self.name}, total value: {self.total_value:.2f}, "
            f"most valuable: {self.most_valuable} ({self.max_value:.2f}), "
            f"resources: {sorted(self.resources, reverse=True)}"
        )

    @property
    def max_value(self) -> float:
        return max(self.resources, key=attrgetter("value")).value

    @property
    def most_valuable(self) -> str:
        return max(self.resources, key=attrgetter("value")).name

    @property
    def total_value(self) -> float:
        return sum(resource.total_value for resource in self.resources)

    def append(self, resource):
        self.resources.append(resource)


@dataclass
class Resource:
    name: str
    richness: str
    output: float
    harvesters: int = field(default_factory=int)

    def __repr__(self):
        return (
            f"Name: {self.name}, richness: {self.richness}, output: {self.output}, value: {self.value:.2f}, "
            f"harvesters: {self.harvesters}, total value: {self.total_value:.2f}"
        )

    def __lt__(self, other):
        return self.value < other.value

    @property
    def total_value(self) -> float:
        return self.value * self.harvesters

    @property
    def value(self) -> float:
        return VALUES[self.name] * self.output


class Optimizer:
    def __init__(
        self,
        input_planets,
        wanted_planets,
        wanted_resources,
        harvesters=8,
        characters=2,
    ):
        self.characters = characters
        self.harvesters = harvesters
        self.input_planets = input_planets
        self.permutations = None
        self.wanted_planets = wanted_planets
        self.wanted_resources = wanted_resources

    def calculate_planet_permutations(self, planet):
        """
        # ToDo: This doesn't find nearly all combinations, so will be scrapped / rewritten
        Returns variations on how the harvesters could be divided
        Amount of variations depends on the amount of resources:
        One permutation is all harvesters to the most valuable resource.
        Second permutation sets one harvester for the second most valuable resource, third on the third etc. until
        either all harvesters are used up or all resources have at least one harvester.
        """
        for _, i in zip(planet.resources, range(self.harvesters)):
            new_permute = copy.deepcopy(planet)
            new_permute.resources = sorted(
                new_permute.resources, key=attrgetter("value"), reverse=True
            )
            new_permute.resources[0].harvesters = self.harvesters - i
            for j, resource in enumerate(
                sorted(new_permute.resources, reverse=True)[: i + 1]
            ):
                if j == 0:
                    continue
                else:
                    resource.harvesters = 1
            self.permutations.append(new_permute)

    def evaluate_yield(self, selections):
        # Round to evens then turn into int indexes
        selections = np.round(np.array(selections)).astype(int)
        select_planets = [self.permutations[selection] for selection in selections]

        # If we have too many selections of the same planet permutation, we return 0
        planet_distribution = Counter(planet.name for planet in select_planets)
        if any(value > self.characters for value in planet_distribution.values()):
            return 0
        # ToDo: If we cannot get all the resources we need as a combination of planet resources, we return 0

        # ToDo: Return the value of the most valuable permutation of the planets

        # If we don't get all the resources we need, we return 0 again # ToDo this check will be obsolete
        select_resources = set()
        for planet in select_planets:
            for resource in planet.resources:
                if resource.harvesters > 0:
                    select_resources.add(resource.name)
        if not self.wanted_resources.issubset(select_resources):
            return 0
        total_value = sum(planet.total_value for planet in select_planets)
        print(f"All resources found! Value: {total_value}")
        print(select_planets)
        return -total_value

    def optimize_planets(self):
        # ToDo: Replace static permutations with dynamic during the evaluation
        self.permutations = []
        for planet in self.input_planets.values():
            self.calculate_planet_permutations(planet)

        res = differential_evolution(
            func=self.evaluate_yield,
            bounds=[
                (0, len(self.permutations) - 1) for _ in range(self.wanted_planets)
            ],
            maxiter=100,
            popsize=len(self.wanted_resources) * 300,
            mutation=(1, 1.9),
            updating="deferred",
            workers=5,
            atol=10000,
        )
        print(f"Message: {res.message}")
        print(f"Selected planets are:")
        result_planets = [self.permutations[round(selection)] for selection in res.x]
        for planet in result_planets:
            print(planet)
        print(f"Total value: {sum(planet.total_value for planet in result_planets)}")


def read_planets(planet_file):
    planets = {}
    with open(planet_file, "r") as f:
        reader = DictReader(f, delimiter="\t")
        for line in reader:
            planet_name = line["Planet Name"]
            if planet_name not in planets:
                planets[planet_name] = Planet(name=planet_name)
            resource = Resource(
                name=line["Resource"],
                richness=line["Richness"],
                output=float(line["Output"]),
            )
            planets[planet_name].append(resource)
    return planets


if __name__ == "__main__":
    from datetime import datetime

    start = datetime.now()
    print(f"Starting optimization at {start}")
    wanted_resources = {
        "Condensed Alloy",
        "Crystal Compound",
        "Dark Compound",
        "Gleaming Alloy",
        "Heavy Metals",
        "Lucent Compound",
        "Motley Compound",
        "Noble Metals",
        "Opulent Compound",
        "Precious Alloy",
        "Reactive Metals",
        "Sheen Compound",
        "Toxic Metals",
    }
    planets = read_planets(r"C:\Users\sqfky\Desktop\ee_planets.txt")
    optimizer = Optimizer(
        input_planets=planets, wanted_resources=wanted_resources, wanted_planets=8
    )
    optimizer.optimize_planets()
    end = datetime.now()
    print(f"Optimization finished at {end}, time taken {end-start}.")
