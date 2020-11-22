# -*- coding: utf-8 -*-

from collections import Counter
import copy
from csv import DictReader
from dataclasses import dataclass, field
from itertools import product
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
        self.planet_permutations = None
        self.wanted_planets = wanted_planets
        self.wanted_resources = wanted_resources

    def can_fullfill_wants(self, select_planets):
        """Determines whether the select planets contain all the wanted resources."""
        select_resources = set()
        for planet in select_planets:
            for resource in planet.resources:
                    select_resources.add(resource.name)

        if self.wanted_resources.issubset(select_resources):
            return True
        else:
            return False

    def check_validity(self, planet):
        # Is the planet setup right, i.e. are harvesters only assigned to wanted resources
        # ToDo: Do we need this check here? Shouldn't we not make non-valid permutations in the first place?
        return set(resource.name for resource in planet.resources if resource.harvesters > 0).issubset(self.wanted_resources)

    def evaluate_yield(self, selections):
        # Round to evens then turn into int indexes
        selections = np.round(np.array(selections)).astype(int)
        selected_planets = [self.input_planets[selection] for selection in selections]

        # If we have too many selections of the same planet, we return 0
        planet_distribution = Counter(planet.name for planet in selected_planets)
        if any(value > self.characters for value in planet_distribution.values()):
            return 0

        # If we cannot get all the resources we need as a combination of planet resources, we return 0
        if not self.can_fullfill_wants(selected_planets):
            return 0

        # Find the combination of the most valuable permutation of the planets
        optimal_planets = self.get_optimal_distribution(selected_planets)

        total_value = sum(planet.total_value for planet in optimal_planets)
        print(f"All resources found! Value: {total_value}")
        print(optimal_planets)
        # Returns the inversion of total value, since the optimization function searches for the minimum value
        return -total_value

    def get_optimal_distribution(self, selected_planets):
        """
        Calculates the most valuable permutation of the select planets that fullfills the wants
        Assumes that we don't need to worry about having too many duplicate planets, and just wants to optimize
        the value based on the harvesters we have assuming we have all the wants covered.

        To optimize the value, we put as many harvesters as possible to the most valuable resource and 1 harvester
        to each of the resource we need to have based on the wants. In order to find the most profitable combination,
        we need all valid permutations (=have harvesters assigned only to wanted resources, and >1 only to the most
        valuable) for each planet, and then select the most valuable valid combination out of those valid permutations.

        1) Create all valid permutations for each planet
        2) Create all the combinations out of the valid planet permutations
        3) Discard all the invalid combinations of permutations that do not satisfy the wants
        4) Find the most valuable valid combination
        """
        self.get_planet_permutations(selected_planets)
        combinations = product(self.planet_permutations)
        valid_combinations = [combination for combination in combinations if self.can_fullfill_wants(combination)]
        # ToDo copypaste code
        # DEBUG
        return selected_planets
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

    def get_planet_permutations(self, selected_planets):
        """
        ToDo: Finds all the valid the planet permutations for each of the selected planets.

        Valid permutations contain harvesters only assigned to the resources that are in the want list.
        Only the most valuable resource can have more than one harvester assigned to it. 0-N other resources can have
        harvesters assigned to them.
        """
        self.planet_permutations = []
        for planet in selected_planets:
            valid_permutations = []
            # ToDo: valid_permutations = [planet for planet in planet.permutations if planet.is_valid(wanted_resources=self.wanted_resources)]
            self.planet_permutations.append(valid_permutations)

    def optimize_planets(self):
        res = differential_evolution(
            func=self.evaluate_yield,
            bounds=[
                (0, len(self.input_planets) - 1) for _ in range(self.wanted_planets)
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
        result_planets = [self.input_planets[round(selection)] for selection in res.x]
        for planet in result_planets:
            print(planet)
        print(f"Total value: {sum(planet.total_value for planet in result_planets)}")

    def sanitize_planets(self):
        """Removes the unneeded data from the pool"""
        # Removing the useless resources from each planet
        for planet in self.input_planets:
            planet.resources = [resource for resource in planet.resources if resource.name in self.wanted_resources]
        # Removing planets without resources from the pool
        self.input_planets = [planet for planet in self.input_planets if planet.resources]


def read_planets(planet_file):
    # We start with a dict so we don't end up with duplicate planets in the list to return
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
    # We return a list of the planets
    return list(planets.values())


if __name__ == "__main__":
    from datetime import datetime

    start = datetime.now()
    print(f"Starting optimization at {start}")
    wanted_resources = {
        "Condensed Alloy",
        "Crystal Compound",
        # "Dark Compound",
        # "Gleaming Alloy",
        # "Heavy Metals",
        # "Lucent Compound",
        # "Motley Compound",
        "Noble Metals",
        # "Opulent Compound",
        # "Precious Alloy",
        # "Reactive Metals",
        # "Sheen Compound",
        # "Toxic Metals",
    }
    planets = read_planets(r"C:\Users\sqfky\Desktop\ee_planets.txt")
    optimizer = Optimizer(
        input_planets=planets, wanted_resources=wanted_resources, wanted_planets=8
    )
    optimizer.sanitize_planets()
    # DEBUG
    print(optimizer.evaluate_yield(selections=[0, 1, 2, 3, 4, 5, 7, 7, 8]))
    # optimizer.optimize_planets()
    end = datetime.now()
    print(f"Optimization finished at {end}, time taken {end-start}.")
