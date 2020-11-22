# -*- coding: utf-8 -*-

from collections import Counter
import copy
from csv import DictReader
from dataclasses import dataclass, field
from itertools import permutations, product
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
    def main_harvesters(self):
        for resource in sorted(self.resources, key=attrgetter("value"), reverse=True):
            return resource.harvesters

    @main_harvesters.setter
    def main_harvesters(self, main_harvesters):
        for resource in sorted(self.resources, key=attrgetter("value"), reverse=True):
            resource.harvesters = int(main_harvesters)
            return

    @property
    def max_value(self) -> float:
        return max(self.resources, key=attrgetter("value")).value

    @property
    def most_valuable(self) -> str:
        return max(self.resources, key=attrgetter("value")).name

    @property
    def sub_harvesters(self):
        harvesters = []
        for i, resource in enumerate(
            sorted(self.resources, key=attrgetter("value"), reverse=True)
        ):
            if i == 0:
                continue
            else:
                harvesters.append(resource.harvesters)
        return harvesters

    @sub_harvesters.setter
    def sub_harvesters(self, sub_harvesters):
        for resource, harvesters in zip(
            sorted(self.resources, key=attrgetter("value"), reverse=True)[1:],
            sub_harvesters,
        ):
            resource.harvesters = int(harvesters)

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
        self.best_combinations = []
        self.characters = characters
        self.harvesters = harvesters
        self.input_planets = input_planets
        self.permutations = None
        self.wanted_planets = wanted_planets
        self.wanted_resources = wanted_resources

    def can_fullfill_wants(self, combination):
        """Checks whether the harvester setup on the different planets fullfills the wants"""
        available_resources = set()
        for planet in combination:
            for resource in planet.resources:
                if resource.harvesters > 0:
                    available_resources.add(resource.name)
        return available_resources == self.wanted_resources

    def evaluate_yield(self, selections):
        # Round to evens then turn into int indexes
        selections = np.round(np.array(selections)).astype(int)
        selected_planets = [self.input_planets[selection] for selection in selections]

        # If we have too many selections of the same planet, we return 0
        planet_distribution = Counter(planet.name for planet in selected_planets)
        if any(value > self.characters for value in planet_distribution.values()):
            return 0

        # If we cannot get all the resources we need as a combination of planet resources, we return 0
        if not self.theoretically_ok(selected_planets):
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
        combinations = product(*self.permutations)
        valid_combinations = [
            combination
            for combination in combinations
            if self.can_fullfill_wants(combination)
        ]
        best_value = max(sum(planet.total_value for planet in combination) for combination in valid_combinations)
        for combination in valid_combinations:
            if np.isclose(sum(planet.total_value for planet in combination), best_value):
                self.best_combinations.append(combination)
                return combination

    def get_planet_permutations(self, selected_planets):
        """
        Valid permutations contain harvesters only assigned to the resources that are in the want list.
        Only the most valuable resource can have more than one harvester assigned to it. 0-N other resources can have
        harvesters assigned to them.
        The number of permutations per planet is 2^(N-1) where N is the number of resources

        Note: Currently assumes that number of harvesters is higher than number of different resources present.
        """
        self.permutations = []
        for planet in selected_planets:
            planet_permutations = []
            for i, _ in enumerate(planet.resources):
                main_harvesters = self.harvesters - i
                other_harvesters = i * "1" + (len(planet.resources) - i - 1) * "0"
                sub_selections = set(permutations(other_harvesters))

                for harv_selections in sub_selections:
                    new_permute = copy.deepcopy(planet)
                    new_permute.main_harvesters = main_harvesters
                    new_permute.sub_harvesters = harv_selections
                    planet_permutations.append(new_permute)

            self.permutations.append(planet_permutations)

    def optimize_planets(self):
        self.sanitize_planets()
        differential_evolution(
            func=self.evaluate_yield,
            bounds=[
                (0, len(self.input_planets) - 1) for _ in range(self.wanted_planets)
            ],
            maxiter=10,
            popsize=len(self.wanted_resources) * 300,
            # mutation=(1, 1.9),
            updating="deferred",
            workers=5,
            atol=10000,
        )
        best_value = max(sum(planet.total_value for planet in combination) for combination in self.best_combinations)
        best_combination = None
        for combination in self.best_combinations:
            if np.isclose(sum(planet.total_value for planet in combination), best_value):
                best_combination = combination
                break
        print(f"Selected planets are:")
        for planet in best_combination:
            print(planet)
        print(f"Total value: {best_value}")

    def sanitize_planets(self):
        """Removes the unneeded data from the pool"""
        # Removing the useless resources from each planet
        for planet in self.input_planets:
            planet.resources = [
                resource
                for resource in planet.resources
                if resource.name in self.wanted_resources
            ]
        # Removing planets without resources from the pool
        self.input_planets = [
            planet for planet in self.input_planets if planet.resources
        ]

    def theoretically_ok(self, select_planets):
        """Determines whether the select planets contain all the wanted resources."""
        select_resources = set()
        for planet in select_planets:
            for resource in planet.resources:
                select_resources.add(resource.name)

        return self.wanted_resources.issubset(select_resources)


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
