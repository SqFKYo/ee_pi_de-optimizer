# -*- coding: utf-8 -*-

from collections import defaultdict
import copy
from csv import DictReader
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations, permutations, product
from operator import attrgetter

import numpy as np

PI_RAW_VALUES = """
Lustering Alloy	99 
Sheen Compound	159 
Gleaming Alloy	198 
Condensed Alloy	157 
Precious Alloy	340 
Motley Compound	134 
Fiber Composite	91 
Lucent Compound	148 
Opulent Compound	147 
Glossy Compound	118 
Crystal Compound	208 
Dark Compound	98 
Reactive Gas	81 
Noble Gas	47 
Base Metals	123 
Heavy Metals	268 
Noble Metals	160 
Reactive Metals	772 
Toxic Metals	761 
Industrial Fibers	70 
Supertensile Plastics	179 
Polyaramids	83 
Coolant	64 
Condensates	121 
Construction Blocks	164 
Nanites	89 
Silicate Glass	173 
Smartfab Units	101 
Heavy Water	3 
Suspended Plasma	10 
Liquid Ozone	34 
Ionic Solutions	174 
Oxygen Isotopes	397 
Plasmoids	494 
"""


def parse_pi_values(round_to=None):
    pi_values = {}
    for raw_line in PI_RAW_VALUES.split("\n"):
        # Skip empty lines
        if not raw_line:
            continue
        name, value = raw_line.split("\t")
        unrounded = float(value.replace(",", ""))
        if round_to is None:
            pi_values[name] = unrounded
        else:
            rounded = round_to * round(unrounded / round_to)
            pi_values[name] = rounded
    return pi_values


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
    def offered_resources(self) -> set:
        return set(resource.name for resource in self.resources)

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
class PlanetResource:
    planet: str
    output: float

    def __lt__(self, other):
        return self.output < other.output


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
        return pi_values[self.name] * self.output


class Optimizer:
    def __init__(
        self,
        input_planets,
        wanted_planets,
        wanted_resources=None,
        harvesters=8,
        characters=2,
    ):
        self.characters = characters
        self.harvesters = harvesters
        self.input_planets = input_planets
        self.permutations = None
        if wanted_resources is None:
            self.wanted_resources = list(pi_values.keys())
        else:
            self.wanted_planets = wanted_planets
        self.wanted_resources = wanted_resources

    def can_fulfill_wants(self, combination):
        """Checks whether the harvester setup on the different planets fulfills the wants"""
        # ToDo: Need to add check to respect the boundaries based on the min/max harvesters that can be tied to one resource
        available_resources = set()
        for planet in combination:
            for resource in planet.resources:
                if resource.harvesters > 0:
                    available_resources.add(resource.name)
        return (self.wanted_resources).issubset(available_resources)

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
        combos = product(*self.permutations)
        valid_combinations = [
            combination for combination in combos if self.can_fulfill_wants(combination)
        ]
        best_value = max(
            sum(planet.total_value for planet in combination)
            for combination in valid_combinations
        )
        for combination in valid_combinations:
            if np.isclose(
                sum(planet.total_value for planet in combination), best_value
            ):
                return combination

    def get_planet_permutations(self, selected_planets):
        """
        # ToDo: Need to redo this in order to allow for more variety in the end result

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

    def is_planet_useless(self, planet):
        """
        Does the planet have resources which are a subset of another input planet but their max value is lower
        """
        for input_planet in self.input_planets:
            if (
                planet.offered_resources.issubset(input_planet.offered_resources)
                and planet.max_value < input_planet.max_value
            ):
                return True
        else:
            return False

    def optimize_planets(self):
        """
        Try to see how many planets can satisfy the needs.
        If e.g. 4 can satisfy the needs, and 8 total, take first 4 most valuable and use 4 others to satisfy needs

        # ToDo: Optimize for more variety. Farming mostly for one / two resources is a huge market risk
        #  Currently 72 harvesters, 7 materials, 9 planets can yield: 1, 1, 1, 1, 7, 13, 48
        """

        self.remove_useless()
        original_inputs = copy.deepcopy(self.input_planets)
        sorted_inputs = list(
            sorted(original_inputs, key=attrgetter("max_value"), reverse=True)
        )

        optimum_planets = None
        for i in range(self.wanted_planets, 0, -1):
            given_planets = sorted_inputs[:i]
            self.input_planets = sorted_inputs[i:]
            self.remove_less_useful()
            optimum_planets = self.optimize_subset(
                given_planets=given_planets, wanted_planets=self.wanted_planets - i,
            )
            if optimum_planets:
                break

        print(f"\nSelected planets are:")
        for planet in optimum_planets:
            print(planet)
        best_value = sum(planet.total_value for planet in optimum_planets)
        print(f"Total value: {best_value}")

    def optimize_subset(self, given_planets, wanted_planets):
        """
        Returns the most valuable combination of planets given that some planets are preselected
        If wanted planets is 0, return given planets
        If no set of planets is found to satisfy the wanted resources, return None
        """
        if wanted_planets < 1:
            if self.theoretically_ok(select_planets=given_planets):
                return given_planets
            else:
                return None

        best_value = 0
        best_combination = None
        count = 0
        subcounter = 0
        for selected_planets in combinations(self.input_planets, wanted_planets):
            selected_planets = list(selected_planets)
            selected_planets.extend(given_planets)
            count += 1
            subcounter += 1
            if subcounter == 10000:
                print(
                    f"We have now calculated through {count} iterations at {datetime.now()}!"
                )
                subcounter = 0
            if not self.theoretically_ok(selected_planets):
                continue
            else:
                # Find the combination of the most valuable permutation of the planets
                optimal_planets = self.get_optimal_distribution(selected_planets)
                total_value = sum(planet.total_value for planet in optimal_planets)
                if total_value > best_value:
                    print(
                        f"Found new best combo totaling {total_value}: {optimal_planets}"
                    )
                    best_value = total_value
                    best_combination = optimal_planets
        print(f"\nCalculated through {count} different combinations of planets.")
        return best_combination

    def print_valuable_pi(self):
        done_resources = []
        for planet in sorted(
            self.input_planets, key=attrgetter("max_value"), reverse=True
        ):
            if planet.most_valuable not in done_resources:
                print(
                    f"Most valuable is {planet.most_valuable} at {planet.max_value:.2f} isk!"
                )
                done_resources.append(planet.most_valuable)

    def print_valuable_planets(self, amount, unique_resources=False):
        i = 0
        done_resources = set()
        for planet in sorted(
            self.input_planets, key=attrgetter("max_value"), reverse=True
        ):
            if unique_resources and planet.most_valuable in done_resources:
                continue
            else:
                print(f"{planet.name}, {planet.most_valuable}, {planet.max_value:.2f}")
                done_resources.add(planet.most_valuable)
                i += 1
            if i == amount:
                break

    def remove_less_useful(self):
        # ToDo: Needs to take into account that less valuable planets might be needed if a single resource is dominating
        # If the resources form identical sets, leave only the ones that have the highest value resource
        resource_groups = defaultdict(list)
        for planet in self.input_planets:
            resources = frozenset([resource.name for resource in planet.resources])
            resource_groups[resources].append(planet)
        self.input_planets = []
        for planet_group in resource_groups.values():
            if len(planet_group) == 1:
                self.input_planets.append(planet_group[0])
                continue
            else:
                self.input_planets.append(
                    sorted(planet_group, key=attrgetter("max_value"), reverse=True)[0]
                )
        print(
            f"Planets after removing the ones containing less useful similar groups: {len(self.input_planets)}"
        )

        planet_groups = defaultdict(list)
        for planet in self.input_planets:
            planet_groups[len(planet.resources)].append(planet)
        # First add back the largest groups of resources, then see if there are valuable additions in the subgroups
        most_resources = max(planet_groups.keys())
        self.input_planets = planet_groups[most_resources]
        for res_len in sorted(planet_groups.keys(), reverse=True):
            if res_len == most_resources:
                continue
            for planet in planet_groups[res_len]:
                if self.is_planet_useless(planet):
                    continue
                else:
                    self.input_planets.append(planet)

        print(
            f"Planets after removing the less valuables that had subset of other planet resources: {len(self.input_planets)}"
        )

    def remove_useless(self):
        """Removes the unneeded data from the pool"""

        # Removing the useless resources from each planet
        print(f"Planets at the beginning: {len(self.input_planets)}")
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
        print(
            f"Planets after removing ones without any of the needed resources: {len(self.input_planets)}"
        )

    def theoretically_ok(self, select_planets):
        """Determines whether the select planets contain all the wanted resources."""
        select_resources = set()
        for planet in select_planets:
            for resource in planet.resources:
                select_resources.add(resource.name)

        return self.wanted_resources.issubset(select_resources)


class PlanetRanker:
    def __init__(self, planet_resources, wanted_resources=None):
        self.planet_resources = planet_resources
        self.wanted_resources = wanted_resources
        self._filter_resources()


    def _filter_resources(self):
        """Filters out the resources we do not need. If planet resources is set to None, we don't filter."""
        if self.wanted_resources is None:
            return
        self.planet_resources = {
            resource: sub_resources
            for resource, sub_resources in self.planet_resources.items()
            if resource in self.wanted_resources
        }

    def print_best_of_each(self, number=10):
        for name, sub_resources in sorted(self.planet_resources.items()):
            if not sub_resources:
                continue
            print(f"\n{name}")
            print(f"Planet, output, isk")
            for res in sorted(sub_resources, reverse=True)[:number]:
                print(f"{res.planet}, {res.output}, {res.output*pi_values[name]:.2f}")


    def print_total_best(self, number=10):
        """
        # ToDo Prints the best planets in the given subspace
        Each planet's value is PI value divided by
        how good the planet is in the set. So the best Toxic Metal planet would get full multiplier, the 2nd
        would get half, 3rd 1/3rd etc. Then sum all the subcomponents we're interested in.
        """

        pass


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


def read_resources(resource_file):
    resources = defaultdict(list)
    with open(resource_file, "r") as f:
        reader = DictReader(f, delimiter="\t")
        for line in reader:
            resources[line["Resource"]].append(
                PlanetResource(planet=line["Planet Name"], output=float(line["Output"]))
            )
    return resources


if __name__ == "__main__":
    start = datetime.now()
    pi_values = parse_pi_values()
    print(f"Starting optimization at {start}")
    wanted_resources = {
        # "Base Metals",
        "Condensed Alloy",
        "Crystal Compound",
        # "Fiber Composite",
        "Gleaming Alloy",
        "Glossy Compound",
        "Heavy Metals",
        # "Lucent Compound",
        "Lustering Alloy",
        "Motley Compound",
        "Noble Metals",
        "Opulent Compound",
        "Precious Alloy",
        "Reactive Metals",
        "Sheen Compound",
        "Toxic Metals",
    }

    planet_resources = read_resources(r"C:\Users\sqfky\Desktop\ee_planets.txt")
    planet_ranker = PlanetRanker(planet_resources, wanted_resources=wanted_resources)
    planet_ranker.print_best_of_each(number=5)
    planet_ranker.print_total_best(number=20)
    # input_planets = read_planets(r"C:\Users\sqfky\Desktop\ee_planets.txt")
    # input_planets = read_planets(r"C:\Users\sqfky\Desktop\ee_planets_nearby.txt")
    # optimizer = Optimizer(
    #     input_planets=input_planets,
    #     wanted_planets=9,
    #     wanted_resources=wanted_resources,
    # )
    # optimizer.print_valuable_pi()
    # optimizer.print_valuable_planets(amount=50, unique_resources=True)
    # optimizer.optimize_planets()
    end = datetime.now()
    print(f"Optimization finished at {end}, time taken {end-start}.")
