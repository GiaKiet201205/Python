from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
import time
import random
import numpy as np
from typing import List, Dict, Tuple
from copy import deepcopy
from multiprocessing import Pool
import json

"""
Advanced MDVRP Solver - Enhanced Version (FIXED)
+ Genetic Algorithm (GA)
+ 3-opt Local Search (Full Implementation)
+ Time Windows Support
+ Parallel Execution
+ Heterogeneous Fleet Support

GA Fixes applied:
- GA chromosome is now a 1D sequence (list) of all customers.
- `_split_customers_into_routes_for_active_fleet` is a proper "decoder".
- `_evaluate_routes` (fitness function) now adds a heavy penalty for unserved customers.
"""


class AdvancedMDVRPSolver:
    def __init__(self, depots, customers, num_vehicles_per_depot,
                 vehicle_capacities=None, demands=None,
                 time_windows=None, service_times=None):
        self.depots = depots
        self.customers = customers
        self.num_vehicles_per_depot = num_vehicles_per_depot
        self.num_depots = len(depots)
        # Note: Keep full fleet count for OR-Tools; GA will manage active vehicles internally
        self.num_vehicles = num_vehicles_per_depot * self.num_depots
        self.num_customers = len(customers)

        self.all_locations = depots + customers
        self.distance_matrix = self._compute_distance_matrix()
        self.time_matrix = self._compute_time_matrix()

        # Demands & Capacities
        self.demands = demands if demands else [0] * self.num_depots + [1] * len(customers)
        # Ensure capacities list matches total number of vehicles if provided
        if vehicle_capacities and len(vehicle_capacities) != self.num_vehicles:
            print(
                f"Warning: vehicle_capacities length ({len(vehicle_capacities)}) != num_vehicles ({self.num_vehicles}). Using first capacity for all.")
            self.vehicle_capacities = [vehicle_capacities[0]] * self.num_vehicles
        elif vehicle_capacities:
            self.vehicle_capacities = vehicle_capacities
        else:
            self.vehicle_capacities = [100] * self.num_vehicles

        # Time Windows: [(start_time, end_time), ...]
        self.time_windows = time_windows if time_windows else [(0, 1000)] * len(self.all_locations)

        # Service times at each node
        self.service_times = service_times if service_times else [0] * self.num_depots + [30] * len(customers)

        # Vehicle start/end (for OR-Tools full model)
        self.starts = []
        self.ends = []
        for depot_idx in range(self.num_depots):
            for _ in range(num_vehicles_per_depot):
                self.starts.append(depot_idx)
                self.ends.append(depot_idx)

        self.best_solution = None
        self.best_distance = float('inf')

        # === GA FIX ===
        # Set of all customer indices (node index) for penalty calculation
        self.all_customer_indices = set(range(self.num_depots, self.num_depots + self.num_customers))
        # Pre-calculate a large penalty for the fitness function
        self._max_penalty = self._calculate_max_penalty()

    def _calculate_max_penalty(self):
        """Calculate a large penalty for unserved customers."""
        if not self.all_customer_indices:
            return 1_000_000  # Default large number

        max_penalty = 0
        for cust_idx in self.all_customer_indices:
            # Find nearest depot
            nearest_depot = min(range(self.num_depots), key=lambda d: self.distance_matrix[cust_idx][d])
            # Add round-trip distance
            max_penalty += self.distance_matrix[nearest_depot][cust_idx] * 2

        # Make penalty significantly larger than any possible route distance
        return max_penalty * 1.5 + 1_000_000

    def _compute_distance_matrix(self):
        """Compute Euclidean distance matrix"""
        distances = {}
        for from_counter, from_node in enumerate(self.all_locations):
            distances[from_counter] = {}
            for to_counter, to_node in enumerate(self.all_locations):
                if from_counter == to_counter:
                    distances[from_counter][to_counter] = 0
                else:
                    distances[from_counter][to_counter] = math.hypot(
                        from_node[0] - to_node[0],
                        from_node[1] - to_node[1]
                    )
        return distances

    def _compute_time_matrix(self):
        """Compute time matrix (assuming speed = 1)"""
        return self.distance_matrix

    def _get_routing_model_with_time_windows(self):
        """Routing model with Time Windows support"""
        manager = pywrapcp.RoutingIndexManager(
            len(self.all_locations),
            self.num_vehicles,
            self.starts,
            self.ends
        )
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(self.distance_matrix[from_node][to_node] * 100)

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Capacity constraint
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return self.demands[from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            self.vehicle_capacities,
            True,
            'Capacity'
        )

        # TIME WINDOWS constraint
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            travel_time = int(self.time_matrix[from_node][to_node])
            # Service time is added at the node, not on the arc
            # But OR-Tools callback includes service time of 'from_node'
            return travel_time + self.service_times[from_node]

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_index,
            3600,  # slack_max (e.g., 1 hour)
            10000,  # capacity (max time for a route)
            False,  # fix_start_cumul_to_zero (False because depots have windows)
            'Time'  # dimension name
        )

        time_dimension = routing.GetDimensionOrDie('Time')
        for location_idx, (start_time, end_time) in enumerate(self.time_windows):
            index = manager.NodeToIndex(location_idx)
            if index >= 0:
                time_dimension.CumulVar(index).SetRange(int(start_time), int(end_time))

        # Also set start/end time windows for depots
        for vehicle_id in range(self.num_vehicles):
            depot_index = routing.Start(vehicle_id)
            time_dimension.CumulVar(depot_index).SetRange(
                int(self.time_windows[self.starts[vehicle_id]][0]),
                int(self.time_windows[self.starts[vehicle_id]][1])
            )
            depot_end_index = routing.End(vehicle_id)
            time_dimension.CumulVar(depot_end_index).SetRange(
                int(self.time_windows[self.ends[vehicle_id]][0]),
                int(self.time_windows[self.ends[vehicle_id]][1])
            )

        return routing, manager

    # ======= NEW HELPERS FOR GA: clustering + active vehicle management ======
    def _assign_customers_to_nearest_depots(self):
        """Assign each customer to its nearest depot (returns dict depot_idx -> list of customer node indices).
        Customer/node indices are the indexes within all_locations (i.e., depots are 0..num_depots-1, customers num_depots..)
        """
        groups = {d: [] for d in range(self.num_depots)}
        for cust_local_idx in range(self.num_depots, self.num_depots + self.num_customers):
            cust_coord = self.all_locations[cust_local_idx]
            nearest = min(range(self.num_depots), key=lambda d: self.distance_matrix[cust_local_idx][d])
            groups[nearest].append(cust_local_idx)
        # Remove empty groups
        groups = {d: groups[d] for d in groups if len(groups[d]) > 0}
        return groups

    def _create_active_fleet_from_groups(self, groups):
        """Return list of (depot_idx, vehicle_slots) for depots that have customers.
        vehicle_slots = number of vehicles to allocate for that depot (<= self.num_vehicles_per_depot)
        Strategy: allocate at least 1 vehicle per active depot, and up to num_vehicles_per_depot; if a depot has many customers, allocate full slots.
        """
        active = []
        for depot_idx, custs in groups.items():
            # --- SIMPLIFIED/FIXED LOGIC ---
            # Just use the max available vehicles per depot if it has customers.
            # The GA's new decoder will handle optimizing *how many* are actually used.
            vehicles_needed = self.num_vehicles_per_depot

            active.append((depot_idx, vehicles_needed))
        return active

    def _split_customers_into_routes_for_active_fleet(self, customers_sequence, active_fleet):
        """
        (!!!) GA DECODER FUNCTION (FIXED) (!!!)
        Distribute customers_sequence across active_fleet vehicles using a sequential
        insertion heuristic that respects capacity and time windows.
        This function *naturally* optimizes the number of vehicles used.

        customers_sequence: list of ALL customer node indices (in all_locations indexing)
        active_fleet: list of tuples (depot_idx, vehicle_count)

        Returns: list of routes where each route is [depot_idx, ...customers..., depot_idx]
        """
        # Build mapping depot -> list of customers (preserve order in customers_sequence)
        depot_to_customers = {depot: [] for depot, _ in active_fleet}

        # Cluster customers to their nearest *active* depot
        active_depot_indices = [d for d, _ in active_fleet]
        if not active_depot_indices:
            return []  # No active depots, no routes

        for c in customers_sequence:
            nearest = min(active_depot_indices, key=lambda d: self.distance_matrix[c][d])
            depot_to_customers[nearest].append(c)

        routes = []

        for depot_idx, vehicle_count in active_fleet:
            # Get customers for this depot, in the order defined by the chromosome
            assigned = depot_to_customers.get(depot_idx, [])
            if not assigned:
                continue  # No customers for this depot

            vehicle_local_idx = 0
            # Keep creating routes while there are customers AND vehicles available
            while assigned and vehicle_local_idx < vehicle_count:

                # Get vehicle info (index is relative to all vehicles)
                global_vehicle_idx = (depot_idx * self.num_vehicles_per_depot) + vehicle_local_idx
                if global_vehicle_idx >= len(self.vehicle_capacities):
                    break  # Safety check

                max_cap = self.vehicle_capacities[global_vehicle_idx]
                current_route = [depot_idx]
                current_load = 0
                # Route starts at depot's opening time
                current_time = self.time_windows[depot_idx][0]

                # Customers who couldn't fit in *this* route
                customers_still_assigned = []

                # Iterate through customers assigned to this depot
                for customer_node in assigned:
                    demand = self.demands[customer_node]
                    service_time = self.service_times[customer_node]
                    last_node = current_route[-1]

                    # --- Check Time Windows and Capacity ---
                    time_to_cust = self.time_matrix[last_node][customer_node]
                    time_at_cust_arrival = current_time + time_to_cust
                    # Wait if arriving early
                    time_at_cust_service_start = max(self.time_windows[customer_node][0], time_at_cust_arrival)

                    time_at_cust_service_end = time_at_cust_service_start + service_time

                    time_to_depot = self.time_matrix[customer_node][depot_idx]
                    time_at_depot_if_return = time_at_cust_service_end + time_to_depot

                    # --- Check Constraints ---
                    can_serve = True
                    # 1. Capacity
                    if current_load + demand > max_cap:
                        can_serve = False
                    # 2. Customer Time Window (cannot start service after it closes)
                    if time_at_cust_service_start > self.time_windows[customer_node][1]:
                        can_serve = False
                    # 3. Depot Return Time Window (cannot return after it closes)
                    if time_at_depot_if_return > self.time_windows[depot_idx][1]:
                        can_serve = False

                    if can_serve:
                        # Add customer to this route
                        current_route.append(customer_node)
                        current_load += demand
                        current_time = time_at_cust_service_end  # Update time to end of service
                    else:
                        # Customer cannot be served by *this* route, save for next route
                        customers_still_assigned.append(customer_node)

                # Done with this route, update the list of remaining customers
                assigned = customers_still_assigned

                # Finalize and add route (only if it served at least one customer)
                if len(current_route) > 1:
                    current_route.append(depot_idx)
                    routes.append(current_route)

                vehicle_local_idx += 1  # Use next vehicle

        # Any customers left in 'assigned' (from any depot) are unserved.
        # The _evaluate_routes function will find and penalize this.
        return routes

    # ============= GENETIC ALGORITHM (RESTRUCTURED) =============

    def genetic_algorithm_mdvrp(self, population_size=50, generations=100,
                                mutation_rate=0.15, time_limit=45, max_active_depots=None):
        """
        (!!!) GA (FIXED) (!!!)
        - Chromosome is now a 1D sequence of customers.
        - Fitness is evaluated by decoding the sequence into routes and applying penalties.
        - This optimizes vehicle count and route density.
        """
        start_time = time.time()

        # Pre-assign customers to nearest depots
        groups = self._assign_customers_to_nearest_depots()
        active_fleet = self._create_active_fleet_from_groups(groups)

        # If user specified cap on active depots, reduce
        if max_active_depots and len(active_fleet) > max_active_depots:
            # choose largest groups to remain active
            active_fleet = sorted(active_fleet, key=lambda x: len(groups[x[0]]), reverse=True)[:max_active_depots]

        # Initialize population (list of customer sequences)
        population = self._initialize_ga_population_seq(population_size)

        # Evaluate initial population
        pop_with_fitness = []
        for seq in population:
            fitness = self._evaluate_sequence(seq, active_fleet)
            pop_with_fitness.append((seq, fitness))

        best_overall_seq, best_fitness = min(pop_with_fitness, key=lambda x: x[1])

        generation = 0
        for generation in range(generations):
            if time.time() - start_time > time_limit:
                break

            # Evaluate fitness scores (lower distance = higher fitness)
            fitness_scores = [1.0 / (fit + 1e-6) for seq, fit in pop_with_fitness]

            # Selection (Tournament)
            selected_seqs = self._tournament_selection_seq(population, fitness_scores, population_size)

            # Crossover + Mutation
            new_population = []
            for i in range(0, len(selected_seqs), 2):
                parent1_seq = selected_seqs[i]
                parent2_seq = selected_seqs[i + 1] if i + 1 < len(selected_seqs) else selected_seqs[0]

                # Crossover (on sequences)
                child1_seq, child2_seq = self._order_crossover_seq(parent1_seq, parent2_seq)

                # Mutation (on sequences)
                if random.random() < mutation_rate:
                    child1_seq = self._swap_mutation_seq(child1_seq)
                if random.random() < mutation_rate:
                    child2_seq = self._swap_mutation_seq(child2_seq)

                new_population.extend([child1_seq, child2_seq])

            population = new_population[:population_size]  # Truncate to population size

            # Evaluate new population
            pop_with_fitness = []
            for seq in population:
                fitness = self._evaluate_sequence(seq, active_fleet)
                pop_with_fitness.append((seq, fitness))

            # Track best
            current_best_seq, current_best_fitness = min(pop_with_fitness, key=lambda x: x[1])

            if current_best_fitness < best_fitness:
                best_overall_seq = deepcopy(current_best_seq)
                best_fitness = current_best_fitness
                print(f"  Gen {generation}: Distance = {best_fitness:.2f}")

        elapsed = time.time() - start_time

        # Final step: decode the best sequence into routes
        final_routes_list = self._split_customers_into_routes_for_active_fleet(best_overall_seq, active_fleet)

        # Convert to standard output format
        routes = self._convert_ga_to_routes(final_routes_list)

        # Recalculate true distance (without penalties)
        true_distance = sum(r['distance'] for r in routes)

        return {
            'status': 'success',
            'strategy': 'GENETIC_ALGORITHM',
            'total_distance': true_distance,  # Report true distance
            'fitness_with_penalty': best_fitness,  # For debugging
            'routes': routes,
            'elapsed_time': elapsed,
            'num_routes': len(routes),
            'generations': generation + 1
        }

    def _initialize_ga_population_seq(self, size):
        """Generate initial population of customer sequences."""
        population = []
        # Flatten all customers into list of customer node indices
        customers = list(self.all_customer_indices)

        for _ in range(size):
            random.shuffle(customers)
            population.append(customers[:])  # Add a copy

        return population

    def _evaluate_sequence(self, customer_sequence, active_fleet):
        """Helper to decode a sequence and evaluate its routes."""
        # 1. Decode sequence into routes
        routes = self._split_customers_into_routes_for_active_fleet(customer_sequence, active_fleet)
        # 2. Evaluate routes (calculates distance + penalty)
        return self._evaluate_routes(routes)

    def _tournament_selection_seq(self, population_seqs, fitness_scores, tournament_size=5):
        """Tournament selection on sequences."""
        selected = []
        for _ in range(len(population_seqs)):
            tournament_idx = random.sample(range(len(population_seqs)), min(tournament_size, len(population_seqs)))
            # Higher fitness score is better
            best_idx = max(tournament_idx, key=lambda i: fitness_scores[i])
            selected.append(deepcopy(population_seqs[best_idx]))
        return selected

    def _order_crossover_seq(self, parent1_seq, parent2_seq):
        """Order Crossover (OX) on customer sequences."""
        size = len(parent1_seq)
        if size < 2:
            return deepcopy(parent1_seq), deepcopy(parent2_seq)

        child1_seq, child2_seq = [-1] * size, [-1] * size

        # Choose crossover points
        cut1, cut2 = sorted(random.sample(range(size), 2))

        # Copy segment from parent 1 to child 1
        child1_seq[cut1:cut2] = parent1_seq[cut1:cut2]

        # Copy segment from parent 2 to child 2
        child2_seq[cut1:cut2] = parent2_seq[cut1:cut2]

        # Fill remaining for child 1
        p2_idx, c1_idx = 0, 0
        while c1_idx < size:
            if c1_idx == cut1:  # Skip over the copied segment
                c1_idx = cut2
            if c1_idx >= size:
                break

            if parent2_seq[p2_idx] not in child1_seq:
                child1_seq[c1_idx] = parent2_seq[p2_idx]
                c1_idx += 1
            p2_idx += 1

        # Fill remaining for child 2
        p1_idx, c2_idx = 0, 0
        while c2_idx < size:
            if c2_idx == cut1:  # Skip over the copied segment
                c2_idx = cut2
            if c2_idx >= size:
                break

            if parent1_seq[p1_idx] not in child2_seq:
                child2_seq[c2_idx] = parent1_seq[p1_idx]
                c2_idx += 1
            p1_idx += 1

        return child1_seq, child2_seq

    def _swap_mutation_seq(self, sequence):
        """Mutation: Swap two customers in the sequence."""
        mutated = deepcopy(sequence)
        if len(mutated) >= 2:
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    def _extract_customers_sequence(self, routes):
        """(DEPRECATED, but kept for reference) Extract customer sequence from routes"""
        sequence = []
        for route in routes:
            for node in route:
                if node >= self.num_depots:
                    sequence.append(node)
        return sequence

    def _evaluate_routes(self, routes):
        """
        (!!!) GA FITNESS FUNCTION (FIXED) (!!!)
        Calculate total distance of routes + PENALTY for unserved customers.
        """
        total_distance = 0
        served_customers = set()

        for route in routes:
            # ignore trivial routes like [depot, depot]
            if len(route) <= 2:
                continue
            for i in range(len(route) - 1):
                total_distance += self.distance_matrix[route[i]][route[i + 1]]

            # Track served customers
            for node in route:
                if node in self.all_customer_indices:
                    served_customers.add(node)

        unserved_customers = self.all_customer_indices - served_customers
        unserved_count = len(unserved_customers)

        # Apply the pre-calculated large penalty for each unserved customer
        penalty = unserved_count * self._max_penalty

        return total_distance + penalty

    def _convert_ga_to_routes(self, ga_routes):
        """Convert GA routes list (from decoder) to standard output format"""
        routes = []
        total_distance = 0

        # We need to assign a unique vehicle_id
        # The GA routes don't have a persistent ID, so we just enumerate them
        vehicle_id_counter = 0
        for route in ga_routes:
            # route example: [depot_idx, cust1, cust2, depot_idx]
            if len(route) > 2:
                distance = sum(self.distance_matrix[route[i]][route[i + 1]]
                               for i in range(len(route) - 1))
                routes.append({
                    'vehicle_id': vehicle_id_counter,
                    'depot': route[0],
                    'route': route,
                    'distance': distance
                })
                total_distance += distance
                vehicle_id_counter += 1

        return routes

    # ============= 3-OPT OPTIMIZATION (FULL) =============

    def three_opt_optimization(self, route, max_iterations=500):
        """
        Full 3-opt Local Search
        (Note: This is a pure distance optimization and does not re-check
        time windows or capacity constraints. It is suitable for post-processing
        routes that are already feasible.)
        """
        best_route = route
        best_distance = self._calculate_route_distance(best_route)
        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Need at least 7 points for a 3-opt (depot, c1, c2, c3, c4, c5, depot)
            # 3 edges to break: (i-1, i), (j-1, j), (k-1, k)
            # Min indices: i=1, j=3, k=5. route[k] = route[5]
            if len(best_route) < 7:
                break  # Not enough nodes to perform 3-opt

            n = len(best_route)
            # i iterates from 1 (after depot) up to n-5
            # j iterates from i+2 (skip at least one node) up to n-3
            # k iterates from j+2 (skip at least one node) up to n-1 (before last depot)
            for i in range(1, n - 4):
                if improved: break
                for j in range(i + 2, n - 2):
                    if improved: break
                    for k in range(j + 2, n):
                        if improved: break

                        # The 3 edges we are breaking:
                        # (i-1) -> i
                        # (j-1) -> j
                        # (k-1) -> k
                        A, B = best_route[i - 1], best_route[i]
                        C, D = best_route[j - 1], best_route[j]
                        E, F = best_route[k - 1], best_route[k]

                        d = self.distance_matrix

                        # Current distance for the 3 edges
                        d0 = d[A][B] + d[C][D] + d[E][F]

                        # --- Try all 7 new combinations ---

                        # Case 1 (2-opt): A-B C-E D-F (reverse seg D...E)
                        d1 = d[A][B] + d[C][E] + d[D][F]
                        if d1 < d0:
                            best_route = best_route[:j] + best_route[k - 1:j - 1:-1] + best_route[k:]
                            best_distance = self._calculate_route_distance(best_route)
                            d0 = best_distance
                            improved = True
                            continue  # Restart loops

                        # Case 2 (2-opt): A-C B-D E-F (reverse seg B...C)
                        d2 = d[A][C] + d[B][D] + d[E][F]
                        if d2 < d0:
                            best_route = best_route[:i] + best_route[j - 1:i - 1:-1] + best_route[j:]
                            best_distance = self._calculate_route_distance(best_route)
                            d0 = best_distance
                            improved = True
                            continue

                        # Case 3 (3-opt): A-D E-B C-F (swap seg B...C and D...E)
                        d3 = d[A][D] + d[E][B] + d[C][F]
                        if d3 < d0:
                            best_route = best_route[:i] + best_route[j:k] + best_route[i:j] + best_route[k:]
                            best_distance = self._calculate_route_distance(best_route)
                            d0 = best_distance
                            improved = True
                            continue

                        # Case 4 (3-opt): A-D E-C B-F (swap; reverse seg B...C)
                        d4 = d[A][D] + d[E][C] + d[B][F]
                        if d4 < d0:
                            best_route = best_route[:i] + best_route[j:k] + best_route[j - 1:i - 1:-1] + best_route[k:]
                            best_distance = self._calculate_route_distance(best_route)
                            d0 = best_distance
                            improved = True
                            continue

                        # Case 5 (3-opt): A-E D-B C-F (swap; reverse seg D...E)
                        d5 = d[A][E] + d[D][B] + d[C][F]
                        if d5 < d0:
                            best_route = best_route[:i] + best_route[k - 1:j - 1:-1] + best_route[i:j] + best_route[k:]
                            best_distance = self._calculate_route_distance(best_route)
                            d0 = best_distance
                            improved = True
                            continue

                        # Case 6 (3-opt): A-C B-E D-F (reverse seg B...C and D...E)
                        d6 = d[A][C] + d[B][E] + d[D][F]
                        if d6 < d0:
                            best_route = best_route[:i] + best_route[j - 1:i - 1:-1] + best_route[k - 1:j - 1:-1] + best_route[k:]
                            best_distance = self._calculate_route_distance(best_route)
                            d0 = best_distance
                            improved = True
                            continue

                        # Case 7 (3-opt): A-E D-C B-F (swap; reverse all)
                        d7 = d[A][E] + d[D][C] + d[B][F]
                        if d7 < d0:
                            best_route = best_route[:i] + best_route[k - 1:j - 1:-1] + best_route[j - 1:i - 1:-1] + best_route[k:]
                            best_distance = self._calculate_route_distance(best_route)
                            d0 = best_distance
                            improved = True
                            continue

        return best_route, best_distance, iteration


    def apply_3opt_to_routes(self, routes):
        """Apply 3-opt to all routes"""
        print(f"    → Starting 3-opt on {len(routes)} routes...")
        optimized_routes = []
        total_improvement = 0

        for idx, route_info in enumerate(routes):
            original_distance = route_info['distance']

            # Skip optimization for trivial routes
            if len(route_info['route']) < 7:
                print(f"      Route {idx + 1}: skipping (too short for 3-opt)")
                optimized_routes.append(route_info)
                continue

            print(f"      Route {idx + 1}: optimizing (original dist: {original_distance:.2f})...", end=" ")

            # Use the 3-opt function
            optimized_route, new_distance, iterations = self.three_opt_optimization(
                route_info['route']
            )

            improvement = original_distance - new_distance
            total_improvement += improvement
            print(f"→ {new_distance:.2f} (improved by {improvement:.2f})")

            optimized_routes.append({
                'vehicle_id': route_info['vehicle_id'],
                'depot': route_info['depot'],
                'route': optimized_route,
                'distance': new_distance,
                'improvement': improvement,
                'method': '3-opt'
            })

        print(f"    → Total 3-opt improvement: {total_improvement:.2f}")
        return optimized_routes, total_improvement

    def _calculate_route_distance(self, route):
        """Calculate total distance of route"""
        total = 0
        for i in range(len(route) - 1):
            total += self.distance_matrix[route[i]][route[i + 1]]
        return total

    # ============= OR-OPT (Relocation) =============
    # (Giữ nguyên, không thay đổi)
    def or_opt_optimization(self, route, segment_size=3, max_iterations=300):
        """
        Or-opt: Relocate segments of nodes
        (Note: Also does not check TW/Capacity constraints)
        """
        improved = True
        best_distance = self._calculate_route_distance(route)
        best_route = route
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Iterate over all possible segment sizes
            for seg_size in range(1, min(segment_size + 1, len(best_route) - 3)):
                # Iterate over all possible start positions (i) for the segment
                # (Must be after depot, must end before last customer)
                for i in range(1, len(best_route) - seg_size - 1):
                    segment = best_route[i:i + seg_size]

                    # Create route without the segment
                    remaining = best_route[:i] + best_route[i + seg_size:]

                    # Iterate over all possible insertion points (j)
                    # (Must be after depot, can be at the end before depot)
                    for j in range(1, len(remaining)):
                        # Avoid re-inserting at the same place
                        if j == i:
                            continue

                        new_route = remaining[:j] + segment + remaining[j:]
                        new_distance = self._calculate_route_distance(new_route)

                        if new_distance < best_distance:
                            best_route = new_route
                            best_distance = new_distance
                            improved = True
                            break

                    if improved:
                        break
                if improved:
                    break
            if improved:
                break  # Restart

        return best_route, best_distance, iteration

    # ============= PARALLEL EXECUTION =============

    def run_strategies_parallel(self, time_limit=45):
        """Run multiple strategies in parallel"""
        start_time = time.time()

        print("\n" + "=" * 80)
        print("RUNNING ADVANCED STRATEGIES - PARALLEL EXECUTION")
        print("=" * 80)

        results = []

        # Strategy 1: GA (FIXED)
        print("\n[1/3] Running Genetic Algorithm (FIXED)...")
        result_ga = self.genetic_algorithm_mdvrp(
            population_size=50,
            generations=200,  # More generations now that fitness is correct
            time_limit=time_limit
        )
        results.append(result_ga)

        # Strategy 2: OR-Tools + 3-opt
        print("[2/3] Running OR-Tools + 3-opt...")
        result_or = self._run_ortools_with_3opt(time_limit)
        results.append(result_or)

        # Strategy 3: GA + 3-opt hybrid
        print("[3/3] Running GA + 3-opt Hybrid...")
        result_hybrid = self.genetic_algorithm_mdvrp(
            population_size=30,
            generations=100,
            time_limit=int(time_limit * 0.4)
        )

        if result_hybrid['status'] == 'success':
            # Apply 3-opt
            opt_routes, improvement = self.apply_3opt_to_routes(result_hybrid['routes'])
            result_hybrid['routes'] = opt_routes
            result_hybrid['total_distance'] = sum(r['distance'] for r in opt_routes)
            result_hybrid['strategy'] = 'GA + 3-OPT_HYBRID'

        results.append(result_hybrid)

        # Compare
        successful = [r for r in results if r['status'] == 'success' and r['total_distance'] > 0]
        if successful:
            best = min(successful, key=lambda x: x['total_distance'])
            print("\n" + "=" * 80)
            print("ADVANCED STRATEGIES COMPARISON")
            print("=" * 80)

            for i, result in enumerate(results, 1):
                if result['status'] == 'success':
                    if result['total_distance'] == 0: continue  # Skip failed/empty

                    gap = ((result['total_distance'] - best['total_distance']) /
                           best['total_distance'] * 100) if best['total_distance'] > 0 else 0
                    marker = "🏆 BEST" if result == best else ""
                    print(f"\nStrategy {i}: {result['strategy']}")
                    print(f"  Distance: {result['total_distance']:.2f}")
                    print(f"  Routes: {result['num_routes']}")
                    print(f"  Time: {result['elapsed_time']:.2f}s")
                    print(f"  Gap: {gap:.2f}% {marker}")

        elapsed = time.time() - start_time
        return {
            'status': 'success',
            'all_results': results,
            'best_result': best if successful else None,
            'total_time': elapsed
        }

    def _run_ortools_with_3opt(self, time_limit=45):
        """OR-Tools + 3-opt optimization"""
        start_time = time.time()
        try:
            print("  → Initializing OR-Tools routing model...")
            routing, manager = self._get_routing_model_with_time_windows()

            print("  → Setting search parameters...")
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
            search_parameters.time_limit.seconds = int(time_limit * 0.6)

            print(f"  → Solving with OR-Tools (time limit: {int(time_limit * 0.6)}s)...")
            solution = routing.SolveWithParameters(search_parameters)
            elapsed = time.time() - start_time

            if solution:
                print(f"  → Solution found! Extracting routes...")
                routes, base_distance = self._extract_routes(routing, manager, solution)
                print(f"  → Base distance: {base_distance:.2f}")

                if not routes:
                    print("  ✗ OR-Tools: Solution found but no routes extracted.")
                    return {
                        'status': 'failed',
                        'strategy': 'OR-TOOLS + 3-OPT',
                        'message': 'No routes extracted from solution',
                        'elapsed_time': elapsed
                    }

                # Apply 3-opt
                print(f"  → Applying 3-opt optimization to {len(routes)} routes...")
                opt_routes, improvement = self.apply_3opt_to_routes(routes)
                new_distance = base_distance - improvement
                print(f"  → 3-opt improvement: {improvement:.2f}")
                print(f"  → Final distance: {new_distance:.2f}")

                # 🧩 Thêm lại coordinates cho route sau khi tối ưu (phòng trường hợp route bị đổi thứ tự)
                for route in opt_routes:
                    route["coordinates"] = [self.all_locations[node] for node in route["route"]]

                return {
                    'status': 'success',
                    'strategy': 'OR-TOOLS + 3-OPT',
                    'total_distance': new_distance,
                    'base_distance': base_distance,
                    'improvement_from_3opt': improvement,
                    'routes': opt_routes,
                    'elapsed_time': elapsed,
                    'num_routes': len(opt_routes)
                }
            else:
                print("  ✗ OR-Tools: No solution found!")
                return {
                    'status': 'failed',
                    'strategy': 'OR-TOOLS + 3-OPT',
                    'message': 'No solution found',
                    'elapsed_time': elapsed
                }

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ✗ OR-Tools ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'failed',
                'strategy': 'OR-TOOLS + 3-OPT',
                'message': str(e),
                'elapsed_time': elapsed
            }

    def _extract_routes(self, routing, manager, solution):
        """Extract routes from solution"""
        routes = []
        total_distance = 0
        time_dimension = routing.GetDimensionOrDie('Time')

        for vehicle_id in range(self.num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)

                previous_index = index
                index = solution.Value(routing.NextVar(index))

                # GetArcCostForVehicle scales by 100
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

                # Add the end node (depot)
            node = manager.IndexToNode(index)
            route.append(node)

            if len(route) > 2:  # Only save non-empty routes
                routes.append({
                    'vehicle_id': vehicle_id,
                    'depot': self.starts[vehicle_id],
                    'route': route,
                    'distance': route_distance / 100.0  # Rescale distance
                })
                total_distance += route_distance

        return routes, total_distance / 100.0


# Export function for backend
def solve_mdvrp_advanced(depots, customers, num_vehicles_per_depot,
                         vehicle_capacities=None, demands=None,
                         time_windows=None, service_times=None,
                         strategy='advanced_benchmark', time_limit=45):
    solver = AdvancedMDVRPSolver(
        depots, customers, num_vehicles_per_depot,
        vehicle_capacities, demands, time_windows, service_times
    )

    if strategy == 'genetic':
        result = solver.genetic_algorithm_mdvrp(time_limit=time_limit)
    elif strategy == '3opt':
        result = solver._run_ortools_with_3opt(time_limit=time_limit)
    elif strategy == 'advanced_benchmark':
        result = solver.run_strategies_parallel(time_limit=time_limit)
    else:
        result = {'status': 'error', 'message': 'Unknown strategy'}

    return result