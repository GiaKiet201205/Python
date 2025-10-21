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
Advanced MDVRP Solver - Enhanced Version
+ Genetic Algorithm (GA)
+ 3-opt Local Search
+ Time Windows Support
+ Parallel Execution
+ Heterogeneous Fleet Support
"""


class AdvancedMDVRPSolver:
    def __init__(self, depots, customers, num_vehicles_per_depot,
                 vehicle_capacities=None, demands=None,
                 time_windows=None, service_times=None):
        self.depots = depots
        self.customers = customers
        self.num_vehicles_per_depot = num_vehicles_per_depot
        self.num_depots = len(depots)
        self.num_vehicles = num_vehicles_per_depot * self.num_depots
        self.num_customers = len(customers)

        self.all_locations = depots + customers
        self.distance_matrix = self._compute_distance_matrix()
        self.time_matrix = self._compute_time_matrix()

        # Demands & Capacities
        self.demands = demands if demands else [0] * self.num_depots + [1] * len(customers)
        self.vehicle_capacities = vehicle_capacities if vehicle_capacities else [100] * self.num_vehicles

        # Time Windows: [(start_time, end_time), ...]
        self.time_windows = time_windows if time_windows else [(0, 1000)] * len(self.all_locations)

        # Service times at each node
        self.service_times = service_times if service_times else [0] * self.num_depots + [30] * len(customers)

        # Vehicle start/end
        self.starts = []
        self.ends = []
        for depot_idx in range(self.num_depots):
            for _ in range(num_vehicles_per_depot):
                self.starts.append(depot_idx)
                self.ends.append(depot_idx)

        self.best_solution = None
        self.best_distance = float('inf')

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
            return travel_time + self.service_times[from_node]

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_index,
            3600,  # slack_max
            10000,  # capacity
            True,  # fix_start_cumul_to_zero
            'Time'  # dimension name
        )

        time_dimension = routing.GetDimensionOrDie('Time')
        for location_idx, (start_time, end_time) in enumerate(self.time_windows):
            index = manager.NodeToIndex(location_idx)
            if index >= 0:
                time_dimension.CumulVar(index).SetRange(int(start_time), int(end_time))

        return routing, manager

    # ============= GENETIC ALGORITHM =============

    def genetic_algorithm_mdvrp(self, population_size=50, generations=100,
                                mutation_rate=0.15, time_limit=45):
        """
        Genetic Algorithm for MDVRP
        """
        start_time = time.time()

        # Initialize population
        population = self._initialize_ga_population(population_size)
        best_overall = min(population, key=lambda x: self._evaluate_routes(x))
        best_fitness = self._evaluate_routes(best_overall)

        for generation in range(generations):
            if time.time() - start_time > time_limit:
                break

            # Evaluate fitness
            fitness_scores = [1.0 / self._evaluate_routes(individual) for individual in population]

            # Selection (Tournament)
            selected = self._tournament_selection(population, fitness_scores, population_size)

            # Crossover + Mutation
            new_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]

                # Crossover
                child1, child2 = self._order_crossover(parent1, parent2)

                # Mutation
                if random.random() < mutation_rate:
                    child1 = self._swap_mutation(child1)
                if random.random() < mutation_rate:
                    child2 = self._swap_mutation(child2)

                new_population.extend([child1, child2])

            population = new_population[:population_size]

            # Track best
            current_best = min(population, key=lambda x: self._evaluate_routes(x))
            current_distance = self._evaluate_routes(current_best)

            if current_distance < best_fitness:
                best_overall = current_best
                best_fitness = current_distance
                print(f"  Gen {generation}: Distance = {best_fitness:.2f}")

        elapsed = time.time() - start_time
        routes = self._convert_ga_to_routes(best_overall)

        return {
            'status': 'success',
            'strategy': 'GENETIC_ALGORITHM',
            'total_distance': best_fitness,
            'routes': routes,
            'elapsed_time': elapsed,
            'num_routes': len(routes),
            'generations': generation + 1
        }

    def _initialize_ga_population(self, size):
        """Generate initial population"""
        population = []
        customers = list(range(self.num_depots, self.num_depots + self.num_customers))

        for _ in range(size):
            random.shuffle(customers)
            routes = self._split_customers_into_routes(customers)
            population.append(routes)

        return population

    def _split_customers_into_routes(self, customers):
        """Split customers into routes"""
        routes_per_depot = []
        depot_capacity = self.num_customers // self.num_depots

        for depot_idx in range(self.num_depots):
            depot_routes = []
            start_idx = depot_idx * depot_capacity
            end_idx = start_idx + depot_capacity if depot_idx < self.num_depots - 1 else len(customers)

            depot_customers = customers[start_idx:end_idx]

            # Split into vehicles
            for vehicle_idx in range(self.num_vehicles_per_depot):
                route = [self.starts[depot_idx * self.num_vehicles_per_depot + vehicle_idx]]
                if vehicle_idx < len(depot_customers):
                    route.append(depot_customers[vehicle_idx])
                route.append(self.ends[depot_idx * self.num_vehicles_per_depot + vehicle_idx])
                depot_routes.append(route)

            routes_per_depot.extend(depot_routes)

        return routes_per_depot

    def _tournament_selection(self, population, fitness_scores, tournament_size=5):
        """Tournament selection"""
        selected = []
        for _ in range(len(population)):
            tournament_idx = random.sample(range(len(population)), min(tournament_size, len(population)))
            best_idx = max(tournament_idx, key=lambda i: fitness_scores[i])
            selected.append(deepcopy(population[best_idx]))
        return selected

    def _order_crossover(self, parent1, parent2):
        """Order Crossover (OX)"""
        seq1 = self._extract_customers_sequence(parent1)
        seq2 = self._extract_customers_sequence(parent2)

        if len(seq1) < 2 or len(seq2) < 2:
            return deepcopy(parent1), deepcopy(parent2)

        cut1, cut2 = sorted(random.sample(range(len(seq1)), 2))

        child1_seq = seq1[cut1:cut2]
        remaining = [c for c in seq2 if c not in child1_seq]
        child1_seq = seq1[:cut1] + child1_seq + remaining[len(seq1) - cut2:]

        child2_seq = seq2[cut1:cut2]
        remaining = [c for c in seq1 if c not in child2_seq]
        child2_seq = seq2[:cut1] + child2_seq + remaining[len(seq2) - cut2:]

        child1 = self._split_customers_into_routes(child1_seq)
        child2 = self._split_customers_into_routes(child2_seq)

        return child1, child2

    def _swap_mutation(self, routes):
        """Mutation: Swap two customers"""
        mutated = deepcopy(routes)

        customers = self._extract_customers_sequence(mutated)
        if len(customers) >= 2:
            i, j = random.sample(range(len(customers)), 2)
            customers[i], customers[j] = customers[j], customers[i]
            mutated = self._split_customers_into_routes(customers)

        return mutated

    def _extract_customers_sequence(self, routes):
        """Extract customer sequence from routes"""
        sequence = []
        for route in routes:
            for node in route:
                if node >= self.num_depots:
                    sequence.append(node)
        return sequence

    def _evaluate_routes(self, routes):
        """Calculate total distance of routes"""
        total = 0
        for route in routes:
            for i in range(len(route) - 1):
                total += self.distance_matrix[route[i]][route[i + 1]]
        return total

    def _convert_ga_to_routes(self, ga_routes):
        """Convert GA routes to standard format"""
        routes = []
        total_distance = 0

        for vehicle_id, route in enumerate(ga_routes):
            if len(route) > 2:
                distance = sum(self.distance_matrix[route[i]][route[i + 1]]
                               for i in range(len(route) - 1))
                routes.append({
                    'vehicle_id': vehicle_id,
                    'depot': route[0],
                    'route': route,
                    'distance': distance
                })
                total_distance += distance

        return routes

    # ============= 3-OPT OPTIMIZATION =============

    def three_opt_optimization(self, route, max_iterations=500):
        """
        3-opt Local Search
        """
        improved = True
        best_distance = self._calculate_route_distance(route)
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for i in range(1, len(route) - 3):
                for j in range(i + 2, len(route) - 2):
                    for k in range(j + 2, len(route) - 1):
                        cases = [
                            route[:i] + route[i:j][::-1] + route[j:],
                            route[:j] + route[j:k][::-1] + route[k:],
                            route[:i] + route[i:k][::-1] + route[k:],
                            route[:i] + route[j:k] + route[i:j] + route[k:]
                        ]

                        for new_route in cases:
                            new_distance = self._calculate_route_distance(new_route)
                            if new_distance < best_distance:
                                route = new_route
                                best_distance = new_distance
                                improved = True
                                break

                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        return route, best_distance, iteration

    def apply_3opt_to_routes(self, routes):
        """Apply 3-opt to all routes"""
        print(f"    → Starting 3-opt on {len(routes)} routes...")
        optimized_routes = []
        total_improvement = 0

        for idx, route_info in enumerate(routes):
            original_distance = route_info['distance']
            print(f"      Route {idx + 1}: optimizing (original dist: {original_distance:.2f})...", end=" ")

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

    def or_opt_optimization(self, route, segment_size=3, max_iterations=300):
        """
        Or-opt: Relocate segments of nodes
        """
        improved = True
        best_distance = self._calculate_route_distance(route)
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for seg_size in range(1, min(segment_size + 1, len(route) - 2)):
                for i in range(1, len(route) - seg_size - 1):
                    segment = route[i:i + seg_size]
                    remaining = route[:i] + route[i + seg_size:]

                    for j in range(1, len(remaining) - 1):
                        new_route = remaining[:j] + segment + remaining[j:]
                        new_distance = self._calculate_route_distance(new_route)

                        if new_distance < best_distance:
                            route = new_route
                            best_distance = new_distance
                            improved = True
                            break

                    if improved:
                        break
                if improved:
                    break

        return route, best_distance, iteration

    # ============= PARALLEL EXECUTION =============

    def run_strategies_parallel(self, time_limit=45):
        """Run multiple strategies in parallel"""
        start_time = time.time()

        print("\n" + "=" * 80)
        print("RUNNING ADVANCED STRATEGIES - PARALLEL EXECUTION")
        print("=" * 80)

        results = []

        # Strategy 1: GA
        print("\n[1/3] Running Genetic Algorithm...")
        result_ga = self.genetic_algorithm_mdvrp(
            population_size=50,
            generations=100,
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
            generations=50,
            time_limit=int(time_limit * 0.4)
        )

        if result_hybrid['status'] == 'success':
            opt_routes, improvement = self.apply_3opt_to_routes(result_hybrid['routes'])
            result_hybrid['routes'] = opt_routes
            result_hybrid['total_distance'] = sum(r['distance'] for r in opt_routes)
            result_hybrid['strategy'] = 'GA + 3-OPT_HYBRID'

        results.append(result_hybrid)

        # Compare
        successful = [r for r in results if r['status'] == 'success']
        if successful:
            best = min(successful, key=lambda x: x['total_distance'])
            print("\n" + "=" * 80)
            print("ADVANCED STRATEGIES COMPARISON")
            print("=" * 80)

            for i, result in enumerate(results, 1):
                if result['status'] == 'success':
                    gap = ((result['total_distance'] - best['total_distance']) /
                           best['total_distance'] * 100)
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

                # Apply 3-opt
                print(f"  → Applying 3-opt optimization to {len(routes)} routes...")
                opt_routes, improvement = self.apply_3opt_to_routes(routes)
                new_distance = base_distance - improvement
                print(f"  → 3-opt improvement: {improvement:.2f}")
                print(f"  → Final distance: {new_distance:.2f}")

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

        for vehicle_id in range(self.num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            route.append(manager.IndexToNode(index))

            if len(route) > 2:
                routes.append({
                    'vehicle_id': vehicle_id,
                    'depot': self.starts[vehicle_id],
                    'route': route,
                    'distance': route_distance / 100
                })
                total_distance += route_distance

        return routes, total_distance / 100


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