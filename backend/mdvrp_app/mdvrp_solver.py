from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
import time
from typing import List, Dict, Tuple
import json

"""
Enhanced MDVRP Solver with 3 Optimization Strategies
- Strategy 1: PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH
- Strategy 2: PATH_MOST_CONSTRAINED_ARC + SIMULATED_ANNEALING
- Strategy 3: NEAREST_NEIGHBOR + TABU_SEARCH
+ 2-opt Post-Optimization
+ Benchmark & Best-Known Comparison
"""


class MDVRPSolver:
    def __init__(self, depots, customers, num_vehicles_per_depot,
                 vehicle_capacities=None, demands=None):
        self.depots = depots
        self.customers = customers
        self.num_vehicles_per_depot = num_vehicles_per_depot
        self.num_depots = len(depots)
        self.num_vehicles = num_vehicles_per_depot * self.num_depots

        self.all_locations = depots + customers
        self.distance_matrix = self._compute_distance_matrix()

        # Demands và capacities
        self.demands = demands if demands else [0] * self.num_depots + [1] * len(customers)
        self.vehicle_capacities = vehicle_capacities if vehicle_capacities else [100] * self.num_vehicles

        # Starts và ends
        self.starts = []
        self.ends = []
        for depot_idx in range(self.num_depots):
            for _ in range(num_vehicles_per_depot):
                self.starts.append(depot_idx)
                self.ends.append(depot_idx)

        self.benchmark_results = {}

    def _compute_distance_matrix(self):
        """Tính ma trận khoảng cách Euclidean"""
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

    def _get_routing_model(self):
        """Tạo routing model cơ bản"""
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

        return routing, manager

    def _extract_routes(self, routing, manager, solution):
        """Trích xuất routes từ solution"""
        routes = []
        total_distance = 0

        for vehicle_id in range(self.num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                lat, lng = self.all_locations[node]
                route.append({
                    "id": node,
                    "lat": lat,
                    "lng": lng
                })
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

                # Thêm điểm kết thúc (về lại depot)
                node = manager.IndexToNode(index)
                lat, lng = self.all_locations[node]
                route.append({
                    "id": node,
                    "lat": lat,
                    "lng": lng
                })

            if len(route) > 2:
                routes.append({
                    'vehicle_id': vehicle_id,
                    'depot': self.starts[vehicle_id],
                    'route': route,
                    'distance': route_distance / 100
                })
                total_distance += route_distance

        return routes, total_distance / 100

    def strategy_1_cheapest_arc_gls(self, time_limit=45):
        """
        Strategy 1: PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH
        Phù hợp với bài toán lớn, cân bằng giữa chất lượng và tốc độ
        """
        start_time = time.time()
        routing, manager = self._get_routing_model()

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = time_limit

        solution = routing.SolveWithParameters(search_parameters)
        elapsed = time.time() - start_time

        if solution:
            routes, total_distance = self._extract_routes(routing, manager, solution)
            return {
                'status': 'success',
                'strategy': 'PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH',
                'total_distance': total_distance,
                'routes': routes,
                'elapsed_time': elapsed,
                'num_routes': len(routes)
            }
        else:
            return {
                'status': 'failed',
                'strategy': 'PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH',
                'message': 'No solution found',
                'elapsed_time': elapsed
            }

    def strategy_2_constrained_sa(self, time_limit=45):
        """
        Strategy 2: PATH_MOST_CONSTRAINED_ARC + SIMULATED_ANNEALING
        Tốt hơn cho bài toán có ràng buộc phức tạp
        """
        start_time = time.time()
        routing, manager = self._get_routing_model()

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
        )
        search_parameters.time_limit.seconds = time_limit

        solution = routing.SolveWithParameters(search_parameters)
        elapsed = time.time() - start_time

        if solution:
            routes, total_distance = self._extract_routes(routing, manager, solution)
            return {
                'status': 'success',
                'strategy': 'PATH_MOST_CONSTRAINED_ARC + SIMULATED_ANNEALING',
                'total_distance': total_distance,
                'routes': routes,
                'elapsed_time': elapsed,
                'num_routes': len(routes)
            }
        else:
            return {
                'status': 'failed',
                'strategy': 'PATH_MOST_CONSTRAINED_ARC + SIMULATED_ANNEALING',
                'message': 'No solution found',
                'elapsed_time': elapsed
            }

    def strategy_3_nearest_neighbor_tabu(self, time_limit=45):
        """Strategy 3: AUTOMATIC + TABU_SEARCH
        Tốt nhất cho lần chạy nhanh, thường cho kết quả khá tốt"""
        start_time = time.time()
        try:
            routing, manager = self._get_routing_model()

            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            # Thay NEAREST_NEIGHBOR bằng AUTOMATIC (tương đương)
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
            )
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
            )
            search_parameters.time_limit.seconds = time_limit

            # FIX: Thêm error handling cho TABU_SEARCH
            try:
                solution = routing.SolveWithParameters(search_parameters)
            except Exception as tabu_error:
                print(f"⚠️ TABU_SEARCH lỗi, fallback to GUIDED_LOCAL_SEARCH: {str(tabu_error)}")
                search_parameters.local_search_metaheuristic = (
                    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
                )
                solution = routing.SolveWithParameters(search_parameters)

            elapsed = time.time() - start_time

            if solution:
                routes, total_distance = self._extract_routes(routing, manager, solution)
                return {
                    'status': 'success',
                    'strategy': 'AUTOMATIC + TABU_SEARCH',
                    'total_distance': total_distance,
                    'routes': routes,
                    'elapsed_time': elapsed,
                    'num_routes': len(routes)
                }
            else:
                return {
                    'status': 'failed',
                    'strategy': 'AUTOMATIC + TABU_SEARCH',
                    'message': 'No solution found',
                    'elapsed_time': elapsed
                }
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Strategy 3 Error: {str(e)}")
            import traceback
            traceback.print_exc()

            return {
                'status': 'failed',
                'strategy': 'AUTOMATIC + TABU_SEARCH',
                'message': f'Error: {str(e)}',
                'elapsed_time': elapsed,
                'error_type': type(e).__name__
            }


    def _two_opt_optimization(self, route, distance_matrix, max_iterations=1000):
        """
        2-opt Local Search Post-Optimization
        Cải thiện route bằng cách thử đảo ngược các cung
        """
        improved = True
        best_distance = self._calculate_route_distance(route, distance_matrix)
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    if j - i == 1:
                        continue

                    # Tạo route mới bằng cách đảo ngược cung [i:j]
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    new_distance = self._calculate_route_distance(new_route, distance_matrix)

                    if new_distance < best_distance:
                        route = new_route
                        best_distance = new_distance
                        improved = True
                        break

                if improved:
                    break

        return route, best_distance, iteration

    def _calculate_route_distance(self, route, distance_matrix):
        """Tính tổng khoảng cách của route"""
        total = 0
        for i in range(len(route) - 1):
            total += distance_matrix[route[i]][route[i + 1]]
        return total

    def apply_2opt_to_routes(self, routes):
        """Áp dụng 2-opt optimization cho tất cả routes"""
        optimized_routes = []
        total_improvement = 0

        for route_info in routes:
            original_distance = route_info['distance']
            optimized_route, new_distance, iterations = self._two_opt_optimization(
                route_info['route'],
                self.distance_matrix
            )

            improvement = original_distance - new_distance
            total_improvement += improvement

            optimized_routes.append({
                'vehicle_id': route_info['vehicle_id'],
                'depot': route_info['depot'],
                'route': optimized_route,
                'distance': new_distance,
                'original_distance': original_distance,
                'improvement': improvement,
                'iterations': iterations
            })

        return optimized_routes, total_improvement

    def benchmark_all_strategies(self, time_limit=45):
        """
        Chạy tất cả 3 chiến lược và so sánh kết quả
        """
        results = []

        print("\n" + "=" * 80)
        print("RUNNING BENCHMARK - 3 STRATEGIES COMPARISON")
        print("=" * 80)

        # Strategy 1
        print("\n[1/3] Strategy 1: PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH...")
        result1 = self.strategy_1_cheapest_arc_gls(time_limit)
        results.append(result1)
        if result1['status'] == 'success':
            print(
                f"    ✓ Distance: {result1['total_distance']:.2f} | Routes: {result1['num_routes']} | Time: {result1['elapsed_time']:.2f}s")

        # Strategy 2
        print("[2/3] Strategy 2: PATH_MOST_CONSTRAINED_ARC + SIMULATED_ANNEALING...")
        result2 = self.strategy_2_constrained_sa(time_limit)
        results.append(result2)
        if result2['status'] == 'success':
            print(
                f"    ✓ Distance: {result2['total_distance']:.2f} | Routes: {result2['num_routes']} | Time: {result2['elapsed_time']:.2f}s")

        # Strategy 3
        print("[3/3] Strategy 3: NEAREST_NEIGHBOR + TABU_SEARCH...")
        result3 = self.strategy_3_nearest_neighbor_tabu(time_limit)
        results.append(result3)
        if result3['status'] == 'success':
            print(
                f"    ✓ Distance: {result3['total_distance']:.2f} | Routes: {result3['num_routes']} | Time: {result3['elapsed_time']:.2f}s")

        # So sánh kết quả
        successful_results = [r for r in results if r['status'] == 'success']
        if successful_results:
            best_result = min(successful_results, key=lambda x: x['total_distance'])

            print("\n" + "=" * 80)
            print("BENCHMARK RESULTS SUMMARY")
            print("=" * 80)
            for i, result in enumerate(results, 1):
                if result['status'] == 'success':
                    gap = ((result['total_distance'] - best_result['total_distance']) / best_result[
                        'total_distance'] * 100) if best_result['total_distance'] > 0 else 0
                    marker = "🏆 BEST" if result == best_result else ""
                    print(f"\nStrategy {i}: {result['strategy']}")
                    print(f"  Total Distance: {result['total_distance']:.2f}")
                    print(f"  Number of Routes: {result['num_routes']}")
                    print(f"  Time: {result['elapsed_time']:.2f}s")
                    print(f"  Gap from Best: {gap:.2f}% {marker}")

        self.benchmark_results = {
            'timestamp': time.time(),
            'all_results': results,
            'best_result': best_result if successful_results else None
        }

        return results

    def benchmark_with_2opt(self, time_limit=45):
        """
        Benchmark các chiến lược rồi áp dụng 2-opt optimization
        """
        results = self.benchmark_all_strategies(time_limit)

        print("\n" + "=" * 80)
        print("APPLYING 2-OPT POST-OPTIMIZATION")
        print("=" * 80)

        for i, result in enumerate(results):
            if result['status'] == 'success':
                print(f"\nApplying 2-opt to Strategy {i + 1}...")
                optimized_routes, total_improvement = self.apply_2opt_to_routes(result['routes'])

                new_total = sum(r['distance'] for r in optimized_routes)
                old_total = result['total_distance']
                improvement_pct = (total_improvement / old_total * 100) if old_total > 0 else 0

                print(f"  Before 2-opt: {old_total:.2f}")
                print(f"  After 2-opt: {new_total:.2f}")
                print(f"  Total Improvement: {total_improvement:.2f} ({improvement_pct:.2f}%)")

                result['2opt_optimized'] = True
                result['2opt_routes'] = optimized_routes
                result['2opt_total_distance'] = new_total
                result['2opt_improvement'] = total_improvement

    def compare_with_known_solution(self, known_best_distance):
        """
        So sánh kết quả với giải pháp tốt nhất đã biết
        """
        if not self.benchmark_results.get('best_result'):
            print("No benchmark results available")
            return

        best = self.benchmark_results['best_result']

        print("\n" + "=" * 80)
        print("COMPARISON WITH BEST-KNOWN SOLUTION")
        print("=" * 80)
        print(f"Best-Known Distance: {known_best_distance:.2f}")
        print(f"Our Best Distance: {best['total_distance']:.2f}")

        gap = ((best['total_distance'] - known_best_distance) / known_best_distance * 100)
        print(f"Gap: {gap:.2f}%")

        if gap <= 0:
            print("✓ BETTER than best-known solution!")
        elif gap <= 5:
            print("✓ EXCELLENT - within 5% of best-known")
        elif gap <= 10:
            print("✓ GOOD - within 10% of best-known")
        else:
            print("⚠ Consider using different parameters")


# Export function cho backend
def solve_mdvrp_enhanced(depots, customers, num_vehicles_per_depot,
                         vehicle_capacities=None, demands=None,
                         strategy='benchmark', time_limit=45):

    solver = MDVRPSolver(depots, customers, num_vehicles_per_depot,
                         vehicle_capacities, demands)

    if strategy == 'strategy1':
        result = solver.strategy_1_cheapest_arc_gls(time_limit)
    elif strategy == 'strategy2':
        result = solver.strategy_2_constrained_sa(time_limit)
    elif strategy == 'strategy3':
        result = solver.strategy_3_nearest_neighbor_tabu(time_limit)
    elif strategy == 'benchmark':
        results = solver.benchmark_all_strategies(time_limit)
        result = {
            'status': 'success',
            'strategy': 'BENCHMARK_ALL_3_STRATEGIES',
            'results': results,
            'best': min([r for r in results if r['status'] == 'success'],
                        key=lambda x: x['total_distance'])
        }
    elif strategy == 'benchmark_with_2opt':
        solver.benchmark_with_2opt(time_limit)
        result = solver.benchmark_results
    else:
        result = {'status': 'error', 'message': 'Unknown strategy'}

    return result
