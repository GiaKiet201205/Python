# -*- coding: utf-8 -*-
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
Advanced MDVRP Solver - Phiên bản Tối ưu hóa GA (Không Time Windows)

+ Genetic Algorithm (GA) được nâng cấp thành Memetic Algorithm (MA)
+ Thêm toán tử 2-opt (Local Search) bên trong GA
+ Thêm toán tử đột biến Inversion (Đảo ngược)
+ 3-opt Local Search (Giữ nguyên để hậu xử lý)
+ Hỗ trợ đội xe không đồng nhất (Heterogeneous Fleet)
+ Loại bỏ hoàn toàn ràng buộc Cửa sổ Thời gian (Time Windows)
"""


class AdvancedMDVRPSolver:
    def __init__(self, depots, customers, num_vehicles_per_depot,
                 vehicle_capacities=None, demands=None,
                 time_windows=None, service_times=None):  # <--- THAY ĐỔI: Vẫn nhận tham số, nhưng sẽ bỏ qua chúng
        self.depots = depots
        self.customers = customers
        self.num_vehicles_per_depot = num_vehicles_per_depot
        self.num_depots = len(depots)
        self.num_vehicles = num_vehicles_per_depot * self.num_depots
        self.num_customers = len(customers)

        self.all_locations = depots + customers
        self.distance_matrix = self._compute_distance_matrix()

        # <--- THAY ĐỔI: Time matrix không còn cần thiết cho logic chính,
        # nhưng giữ lại để không làm hỏng cấu trúc nếu có phần nào đó gọi
        self.time_matrix = self._compute_time_matrix()

        # Demands & Capacities
        self.demands = demands if demands else [0] * self.num_depots + [1] * len(customers)
        if vehicle_capacities and len(vehicle_capacities) != self.num_vehicles:
            print(
                f"Warning: vehicle_capacities length ({len(vehicle_capacities)}) != num_vehicles ({self.num_vehicles}). Using first capacity for all.")
            self.vehicle_capacities = [vehicle_capacities[0]] * self.num_vehicles
        elif vehicle_capacities:
            self.vehicle_capacities = vehicle_capacities
        else:
            self.vehicle_capacities = [100] * self.num_vehicles

        # <--- THAY ĐỔI: Vô hiệu hóa Time Windows và Service Times
        # Đặt thời gian phục vụ = 0 để chúng không ảnh hưởng đến tính toán (nếu có)
        self.time_windows = None
        self.service_times = [0] * len(self.all_locations)
        # <--- KẾT THÚC THAY ĐỔI

        # Vehicle start/end (for OR-Tools full model)
        self.starts = []
        self.ends = []
        for depot_idx in range(self.num_depots):
            for _ in range(num_vehicles_per_depot):
                self.starts.append(depot_idx)
                self.ends.append(depot_idx)

        self.best_solution = None
        self.best_distance = float('inf')

        self.all_customer_indices = set(range(self.num_depots, self.num_depots + self.num_customers))
        self._max_penalty = self._calculate_max_penalty()

    def _calculate_max_penalty(self):
        """Calculate a large penalty for unserved customers."""
        if not self.all_customer_indices:
            return 1_000_000

        max_penalty = 0
        for cust_idx in self.all_customer_indices:
            nearest_depot = min(range(self.num_depots), key=lambda d: self.distance_matrix[cust_idx][d])
            max_penalty += self.distance_matrix[nearest_depot][cust_idx] * 2

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
        # <--- THAY ĐỔI: Hàm này vẫn trả về distance_matrix, nhưng logic thời gian sẽ không dùng nó
        return self.distance_matrix

    def _get_routing_model(self):  # <--- THAY ĐỔI: Đổi tên (bỏ 'with_time_windows')
        """Routing model (ĐÃ LOẠI BỎ Time Windows)"""
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
            # <--- THAY ĐỔI: Tăng độ chính xác (ví dụ: nhân 1000)
            return int(self.distance_matrix[from_node][to_node] * 1000)

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

        # <--- THAY ĐỔI: Toàn bộ khối "TIME WINDOWS constraint" đã bị XÓA.
        # <--- KẾT THÚC THAY ĐỔI

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
        """
        active = []
        for depot_idx, custs in groups.items():
            vehicles_needed = self.num_vehicles_per_depot
            active.append((depot_idx, vehicles_needed))
        return active

    def _split_customers_into_routes_for_active_fleet(self, customers_sequence, active_fleet):
        """
        (!!!) GA DECODER FUNCTION (ĐÃ LOẠI BỎ TIME WINDOWS) (!!!)
        Phân phối 'customers_sequence' cho các xe trong 'active_fleet'
        chỉ dựa trên RÀNG BUỘC TẢI TRỌNG (CAPACITY).
        """
        depot_to_customers = {depot: [] for depot, _ in active_fleet}

        active_depot_indices = [d for d, _ in active_fleet]
        if not active_depot_indices:
            return []

        for c in customers_sequence:
            nearest = min(active_depot_indices, key=lambda d: self.distance_matrix[c][d])
            depot_to_customers[nearest].append(c)

        routes = []

        for depot_idx, vehicle_count in active_fleet:
            assigned = depot_to_customers.get(depot_idx, [])
            if not assigned:
                continue

            vehicle_local_idx = 0
            while assigned and vehicle_local_idx < vehicle_count:
                global_vehicle_idx = (depot_idx * self.num_vehicles_per_depot) + vehicle_local_idx
                if global_vehicle_idx >= len(self.vehicle_capacities):
                    break

                max_cap = self.vehicle_capacities[global_vehicle_idx]
                current_route = [depot_idx]
                current_load = 0

                # <--- THAY ĐỔI: Loại bỏ biến 'current_time'

                customers_still_assigned = []

                for customer_node in assigned:
                    demand = self.demands[customer_node]

                    # <--- THAY ĐỔI: Toàn bộ logic tính toán thời gian đã bị XÓA
                    # (time_to_cust, time_at_cust_arrival, v.v.)

                    # --- Chỉ kiểm tra Ràng buộc Tải trọng ---
                    can_serve = True

                    # 1. Capacity
                    if current_load + demand > max_cap:
                        can_serve = False

                    # <--- THAY ĐỔI: Ràng buộc 2 (Customer TW) và 3 (Depot TW) đã bị XÓA

                    if can_serve:
                        current_route.append(customer_node)
                        current_load += demand
                        # <--- THAY ĐỔI: Loại bỏ cập nhật 'current_time'
                    else:
                        customers_still_assigned.append(customer_node)

                assigned = customers_still_assigned

                if len(current_route) > 1:
                    current_route.append(depot_idx)
                    routes.append(current_route)

                vehicle_local_idx += 1

        return routes

    # ============= GENETIC ALGORITHM (RESTRUCTURED + MEMETIC) =============

    def genetic_algorithm_mdvrp(self, population_size=100, generations=200,
                                mutation_rate=0.15, crossover_rate=0.8,
                                inversion_rate=0.1, use_memetic_ls=True,
                                time_limit=45, max_active_depots=None):
        """
        (!!!) GA (ĐÃ TỐI ƯU HÓA thành MEMETIC ALGORITHM) (!!!)
        - 'use_memetic_ls=True' sẽ áp dụng 2-opt cho các cá thể mới trong mỗi thế hệ.
        - Thêm đột biến Inversion (đảo ngược).
        """
        start_time = time.time()

        strategy_name = "MEMETIC_ALGORITHM (GA + Internal 2-opt)" if use_memetic_ls else "GENETIC_ALGORITHM (Base)"
        print(f"  → Starting {strategy_name}...")

        groups = self._assign_customers_to_nearest_depots()
        active_fleet = self._create_active_fleet_from_groups(groups)

        if max_active_depots and len(active_fleet) > max_active_depots:
            active_fleet = sorted(active_fleet, key=lambda x: len(groups[x[0]]), reverse=True)[:max_active_depots]

        population = self._initialize_ga_population_seq(population_size)

        pop_with_fitness = []
        for seq in population:
            # <--- THAY ĐỔI: Áp dụng LS (2-opt) ngay cả với quần thể ban đầu
            fitness = self._evaluate_sequence(seq, active_fleet, use_memetic_ls)
            pop_with_fitness.append((seq, fitness))

        best_overall_seq, best_fitness = min(pop_with_fitness, key=lambda x: x[1])

        generation = 0
        for generation in range(generations):
            if time.time() - start_time > time_limit:
                print(f"  Gen {generation}: Time limit reached.")
                break

            fitness_scores = [1.0 / (fit + 1e-6) for seq, fit in pop_with_fitness]

            selected_seqs = self._tournament_selection_seq(population, fitness_scores, population_size)

            new_population = []
            for i in range(0, len(selected_seqs), 2):
                parent1_seq = selected_seqs[i]
                parent2_seq = selected_seqs[i + 1] if i + 1 < len(selected_seqs) else selected_seqs[0]

                # <--- THAY ĐỔI: Thêm Crossover Rate
                if random.random() < crossover_rate:
                    child1_seq, child2_seq = self._order_crossover_seq(parent1_seq, parent2_seq)
                else:
                    # Elitism (giữ lại cha mẹ)
                    child1_seq, child2_seq = deepcopy(parent1_seq), deepcopy(parent2_seq)

                # Đột biến 1: Swap
                if random.random() < mutation_rate:
                    child1_seq = self._swap_mutation_seq(child1_seq)
                if random.random() < mutation_rate:
                    child2_seq = self._swap_mutation_seq(child2_seq)

                # <--- THAY ĐỔI: Thêm Đột biến 2: Inversion (Rất tốt cho VRP)
                if random.random() < inversion_rate:
                    child1_seq = self._inversion_mutation_seq(child1_seq)
                if random.random() < inversion_rate:
                    child2_seq = self._inversion_mutation_seq(child2_seq)

                new_population.extend([child1_seq, child2_seq])

            population = new_population[:population_size]

            pop_with_fitness = []
            for seq in population:
                # <--- THAY ĐỔI: Đánh giá cá thể mới (và áp dụng 2-opt nếu 'use_memetic_ls' là True)
                fitness = self._evaluate_sequence(seq, active_fleet, use_memetic_ls)
                pop_with_fitness.append((seq, fitness))

            current_best_seq, current_best_fitness = min(pop_with_fitness, key=lambda x: x[1])

            if current_best_fitness < best_fitness:
                best_overall_seq = deepcopy(current_best_seq)
                best_fitness = current_best_fitness
                # <--- THAY ĐỔI: In Fitness (bao gồm cả penalty)
                print(f"  Gen {generation}: New Best Fitness = {best_fitness:.2f}")

        elapsed = time.time() - start_time

        final_routes_list = self._split_customers_into_routes_for_active_fleet(best_overall_seq, active_fleet)
        routes = self._convert_ga_to_routes(final_routes_list)
        true_distance = sum(r['distance'] for r in routes)

        print(f"  → {strategy_name} Finished. True Distance: {true_distance:.2f}, Final Fitness: {best_fitness:.2f}")

        return {
            'status': 'success',
            'strategy': strategy_name,  # <--- THAY ĐỔI
            'total_distance': true_distance,
            'fitness_with_penalty': best_fitness,
            'routes': routes,
            'elapsed_time': elapsed,
            'num_routes': len(routes),
            'generations': generation + 1
        }

    def _initialize_ga_population_seq(self, size):
        """Generate initial population of customer sequences."""
        population = []
        customers = list(self.all_customer_indices)
        for _ in range(size):
            random.shuffle(customers)
            population.append(customers[:])
        return population

    def _evaluate_sequence(self, customer_sequence, active_fleet, apply_local_search=False):  # <--- THAY ĐỔI
        """Helper to decode a sequence and evaluate its routes."""
        # 1. Decode sequence into routes
        routes = self._split_customers_into_routes_for_active_fleet(customer_sequence, active_fleet)

        # <--- THAY ĐỔI: (Memetic Step) Áp dụng local search (2-opt) trước khi đánh giá
        if apply_local_search:
            routes = self._apply_local_search_to_routes_list(routes)

        # 2. Evaluate routes (calculates distance + penalty)
        return self._evaluate_routes(routes)

    def _tournament_selection_seq(self, population_seqs, fitness_scores, tournament_size=5):
        """Tournament selection on sequences."""
        selected = []
        for _ in range(len(population_seqs)):
            tournament_idx = random.sample(range(len(population_seqs)), min(tournament_size, len(population_seqs)))
            best_idx = max(tournament_idx, key=lambda i: fitness_scores[i])
            selected.append(deepcopy(population_seqs[best_idx]))
        return selected

    def _order_crossover_seq(self, parent1_seq, parent2_seq):
        """Order Crossover (OX) on customer sequences."""
        size = len(parent1_seq)
        if size < 2:
            return deepcopy(parent1_seq), deepcopy(parent2_seq)

        child1_seq, child2_seq = [-1] * size, [-1] * size
        cut1, cut2 = sorted(random.sample(range(size), 2))
        child1_seq[cut1:cut2] = parent1_seq[cut1:cut2]
        child2_seq[cut1:cut2] = parent2_seq[cut1:cut2]

        p2_idx, c1_idx = 0, 0
        while c1_idx < size:
            if c1_idx == cut1:
                c1_idx = cut2
            if c1_idx >= size:
                break
            if parent2_seq[p2_idx] not in child1_seq:
                child1_seq[c1_idx] = parent2_seq[p2_idx]
                c1_idx += 1
            p2_idx += 1

        p1_idx, c2_idx = 0, 0
        while c2_idx < size:
            if c2_idx == cut1:
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

    # <--- THAY ĐỔI: Thêm toán tử đột biến INVERSION (2-opt trên sequence)
    def _inversion_mutation_seq(self, sequence):
        """Mutation: Reverses a random subsequence."""
        mutated = deepcopy(sequence)
        if len(mutated) < 2:
            return mutated

        i, j = sorted(random.sample(range(len(mutated)), 2))

        # Đảo ngược đoạn slice
        mutated[i:j + 1] = mutated[i:j + 1][::-1]
        return mutated

    # <--- KẾT THÚC THAY ĐỔI

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
            if len(route) <= 2:
                continue
            for i in range(len(route) - 1):
                total_distance += self.distance_matrix[route[i]][route[i + 1]]
            for node in route:
                if node in self.all_customer_indices:
                    served_customers.add(node)

        unserved_customers = self.all_customer_indices - served_customers
        unserved_count = len(unserved_customers)
        penalty = unserved_count * self._max_penalty

        return total_distance + penalty

    def _convert_ga_to_routes(self, ga_routes):
        """Convert GA routes list (from decoder) to standard output format"""
        routes = []
        total_distance = 0
        vehicle_id_counter = 0
        for route in ga_routes:
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

    # <--- THAY ĐỔI: Thêm hàm 2-OPT (nhanh hơn 3-opt, dùng cho Memetic GA)
    def _two_opt_route(self, route):
        """
        Tối ưu hóa một route bằng 2-opt.
        Không kiểm tra ràng buộc (như hàm 3-opt gốc).
        """
        best_route = route
        best_distance = self._calculate_route_distance(best_route)
        improved = True

        # Cần ít nhất 4 điểm (D, C1, C2, D) để thực hiện 2-opt
        if len(route) < 4:
            return route

        while improved:
            improved = False
            # i duyệt từ 1 (điểm khách hàng đầu tiên)
            # j duyệt từ i + 1 (điểm khách hàng tiếp theo)
            for i in range(1, len(best_route) - 2):
                if improved: break
                for j in range(i + 1, len(best_route) - 1):
                    # Chúng ta đang xem xét phá vỡ 2 cạnh:
                    # (i-1) -> i  (ví dụ: A -> B)
                    # j -> (j+1)  (ví dụ: C -> D)
                    A, B = best_route[i - 1], best_route[i]
                    C, D = best_route[j], best_route[j + 1]

                    d = self.distance_matrix

                    # Khoảng cách mới: A -> C và B -> D
                    # (Đồng thời đảo ngược đoạn B...C)
                    new_dist = d[A][C] + d[B][D]
                    old_dist = d[A][B] + d[C][D]

                    if new_dist < old_dist:
                        # Thực hiện swap
                        best_route = best_route[:i] + best_route[j:i - 1:-1] + best_route[j + 1:]
                        best_distance = self._calculate_route_distance(best_route)
                        improved = True
                        break

        return best_route

    def _apply_local_search_to_routes_list(self, routes_list):
        """
        Áp dụng _two_opt_route cho một danh sách các routes (đầu ra từ decoder).
        """
        polished_routes = []
        for route in routes_list:
            if len(route) > 4:  # Chỉ chạy 2-opt nếu có đủ điểm
                polished_route = self._two_opt_route(deepcopy(route))
                polished_routes.append(polished_route)
            else:
                polished_routes.append(route)
        return polished_routes

    # <--- KẾT THÚC THAY ĐỔI

    # ============= 3-OPT OPTIMIZATION (FULL) =============
    # (Giữ nguyên, không thay đổi, dùng để HẬU XỬ LÝ)
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
            if len(best_route) < 7:
                break

            n = len(best_route)
            for i in range(1, n - 4):
                if improved: break
                for j in range(i + 2, n - 2):
                    if improved: break
                    for k in range(j + 2, n):
                        if improved: break

                        A, B = best_route[i - 1], best_route[i]
                        C, D = best_route[j - 1], best_route[j]
                        E, F = best_route[k - 1], best_route[k]
                        d = self.distance_matrix
                        d0 = d[A][B] + d[C][D] + d[E][F]

                        # Case 1 (2-opt): A-B C-E D-F (reverse seg D...E)
                        d1 = d[A][B] + d[C][E] + d[D][F]
                        if d1 < d0:
                            best_route = best_route[:j] + best_route[k - 1:j - 1:-1] + best_route[k:]
                            best_distance = self._calculate_route_distance(best_route)
                            d0 = best_distance
                            improved = True
                            continue

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
                            best_route = best_route[:i] + best_route[j - 1:i - 1:-1] + best_route[
                                k - 1:j - 1:-1] + best_route[k:]
                            best_distance = self._calculate_route_distance(best_route)
                            d0 = best_distance
                            improved = True
                            continue

                        # Case 7 (3-opt): A-E D-C B-F (swap; reverse all)
                        d7 = d[A][E] + d[D][C] + d[B][F]
                        if d7 < d0:
                            best_route = best_route[:i] + best_route[k - 1:j - 1:-1] + best_route[
                                j - 1:i - 1:-1] + best_route[k:]
                            best_distance = self._calculate_route_distance(best_route)
                            d0 = best_distance
                            improved = True
                            continue

        return best_route, best_distance, iteration

    def apply_3opt_to_routes(self, routes):
        """Apply 3-opt to all routes"""
        print(f"    → Starting 3-opt (Post-Processing) on {len(routes)} routes...")
        optimized_routes = []
        total_improvement = 0

        for idx, route_info in enumerate(routes):
            original_distance = route_info['distance']
            if len(route_info['route']) < 7:
                print(f"      Route {idx + 1}: skipping (too short for 3-opt)")
                optimized_routes.append(route_info)
                continue

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

            for seg_size in range(1, min(segment_size + 1, len(best_route) - 3)):
                if improved: break
                for i in range(1, len(best_route) - seg_size - 1):
                    if improved: break
                    segment = best_route[i:i + seg_size]
                    remaining = best_route[:i] + best_route[i + seg_size:]
                    for j in range(1, len(remaining)):
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

        return best_route, best_distance, iteration

    # ============= PARALLEL EXECUTION =============

    def run_strategies_parallel(self, time_limit=45):
        """Run multiple strategies in parallel"""
        start_time = time.time()

        print("\n" + "=" * 80)
        print("RUNNING ADVANCED STRATEGIES ")
        print("=" * 80)

        results = []

        # <--- THAY ĐỔI: Chiến lược 1 là GA Memetic (tối ưu hóa)
        print("\n[1/3] Running Memetic Algorithm (GA + Internal 2-opt)...")
        result_ga_memetic = self.genetic_algorithm_mdvrp(
            population_size=100,
            generations=250,
            use_memetic_ls=True,  # <--- Bật 2-opt nội bộ
            time_limit=time_limit
        )
        results.append(result_ga_memetic)

        # <--- THAY ĐỔI: Chiến lược 2 là OR-Tools (đã bỏ TW) + 3-opt
        print("\n[2/3] Running OR-Tools (No TW) + 3-opt Post-Processing...")
        result_or = self._run_ortools_with_3opt(time_limit)
        results.append(result_or)

        # <--- THAY ĐỔI: Chiến lược 3 là GA cơ bản + 3-opt (để so sánh)
        print("\n[3/3] Running Base GA + 3-opt Post-Processing...")
        result_ga_base = self.genetic_algorithm_mdvrp(
            population_size=100,
            generations=250,
            use_memetic_ls=False,  # <--- Tắt 2-opt nội bộ
            time_limit=int(time_limit * 0.5)  # Chạy GA cơ bản nhanh hơn
        )

        if result_ga_base['status'] == 'success':
            # Áp dụng 3-opt HẬU XỬ LÝ
            opt_routes, improvement = self.apply_3opt_to_routes(result_ga_base['routes'])
            result_ga_base['routes'] = opt_routes
            result_ga_base['total_distance'] = sum(r['distance'] for r in opt_routes)
            result_ga_base['strategy'] = 'BASE_GA + 3-OPT_HYBRID'
        results.append(result_ga_base)

        # Compare
        successful = [r for r in results if r['status'] == 'success' and r['total_distance'] > 0]
        if successful:
            best = min(successful, key=lambda x: x['total_distance'])
            print("\n" + "=" * 80)
            print("ADVANCED STRATEGIES COMPARISON (NO TIME WINDOWS)")
            print("=" * 80)

            for i, result in enumerate(successful, 1):
                gap = ((result['total_distance'] - best['total_distance']) /
                       best['total_distance'] * 100) if best['total_distance'] > 0 else 0
                marker = "🏆 BEST" if result == best else ""
                print(f"\nStrategy {i}: {result['strategy']}")
                print(f"  Distance: {result['total_distance']:.2f}")
                print(f"  Routes: {result['num_routes']}")
                print(f"  Time: {result['elapsed_time']:.2f}s")
                print(f"  Gap: {gap:.2f}% {marker}")
        else:
            print("\n" + "=" * 80)
            print("All strategies failed.")

        elapsed = time.time() - start_time
        return {
            'status': 'success',
            'all_results': results,
            'best_result': best if successful else None,
            'total_time': elapsed
        }

    def _run_ortools_with_3opt(self, time_limit=45):
        """OR-Tools + 3-opt optimization (NO TIME WINDOWS)"""
        start_time = time.time()
        try:
            print("  → Initializing OR-Tools routing model (No TW)...")
            # <--- THAY ĐỔI: Gọi hàm _get_routing_model (đã bỏ TW)
            routing, manager = self._get_routing_model()

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
                # <--- THAY ĐỔI: Tăng độ chính xác khi chia lại (từ 100 lên 1000)
                routes, base_distance = self._extract_routes(routing, manager, solution, 1000.0)
                print(f"  → Base distance: {base_distance:.2f}")

                if not routes:
                    print("  ✗ OR-Tools: Solution found but no routes extracted.")
                    return {
                        'status': 'failed',
                        'strategy': 'OR-TOOLS (No TW) + 3-OPT',
                        'message': 'No routes extracted from solution',
                        'elapsed_time': elapsed
                    }

                print(f"  → Applying 3-opt optimization to {len(routes)} routes...")
                opt_routes, improvement = self.apply_3opt_to_routes(routes)
                new_distance = base_distance - improvement
                print(f"  → 3-opt improvement: {improvement:.2f}")
                print(f"  → Final distance: {new_distance:.2f}")

                for route in opt_routes:
                    route["coordinates"] = [self.all_locations[node] for node in route["route"]]

                return {
                    'status': 'success',
                    'strategy': 'OR-TOOLS (No TW) + 3-OPT',
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
                    'strategy': 'OR-TOOLS (No TW) + 3-OPT',
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
                'strategy': 'OR-TOOLS (No TW) + 3-OPT',
                'message': str(e),
                'elapsed_time': elapsed
            }

    def _extract_routes(self, routing, manager, solution, scale_factor=100.0):  # <--- THAY ĐỔI
        """Extract routes from solution"""
        routes = []
        total_distance = 0

        # <--- THAY ĐỔI: Xóa 'time_dimension' vì nó không còn tồn tại
        # time_dimension = routing.GetDimensionOrDie('Time')

        for vehicle_id in range(self.num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                previous_index = index
                index = solution.Value(routing.NextVar(index))

                # <--- THAY ĐỔI: Sử dụng 'scale_factor'
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            node = manager.IndexToNode(index)
            route.append(node)

            if len(route) > 2:
                routes.append({
                    'vehicle_id': vehicle_id,
                    'depot': self.starts[vehicle_id],
                    'route': route,
                    'distance': route_distance / scale_factor  # <--- THAY ĐỔI
                })
                total_distance += route_distance

        return routes, total_distance / scale_factor  # <--- THAY ĐỔI


# Export function for backend
def solve_mdvrp_advanced(depots, customers, num_vehicles_per_depot,
                         vehicle_capacities=None, demands=None,
                         time_windows=None, service_times=None,  # <--- THAY ĐỔI: Vẫn giữ, nhưng sẽ bị bỏ qua
                         strategy='advanced_benchmark', time_limit=45):
    print("Initializing AdvancedMDVRPSolver (Note: Time Windows are disabled in this version).")

    solver = AdvancedMDVRPSolver(
        depots, customers, num_vehicles_per_depot,
        vehicle_capacities, demands,
        time_windows, service_times  # <--- THAY ĐỔI: Truyền vào, nhưng __init__ sẽ bỏ qua chúng
    )

    if strategy == 'genetic':
        result = solver.genetic_algorithm_mdvrp(time_limit=time_limit, use_memetic_ls=True)
    elif strategy == '3opt':
        result = solver._run_ortools_with_3opt(time_limit=time_limit)
    elif strategy == 'advanced_benchmark':
        result = solver.run_strategies_parallel(time_limit=time_limit)
    else:
        result = {'status': 'error', 'message': 'Unknown strategy'}

    return result