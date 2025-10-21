from flask import Flask, request, jsonify
from flask_cors import CORS
from mdvrp_solver import solve_mdvrp_advanced
import json
import traceback
import sys

app = Flask(__name__)
CORS(app)

# Enable detailed error logging
app.config['PROPAGATE_EXCEPTIONS'] = True


@app.route('/api/calculate/', methods=['POST', 'OPTIONS'])
def calculate_routes():
    """POST request from frontend"""

    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 200

    print("\n" + "=" * 80)
    print("NEW REQUEST RECEIVED")
    print("=" * 80)

    try:
        # Step 1: Get request data
        print("Step 1: Parsing request data...")
        data = request.json

        if not data:
            print("ERROR: No JSON data received!")
            return jsonify({
                'status': 'error',
                'message': 'No JSON data received'
            }), 400

        print(f"Request data: {data}")

        num_vehicles = data.get('num_vehicles_per_depot', 2)
        strategy = data.get('strategy', 'advanced_benchmark')
        time_limit = data.get('time_limit', 45)

        print(f"  - Num vehicles: {num_vehicles}")
        print(f"  - Strategy: {strategy}")
        print(f"  - Time limit: {time_limit}")

        # Step 2: Map strategy names
        print("\nStep 2: Mapping strategy...")
        strategy_mapping = {
            'benchmark': 'advanced_benchmark',
            'strategy1': 'genetic',
            'strategy2': '3opt',
            'strategy3': 'advanced_benchmark',
            'benchmark_with_2opt': 'advanced_benchmark'
        }
        backend_strategy = strategy_mapping.get(strategy, strategy)
        print(f"  - Backend strategy: {backend_strategy}")

        # Step 3: Load data files
        print("\nStep 3: Loading data files...")
        try:
            with open('data/depots.json', 'r', encoding='utf-8') as f:
                depots_data = json.load(f)
            print(f"  - Loaded {len(depots_data)} depots")
        except FileNotFoundError:
            print("  ERROR: data/depots.json not found!")
            return jsonify({
                'status': 'error',
                'message': 'Depots data file not found'
            }), 500
        except json.JSONDecodeError as e:
            print(f"  ERROR: Invalid JSON in depots.json: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Invalid depots JSON: {str(e)}'
            }), 500

        try:
            with open('data/customers.json', 'r', encoding='utf-8') as f:
                customers_data = json.load(f)
            print(f"  - Loaded {len(customers_data)} customers")
        except FileNotFoundError:
            print("  ERROR: data/customers.json not found!")
            return jsonify({
                'status': 'error',
                'message': 'Customers data file not found'
            }), 500
        except json.JSONDecodeError as e:
            print(f"  ERROR: Invalid JSON in customers.json: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Invalid customers JSON: {str(e)}'
            }), 500

        # Step 4: Convert to coordinates
        print("\nStep 4: Converting to coordinates...")
        depots = [(d['longitude'], d['latitude']) for d in depots_data]
        customers = [(c['longitude'], c['latitude']) for c in customers_data]
        print(f"  - Depots: {len(depots)}, Customers: {len(customers)}")

        # Step 5: Prepare demands
        print("\nStep 5: Preparing demands...")
        demands = [0] * len(depots)
        for customer in customers_data:
            if 'demand' in customer:
                weight = customer['demand'].get('weight', 1) if isinstance(customer['demand'], dict) else 1
                demands.append(weight)
            else:
                demands.append(1)
        print(f"  - Total demands: {len(demands)}")

        # Step 6: Prepare service times
        print("\nStep 6: Preparing service times...")
        service_times = [0] * len(depots)
        for customer in customers_data:
            service_time = customer.get('service_time', 30)
            service_times.append(service_time)
        print(f"  - Total service times: {len(service_times)}")

        # Step 7: Call solver
        print("\nStep 7: Calling MDVRP solver...")
        print("-" * 80)

        result = solve_mdvrp_advanced(
            depots=depots,
            customers=customers,
            num_vehicles_per_depot=num_vehicles,
            demands=demands,
            service_times=service_times,
            strategy=backend_strategy,
            time_limit=time_limit
        )

        print("-" * 80)
        print(f"Solver completed with status: {result.get('status')}")

        if result.get('status') != 'success':
            print(f"Solver failed: {result.get('message', 'Unknown error')}")
            return jsonify({
                'status': 'error',
                'message': f"Solver failed: {result.get('message', 'Unknown error')}",
                'solver_result': result
            }), 500

        # Step 8: Format response
        print("\nStep 8: Formatting response...")
        formatted_result = format_response_for_frontend(result, depots_data, customers_data)

        print("SUCCESS! Returning response to frontend")
        print("=" * 80 + "\n")

        return jsonify({
            'status': 'success',
            'data': formatted_result
        })

    except Exception as e:
        print("\n" + "!" * 80)
        print("EXCEPTION CAUGHT!")
        print("!" * 80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        print("-" * 80)
        traceback.print_exc(file=sys.stdout)
        print("-" * 80)
        print("!" * 80 + "\n")

        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc()
        }), 500


def format_response_for_frontend(result, depots_data, customers_data):
    """Format solver result to match frontend expectations"""

    print("  Formatting: Checking result structure...")

    try:
        if result['status'] != 'success':
            print(f"  Formatting: Result status is '{result['status']}', returning as-is")
            return result

        # Combine locations
        all_locations = depots_data + customers_data
        print(f"  Formatting: Total locations = {len(all_locations)}")

        # Handle advanced_benchmark (multiple strategies)
        if 'best_result' in result:
            print("  Formatting: Found 'best_result' key")
            best = result['best_result']

            if best is None:
                print("  ERROR: best_result is None!")
                return {
                    'status': 'error',
                    'message': 'No successful strategy found',
                    'all_results': result.get('all_results', [])
                }

            print(f"  Formatting: best_result keys = {list(best.keys())}")

            if 'routes' not in best:
                print("  ERROR: best_result has no 'routes' key!")
                return {
                    'status': 'error',
                    'message': 'Best result has no routes',
                    'best_result': best
                }

            print(f"  Formatting: Found {len(best['routes'])} routes in best_result")
            formatted_routes = []

            for idx, route_info in enumerate(best.get('routes', [])):
                print(f"    Formatting route {idx + 1}...", end=" ")
                formatted_route = format_single_route(route_info, all_locations)
                formatted_routes.append(formatted_route)
                print("OK")

            result_data = {
                'status': 'success',
                'strategy': best.get('strategy', 'unknown'),
                'total_distance': best.get('total_distance', 0),
                'elapsed_time': best.get('elapsed_time', 0),
                'num_routes': best.get('num_routes', 0),
                'routes': formatted_routes,
                'all_results': result.get('all_results', [])
            }

            print(f"  Formatting: Complete! Returning {len(formatted_routes)} routes")
            return result_data

        # Handle single strategy result
        elif 'routes' in result:
            print("  Formatting: Found 'routes' key (single strategy)")
            print(f"  Formatting: Found {len(result['routes'])} routes")

            formatted_routes = []
            for idx, route_info in enumerate(result['routes']):
                print(f"    Formatting route {idx + 1}...", end=" ")
                formatted_route = format_single_route(route_info, all_locations)
                formatted_routes.append(formatted_route)
                print("OK")

            result_data = {
                'status': 'success',
                'strategy': result.get('strategy', 'unknown'),
                'total_distance': result.get('total_distance', 0),
                'elapsed_time': result.get('elapsed_time', 0),
                'num_routes': len(formatted_routes),
                'routes': formatted_routes
            }

            print(f"  Formatting: Complete! Returning {len(formatted_routes)} routes")
            return result_data

        else:
            print("  ERROR: Result has neither 'best_result' nor 'routes' key!")
            print(f"  Result keys: {list(result.keys())}")
            return {
                'status': 'error',
                'message': 'Invalid result structure',
                'result_keys': list(result.keys())
            }

    except Exception as e:
        print(f"  ERROR in format_response_for_frontend: {str(e)}")
        traceback.print_exc()
        raise


def format_single_route(route_info, all_locations):
    """Format a single route with location details"""
    route_nodes = route_info.get('route', [])
    route_with_details = []

    for node_idx in route_nodes:
        if 0 <= node_idx < len(all_locations):
            location = all_locations[node_idx]
            if isinstance(location, dict):
                route_with_details.append({
                    'index': int(node_idx),
                    'latitude': float(location.get('latitude', 0)),
                    'longitude': float(location.get('longitude', 0)),
                    'name': str(location.get('name', f'Location {node_idx}')),
                    'address': str(location.get('address', 'Unknown'))
                })
        else:
            print(f"    [Warning] Node index {node_idx} out of range (0-{len(all_locations) - 1})")

    return {
        'vehicle_id': int(route_info.get('vehicle_id', 0)),
        'depot': int(route_info.get('depot', 0)),
        'route': [int(x) for x in route_nodes],
        'route_details': route_with_details,
        'distance': float(route_info.get('distance', 0)),
        'improvement': float(route_info.get('improvement', 0)),
        'method': str(route_info.get('method', 'unknown'))
    }


@app.route('/api/strategies/', methods=['GET'])
def get_strategies():
    """Return list of available strategies"""
    return jsonify({
        'strategies': [
            {'id': 'genetic', 'name': 'Genetic Algorithm'},
            {'id': '3opt', 'name': 'OR-Tools + 3-opt Optimization'},
            {'id': 'advanced_benchmark', 'name': 'Advanced Benchmark (GA + 3-opt + Hybrid)'},
            {'id': 'benchmark', 'name': 'Benchmark All Strategies (alias)'}
        ]
    })


@app.route('/api/health/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'MDVRP Solver API is running'
    })


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("STARTING MDVRP SOLVER API")
    print("=" * 80)
    print("Server: http://localhost:8000")
    print("Endpoints:")
    print("  - POST /api/calculate/")
    print("  - GET  /api/strategies/")
    print("  - GET  /api/health/")
    print("=" * 80 + "\n")

    app.run(debug=True, port=8000, host='0.0.0.0')