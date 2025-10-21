from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from .mdvrp_solver import solve_mdvrp_advanced
from .utils import get_coordinates
import json
import os

def calculate_routes(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode('utf-8'))

            # Giả sử lấy sẵn dữ liệu từ file JSON
            base_dir = os.path.dirname(__file__)
            depots_path = os.path.join(base_dir, "../../data/depots.json")
            customers_path = os.path.join(base_dir, "../../data/customers.json")

            with open(depots_path, "r", encoding="utf-8") as f:
                depots = [(d["latitude"], d["longitude"]) for d in json.load(f)]

            with open(customers_path, "r", encoding="utf-8") as f:
                customers = [(c["latitude"], c["longitude"]) for c in json.load(f)]

            # Gọi solver
            result = solve_mdvrp_advanced(
                depots=depots,
                customers=customers,
                num_vehicles_per_depot=data.get("num_vehicles_per_depot", 2)
            )

            return JsonResponse(result, safe=False)

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    return JsonResponse({"status": "failed", "message": "Only POST allowed"}, status=405)
