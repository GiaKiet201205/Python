import json
import os
import re
import math
import pandas as pd
from datetime import datetime
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def switch_drivers_depot(request):
    """
    API endpoint để hoán đổi depot_id giữa 2 drivers
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode('utf-8'))
            driver_id_1 = data.get("driver_id_1")
            driver_id_2 = data.get("driver_id_2")

            if not driver_id_1 or not driver_id_2:
                return JsonResponse({
                    "status": "error",
                    "message": "Thiếu thông tin driver ID!"
                }, status=400)

            # Đường dẫn đến file drivers.json
            # Từ backend/mdvrp_app/ops.py lên 2 cấp là backend/, rồi lên 1 cấp nữa là root project
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            drivers_path = os.path.join(base_dir, "data", "drivers.json")

            # Đọc file drivers.json
            with open(drivers_path, "r", encoding="utf-8") as f:
                drivers = json.load(f)

            # Tìm 2 drivers cần switch
            driver1 = None
            driver2 = None
            driver1_index = -1
            driver2_index = -1

            for i, driver in enumerate(drivers):
                if driver["id"] == driver_id_1:
                    driver1 = driver
                    driver1_index = i
                elif driver["id"] == driver_id_2:
                    driver2 = driver
                    driver2_index = i

            if driver1 is None or driver2 is None:
                return JsonResponse({
                    "status": "error",
                    "message": "Không tìm thấy một hoặc cả hai drivers!"
                }, status=404)

            # Kiểm tra 2 drivers khác nhau
            if driver_id_1 == driver_id_2:
                return JsonResponse({
                    "status": "error",
                    "message": "Vui lòng chọn 2 drivers khác nhau!"
                }, status=400)

            # Hoán đổi depot_id
            original_depot1 = driver1["depot_id"]
            original_depot2 = driver2["depot_id"]

            drivers[driver1_index]["depot_id"] = original_depot2
            drivers[driver2_index]["depot_id"] = original_depot1

            # Ghi lại file
            with open(drivers_path, "w", encoding="utf-8") as f:
                json.dump(drivers, f, ensure_ascii=False, indent=2)

            return JsonResponse({
                "status": "success",
                "message": f"Đã hoán đổi depot thành công!",
                "details": {
                    f"driver_{driver_id_1}": {
                        "old_depot_id": original_depot1,
                        "new_depot_id": original_depot2
                    },
                    f"driver_{driver_id_2}": {
                        "old_depot_id": original_depot2,
                        "new_depot_id": original_depot1
                    }
                }
            })

        except Exception as e:
            return JsonResponse({
                "status": "error",
                "message": f"Lỗi server: {str(e)}"
            }, status=500)

    return JsonResponse({
        "status": "error",
        "message": "Chỉ chấp nhận phương thức POST"
    }, status=405)


@csrf_exempt
def add_customer(request):
    if request.method == "POST":
        try:
            # Đường dẫn đến file customers.json
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            customers_path = os.path.join(base_dir, "data", "customers.json")

            with open(customers_path, "r", encoding="utf-8") as f:
                customers = json.load(f)
            next_id = generate_next_customer_id(customers)
            if request.content_type and 'multipart' in request.content_type:
                return handle_excel_upload(request, customers, next_id, customers_path)
            else:
                return handle_manual_input(request, customers, next_id, customers_path)
        except Exception as e:
            return JsonResponse({
                "status": "error",
                "message": f"Lỗi server: {str(e)}"
            }, status=500)
    return JsonResponse({
        "status": "error",
        "message": "Chỉ chấp nhận phương thức POST"
    }, status=405)
def generate_next_customer_id(customers):
    if not customers:
        return "C0001"
    last_customer = max(customers, key=lambda x: int(x["id"][1:]))
    last_id_number = int(last_customer["id"][1:])
    next_id_number = last_id_number + 1
    return f"C{next_id_number:04d}"
def handle_manual_input(request, customers, next_id, customers_path):
    try:
        data = json.loads(request.body.decode('utf-8'))
        required_fields = ["name", "address", "phone"]
        for field in required_fields:
            if not data.get(field):
                return JsonResponse({
                    "status": "error",
                    "message": f"Thiếu thông tin bắt buộc: {field}"
                }, status=400)
        new_customer = {
            "id": next_id,
            "name": data["name"],
            "address": data["address"],
            "phone": data["phone"],
            "email": data.get("email", ""),
            "latitude": data.get("latitude", 0),
            "longitude": data.get("longitude", 0)
        }
        customers.append(new_customer)
        with open(customers_path, "w", encoding="utf-8") as f:
            json.dump(customers, f, ensure_ascii=False, indent=2)
        return JsonResponse({
            "status": "success",
            "message": f"Đã thêm khách hàng {next_id} thành công!",
            "customer": new_customer
        })
    except json.JSONDecodeError:
        return JsonResponse({
            "status": "error",
            "message": "Dữ liệu JSON không hợp lệ"
        }, status=400)
def handle_excel_upload(request, customers, next_id, customers_path):
    try:
        if 'file' not in request.FILES:
            return JsonResponse({
                "status": "error",
                "message": "Không tìm thấy file Excel"
            }, status=400)

        excel_file = request.FILES['file']
        if not excel_file.name.endswith(('.xlsx', '.xls')):
            return JsonResponse({
                "status": "error",
                "message": "Chỉ chấp nhận file Excel (.xlsx, .xls)"
            }, status=400)
        df = pd.read_excel(excel_file)
        required_columns = ["name", "address", "phone"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return JsonResponse({
                "status": "error",
                "message": f"Thiếu các cột bắt buộc: {', '.join(missing_columns)}"
            }, status=400)

        added_customers = []
        current_id = int(next_id[1:])
        for index, row in df.iterrows():
            current_id += 1
            customer_id = f"C{current_id:04d}"
            new_customer = {
                "id": customer_id,
                "name": str(row["name"]),
                "address": str(row["address"]),
                "phone": str(row["phone"]),
                "email": str(row.get("email", "")),
                "latitude": float(row.get("latitude", 0)),
                "longitude": float(row.get("longitude", 0))
            }

            customers.append(new_customer)
            added_customers.append(new_customer)
        with open(customers_path, "w", encoding="utf-8") as f:
            json.dump(customers, f, ensure_ascii=False, indent=2)

        return JsonResponse({
            "status": "success",
            "message": f"Đã thêm {len(added_customers)} khách hàng từ file Excel thành công!",
            "added_customers": added_customers,
            "next_available_id": f"C{current_id + 1:04d}"
        })

    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": f"Lỗi xử lý file Excel: {str(e)}"
        }, status=400)


@csrf_exempt
def get_next_customer_id(request):
    if request.method == "GET":
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            customers_path = os.path.join(base_dir, "data", "customers.json")
            with open(customers_path, "r", encoding="utf-8") as f:
                customers = json.load(f)
            next_id = generate_next_customer_id(customers)
            return JsonResponse({
                "status": "success",
                "next_id": next_id
            })
        except Exception as e:
            return JsonResponse({
                "status": "error",
                "message": f"Lỗi: {str(e)}"
            }, status=500)

    return JsonResponse({
        "status": "error",
        "message": "Chỉ chấp nhận phương thức GET"
    }, status=405)