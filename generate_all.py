import json
import random
import os
import math
from shapely.geometry import shape, Point
from shapely.ops import unary_union

# --- CÁC THAM SỐ CẤU HÌNH ---
NUM_DEPOTS = 500
NUM_CUSTOMERS = 1000
NUM_DRIVERS = 1000
OUTPUT_DIR = "data"

BOUNDARY_FILE = os.path.join(OUTPUT_DIR, "vietnam_boundary.geojson")


def load_boundary_polygon(filepath):
    """
    Tải ranh giới từ file GeoJSON.
    Hàm này sẽ đọc TẤT CẢ các feature (Polygon/MultiPolygon)
    và gộp chúng lại thành một shape duy nhất.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)

        # Tạo một danh sách các 'shape' từ tất cả features
        all_geometries = []
        for feature in geojson_data['features']:
            geom = shape(feature['geometry'])
            all_geometries.append(geom)

        # Gộp tất cả các shape lại thành một (MultiPolygon)
        # Điều này đảm bảo điểm có thể được tạo ở cả đất liền và các đảo
        if not all_geometries:
            print(f"Lỗi: Không tìm thấy 'features' nào trong file {filepath}.")
            return None

        return unary_union(all_geometries)

    except Exception as e:
        print(f"Lỗi: Không thể tải file ranh giới '{filepath}'. Lỗi: {e}")
        return None


# <<< SỬA LỖI 1: ĐÃ UN-COMMENT HÀM NÀY >>>
def generate_random_point_in_polygon(polygon):
    """Tạo một điểm ngẫu nhiên chắc chắn nằm trong Polygon/MultiPolygon cho trước."""
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    while True:
        # Tạo điểm ngẫu nhiên trong bounding box
        random_point = Point(random.uniform(min_lon, max_lon), random.uniform(min_lat, max_lat))
        # Kiểm tra xem điểm có nằm trong ranh giới (đã gộp) không
        if polygon.contains(random_point):
            return round(random_point.y, 6), round(random_point.x, 6)  # Trả về (latitude, longitude)


def generate_random_address():
    """ Tạo địa chỉ ngẫu nhiên đơn giản, chung chung cho Việt Nam """
    provinces = ["Hà Nội", "TP. HCM", "Đà Nẵng", "Hải Phòng", "Cần Thơ", "An Giang", "Bình Dương", "Bắc Ninh",
                 "Quảng Ninh", "Nghệ An", "Thanh Hóa", "Khánh Hòa"]
    districts = ["Quận 1", "Quận Cầu Giấy", "Quận Hải Châu", "Quận Hồng Bàng", "Quận Ninh Kiều", "Huyện Châu Đốc",
                 "TP. Thủ Dầu Một", "TP. Bắc Ninh"]
    streets = ["Lý Thường Kiệt", "Trần Hưng Đạo", "Võ Nguyên Giáp", "Phạm Văn Đồng", "Nguyễn Văn Cừ", "Lê Lợi"]

    house_number = random.randint(1, 1500)
    street = random.choice(streets)
    district = random.choice(districts)
    province = random.choice(provinces)

    return f"Số {house_number}, Đường {street}, {district}, {province}"


def generate_depots_data(num_depots, polygon):
    """Tạo dữ liệu kho với tọa độ an toàn."""
    depots_list = []
    print(f"Đang tạo {num_depots} depots...")
    for i in range(num_depots):
        depot_id = i + 1
        lat, lon = generate_random_point_in_polygon(polygon)

        depot_id_str = f"{depot_id:03d}"

        depot = {
            "id": depot_id_str,
            "name": f"Kho Việt Nam - {depot_id_str}",
            "address": generate_random_address(),
            "latitude": lat,
            "longitude": lon
        }
        depots_list.append(depot)
    print("-> Tạo depots thành công!")
    return depots_list


def generate_customers_data(num_customers, polygon):
    """Tạo dữ liệu khách hàng với tọa độ an toàn."""
    customers_list = []
    print(f"Đang tạo {num_customers} customers...")
    for i in range(num_customers):
        customer_id = i + 1
        lat, lon = generate_random_point_in_polygon(polygon)
        customer = {
            "id": customer_id,
            "name": f"Khách hàng {customer_id}",
            "phone": f"09{random.randint(10000000, 99999999)}",
            "email": f"customer.{customer_id}@example.com",
            "address": generate_random_address(),
            "latitude": lat,
            "longitude": lon
        }
        customers_list.append(customer)
    print("-> Tạo customers thành công!")
    return customers_list


def generate_drivers_data(num_drivers, num_depots):
    """Tạo dữ liệu tài xế."""
    drivers_list = []
    print(f"Đang tạo {num_drivers} drivers...")

    drivers_per_depot = math.ceil(num_drivers / num_depots)  # Sẽ là 2

    for i in range(num_drivers):
        driver_id = i + 1
        depot_id = math.floor(i / drivers_per_depot) + 1

        driver_id_str = f"{driver_id:04d}"

        # Đảm bảo depot_id không vượt quá số lượng depot
        # (quan trọng nếu num_drivers không chia hết)
        actual_depot_id = min(depot_id, num_depots)
        depot_id_str = f"{actual_depot_id:03d}"

        driver = {
            "id": driver_id_str,
            "name": f"Tài xế {driver_id_str}",
            "phone": f"09{random.randint(10000000, 99999999)}",
            "depot_id": depot_id_str
        }
        drivers_list.append(driver)
    print("-> Tạo drivers thành công!")
    return drivers_list


def save_to_json(data, filename):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu thành công dữ liệu vào file: {filepath}\n")


# --- CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    # 1. Tải ranh giới (Việt Nam - đã gộp tất cả features)
    vietnam_polygon = load_boundary_polygon(BOUNDARY_FILE)

    if vietnam_polygon:
        print("Tải ranh giới địa lý (đất liền và các đảo) thành công!")

        # 2. Tạo dữ liệu drivers (không cần ranh giới)
        drivers_data = generate_drivers_data(NUM_DRIVERS, NUM_DEPOTS)
        save_to_json(drivers_data, "drivers.json")

        # 3. Tạo depots và customers với tọa độ được đảm bảo
        depots_data = generate_depots_data(NUM_DEPOTS, vietnam_polygon)
        save_to_json(depots_data, "depots.json")

        customers_data = generate_customers_data(NUM_CUSTOMERS, vietnam_polygon)
        save_to_json(customers_data, "customers.json")

        print("Hoàn tất! Đã tạo 3 file JSON với tọa độ chính xác.")
    else:
        print("Không thể tạo dữ liệu do không tải được file ranh giới.")
        print(f"Vui lòng kiểm tra file: {BOUNDARY_FILE}")