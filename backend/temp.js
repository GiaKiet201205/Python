// Initialize map
let map;
let markers = [];
let routes = [];
let currentTab = 'orders';
document.addEventListener('DOMContentLoaded', async function() {
    initMap();
    populateOrdersTable();
    await loadDriversData(); 
    showNotification('Các tuyến đường đã được lập kế hoạch. Hoàn tác thay đổi');
});

function initMap() {
    map = L.map('map').setView([10.8231, 106.6297], 12);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors | <a href="https://www.mapbox.com/">MapBox Streets</a>'
    }).addTo(map);

    const depotIcon = L.icon({
      iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png',
      shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
      iconSize: [25, 41],
      iconAnchor: [12, 41],
      popupAnchor: [1, -34],
      shadowSize: [41, 41]
    });

    const customerIcon = L.icon({
      iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
      shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
      iconSize: [25, 41],
      iconAnchor: [12, 41],
      popupAnchor: [1, -34],
      shadowSize: [41, 41]
    });

    fetch("../data/depots.json")
      .then(response => response.json())
      .then(data => {
        data.forEach(depot => {
          const marker = L.marker([depot.latitude, depot.longitude], { icon: depotIcon }).addTo(map);
          marker.bindPopup(`<b>${depot.name}</b><br>Lat: ${depot.latitude}<br>Lon: ${depot.longitude}`);
        });
      })
      .catch(error => console.error("Lỗi tải JSON depot:", error));

    fetch("../data/customers.json")
      .then(response => response.json())
      .then(data => {
        data.forEach(order => {
          const marker = L.marker([order.latitude, order.longitude], { icon: customerIcon }).addTo(map);
          marker.bindPopup(`<b>${order.name}</b><br>Lat: ${order.latitude}<br>Lon: ${order.longitude}`);
        });
      })
      .catch(error => console.error("Lỗi tải JSON customer:", error));

}

function toggleBottomPanel() {
    const panel = document.querySelector('.bottom-panel');
    panel.classList.toggle('collapsed');
}

function switchTab(tabElement, tabName) {
    document.querySelectorAll('.panel-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    tabElement.classList.add('active');
    currentTab = tabName;
    updatePanelContent(tabName);
}

function updatePanelContent(tabName) {
    const tbody = document.getElementById('orders-tbody');

    if (tabName === 'orders') {
        populateOrdersTable();
    } else if (tabName === 'routes') {
        tbody.innerHTML = '<tr><td colspan="13" class="text-center py-4">Routes information will be displayed here</td></tr>';
    } else if (tabName === 'timeline') {
        tbody.innerHTML = '<tr><td colspan="13" class="text-center py-4">Timeline view will be displayed here</td></tr>';
    } else if (tabName === 'not-scheduled') {
        tbody.innerHTML = '<tr><td colspan="13" class="text-center py-4">Unscheduled orders will be displayed here</td></tr>';
    }
}
function populateOrdersTable() {
    const tbody = document.getElementById('orders-tbody');
    tbody.innerHTML = '';

    ordersData.forEach((order, index) => {
        const priorityClass = `priority-${order.priority}`;
        const row = `
            <tr>
                <td><input type="checkbox" class="order-checkbox"></td>
                <td><strong>${order.id}</strong></td>
                <td><span class="order-priority ${priorityClass}"></span>${order.priority.charAt(0).toUpperCase() + order.priority.slice(1)}</td>
                <td>${order.location}</td>
                <td>${order.address}</td>
                <td>-</td>
                <td>-</td>
                <td>${order.weight}</td>
                <td>${order.volume}</td>
                <td>${order.duration}</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
        `;
        tbody.innerHTML += row;
    });
}

function toggleDriver(element, driverId) {
    const checkbox = element.querySelector('.driver-checkbox');
    checkbox.checked = !checkbox.checked;

    if (checkbox.checked) {
        element.classList.add('selected');
    } else {
        element.classList.remove('selected');
    }

    updateSelectedDrivers();
}

function updateSelectedDrivers() {
    selectedDrivers = [];
    const checkboxes = document.querySelectorAll('.driver-checkbox:checked');

    checkboxes.forEach(checkbox => {
        const driverItem = checkbox.closest('.driver-item');
        const driverId = driverItem.getAttribute('onclick').match(/'([^']+)'/)[1];
        const driver = driversData.find(d => d.id === driverId);
        if (driver) {
            selectedDrivers.push(driver);
        }
    });

    const badge = document.querySelector('.badge');
    if (badge) {
        badge.textContent = `${selectedDrivers.length} drivers selected`;
    }
}

function centerMap() {
    map.setView([10.8231, 106.6297], 12);
}
function zoomIn() {
    map.zoomIn();
}
function zoomOut() {
    map.zoomOut();
}

function toggleSatellite() {
    const currentLayer = map._layers[Object.keys(map._layers)[0]];
    map.removeLayer(currentLayer);
    if (currentLayer._url.includes('openstreetmap')) {
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles © Esri'
        }).addTo(map);
    } else {
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
    }
}

function toggleTraffic() {
    showNotification('Lớp giao thông đã được chuyển đổi');
}

function exportMap() {
    showNotification('Bản đồ đã được xuất thành công');
}
function fullScreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
        document.exitFullscreen();
    }
}

async function optimizeRoutes() {
  showLoading();
  try {
    // Chọn chiến lược (thay đổi nếu cần)
    const strategy = 'benchmark'; // hoặc 'strategy1', 'strategy2', 'strategy3'

    const response = await fetch("http://127.0.0.1:8000/api/calculate/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        num_vehicles_per_depot: 2,
        strategy: strategy,
        time_limit:45
      })
    });

    const result = await response.json();
    hideLoading();

    if (result.status === "success") {
      showNotification("Các tuyến đường đã được tối ưu hoá thành công!", "success");
      console.log("Kết quả:", result.data);

      // Render routes lên map
      renderRoutesOnMap(result.data);
    } else {
      showNotification(result.message || "Không tìm thấy giải pháp!", "error");
    }
  } catch (error) {
    hideLoading();
    console.error("Error:", error);
    showNotification("Lỗi khi tối ưu hoá tuyến đường!", "error");
  }
}

// Hàm vẽ routes lên map
function renderRoutesOnMap(data) {
  // Xoá các routes cũ nếu có
  routes.forEach(route => map.removeLayer(route));
  routes = [];
  const colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow'];
  if (data.results) {
    // Benchmark results có nhiều chiến lược
    data.results.forEach((strategyResult, index) => {
      if (strategyResult.status === 'success') {
        strategyResult.routes.forEach((route, routeIdx) => {
          drawRoute(route, colors[routeIdx % colors.length]);
        });
      }
    });
  } else if (data.routes) {
    data.routes.forEach((route, index) => {
      drawRoute(route, colors[index % colors.length]);
    });
  }
}
function drawRoute(route, color) {
  const routeCoordinates = [];
  route.route.forEach(nodeIndex => {
    const location = ordersData[nodeIndex] || {latitude: 0, longitude: 0};
    routeCoordinates.push([location.latitude, location.longitude]);
  });
  if (routeCoordinates.length > 0) {
    const polyline = L.polyline(routeCoordinates, {
      color: color,
      weight: 3,
      opacity: 0.7,
      dashArray: '5, 5'
    }).addTo(map);
    routes.push(polyline);
  }
}
let driversData = [];
let selectedDrivers = [];
let ordersData = [];
async function loadDriversData() {
    try {
        const depotsResponse = await fetch('../data/depots.json');
        const customersResponse = await fetch('../data/customers.json');
        const driversResponse = await fetch('../data/drivers.json');
        const depots = await depotsResponse.json();
        const customers = await customersResponse.json();
        const drivers = await driversResponse.json();

        // Load orders data from customers
        ordersData = customers.map(customer => ({
            id: customer.id,
            location: customer.name,
            address: customer.address || 'Unknown Address',
            priority: customer.priority || 'normal',
            weight: customer.demand ? customer.demand.weight || 0 : 0,
            volume: customer.demand ? customer.demand.volume || 0 : 0,
            duration: customer.service_time || 45,
            latitude: customer.latitude,
            longitude: customer.longitude
        }));
        driversData = drivers.map(driver => {
            const depot = depots.find(d => d.id === driver.depot_id);
            return {
                id: driver.id,
                name: driver.name,
                depot_id: driver.depot_id,
                depot_name: depot ? depot.name : 'Unknown',
                depot_address: depot ? depot.address : 'Unknown',
                latitude: depot ? depot.latitude : 0,
                longitude: depot ? depot.longitude : 0,
                orders: [] // Will be populated with customers if needed
            };
        });
        console.log('Drivers data loaded:', driversData.length);
        console.log('Orders data loaded:', ordersData.length);
    } catch (error) {
        console.error('Error loading drivers data:', error);
        showNotification('Lỗi tải dữ liệu tài xế', 'error');
    }
}

async function switchView() {
    if (selectedDrivers.length !== 2) {
        showNotification('Vui lòng chọn chính xác 2 tài xế để chuyển đổi!', 'error');
        return;
    }

    try {
        showLoading();
        const result = await performSwitch(selectedDrivers[0].id, selectedDrivers[1].id);
        hideLoading();
        if (result.success) {
            showNotification('Các tài xế đã được chuyển đổi thành công!', 'success');
            console.log('Switch result:', result);
        } else {
            showNotification(result.message || 'Chuyển đổi thất bại!', 'error');
        }
    } catch (error) {
        hideLoading();
        console.error('Switch error:', error);
        showNotification('Lỗi trong quá trình chuyển đổi!', 'error');
    }
}

async function performSwitch(driverId1, driverId2) {
    try {
        const driver1 = driversData.find(d => d.id === driverId1);
        const driver2 = driversData.find(d => d.id === driverId2);
        if (!driver1 || !driver2) {
            return {
                success: false,
                message: 'Driver not found!'
            };
        }
        if (driver1.id === driver2.id) {
            return {
                success: false,
                message: 'Please select 2 different drivers!'
            };
        }
        const tempOrders = [...driver1.orders];
        driver1.orders = [...driver2.orders];
        driver2.orders = tempOrders;
        return {
            success: true,
            message: 'Switch completed successfully!',
            details: {
                driver1: {
                    id: driver1.id,
                    oldOrders: tempOrders,
                    newOrders: driver1.orders
                },
                driver2: {
                    id: driver2.id,
                    oldOrders: driver2.orders,
                    newOrders: driver2.orders
                }
            }
        };
    } catch (error) {
        console.error('Switch error:', error);
        return {
            success: false,
            message: 'Error during switch: ' + error.message
        };
    }
}

async function performReverse(driverId) {
    try {
        const driver = driversData.find(d => d.id === driverId);
        if (!driver) {
            return {
                success: false,
                message: 'Driver not found!'
            };
        }
        if (!driver.orders || driver.orders.length <= 2) {
            return {
                success: false,
                message: 'Tuyến đường ngắn (<2 depot) , không thể đảo ngược!'
            };
        }
        const originalOrders = [...driver.orders];
        const middleOrders = driver.orders.slice(1, -1);
        const reversedMiddleOrders = middleOrders.reverse();
        driver.orders = [
            driver.orders[0],
            ...reversedMiddleOrders,
            driver.orders[driver.orders.length - 1]
        ];
        return {
            success: true,
            message: 'Route reversed successfully!',
            details: {
                driver: {
                    id: driver.id,
                    originalRoute: originalOrders,
                    reversedRoute: driver.orders
                }
            }
        };
    } catch (error) {
        console.error('Reverse error:', error);
        return {
            success: false,
            message: 'Error during reverse: ' + error.message
        };
    }
}
function copyRoutes() {
    showNotification('Routes copied to clipboard');
}
function hideDrivers() {
    const driversSection = document.querySelector('.drivers-section');
    driversSection.style.display = driversSection.style.display === 'none' ? 'block' : 'none';
}
function importOrders() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.csv,.xlsx,.json';
    input.onchange = function(e) {
        const file = e.target.files[0];
        if (file) {
            showNotification(`Importing ${file.name}...`);
            setTimeout(() => {
                showNotification('Orders imported successfully!', 'success');
            }, 2000);
        }
    };
    input.click();
}
function planRoutes() {
    showLoading();
    setTimeout(() => {
        hideLoading();
        showNotification('Routes planned successfully!', 'success');
    }, 2500);
}
function shareRoutes() {
    if (navigator.share) {
        navigator.share({
            title: 'Route Optimization Results',
            text: 'Check out these optimized routes!',
            url: window.location.href
        });
    } else {
        navigator.clipboard.writeText(window.location.href);
        showNotification('Link copied to clipboard!');
    }
}
function selectAllOrders(checkbox) {
    const orderCheckboxes = document.querySelectorAll('.order-checkbox');
    orderCheckboxes.forEach(cb => {
        cb.checked = checkbox.checked;
    });
}
function viewOrderDetails(orderIndex) {
    const order = ordersData[orderIndex];
    alert(`Order Details:\n\nID: ${order.id}\nLocation: ${order.location}\nAddress: ${order.address}\nPriority: ${order.priority}\nWeight: ${order.weight}kg\nVolume: ${order.volume}m³\nDuration: ${order.duration}`);
}
function showNotification(message, type = 'default') {
    const notification = document.getElementById('notification');
    const text = document.getElementById('notification-text');
    text.textContent = message;
    notification.className = `notification ${type}`;
    notification.style.display = 'flex';
    setTimeout(() => {
        hideNotification();
    }, 5000);
}
function hideNotification() {
    document.getElementById('notification').style.display = 'none';
}
function showLoading() {
    document.getElementById('loading').style.display = 'block';
}
function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}
window.addEventListener('resize', function() {
    if (map) {
        setTimeout(() => {
            map.invalidateSize();
        }, 100);
    }
});