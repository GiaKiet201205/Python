async function readDataFile(filename) {
    const fs = require('fs').promises;
    const path = require('path');

    // Đường dẫn tuyệt đối đến thư mục data
    const baseDir = path.join(__dirname, '..', '..', 'data');
    const filePath = path.join(baseDir, filename);

    try {
        const data = await fs.readFile(filePath, 'utf8');
        return JSON.parse(data);
    } catch (error) {
        console.error(`Lỗi đọc file ${filename}:`, error);
        throw error;
    }
}

async function writeDataFile(filename, data) {
    const fs = require('fs').promises;
    const path = require('path');
    const filePath = path.join(__dirname, '..', 'data', filename);
    await fs.writeFile(filePath, JSON.stringify(data, null, 2), 'utf8');
}

/**
 * Hoán đổi orders giữa 2 drivers
 * @param {Array} selectedDrivers - Mảng 2 drivers cần switch
 * @returns {Object} Kết quả switch
 */
function switchDrivers(selectedDrivers) {
    const validation = validateSwitchConditions(selectedDrivers);
    if (!validation.valid) {
        return { success: false, message: validation.message };
    }

    const [driver1, driver2] = selectedDrivers;

    const driver1Orders = [...driver1.orders];
    const driver2Orders = [...driver2.orders];

    // Hoán đổi orders giữa 2 drivers
    [driver1.orders, driver2.orders] = [driver2.orders, driver1.orders];

    return {
        success: true,
        message: `Đã hoán đổi orders thành công!`,
        details: {
            driver1: {
                id: driver1.id,
                oldOrders: driver1Orders,
                newOrders: driver1.orders
            },
            driver2: {
                id: driver2.id,
                oldOrders: driver2Orders,
                newOrders: driver2.orders
            }
        }
    };
}

function validateConditions(conditions) {
    const { minItems, maxItems, requiredFields, customChecks } = conditions;

    if (minItems !== undefined && (!conditions.items || conditions.items.length < minItems)) {
        return { valid: false, message: conditions.minMessage };
    }

    if (maxItems !== undefined && conditions.items && conditions.items.length > maxItems) {
        return { valid: false, message: conditions.maxMessage };
    }

    if (requiredFields) {
        for (const field of requiredFields) {
            if (!field.value || !field.value[field.key]) {
                return { valid: false, message: field.message };
            }
        }
    }

    if (customChecks) {
        for (const check of customChecks) {
            if (!check.condition) {
                return { valid: false, message: check.message };
            }
        }
    }

    return { valid: true, message: conditions.successMessage };
}

function validateSwitchConditions(selectedDrivers) {
    return validateConditions({
        minItems: 2,
        maxItems: 2,
        minMessage: 'Vui lòng chọn 2 drivers để thực hiện switch!',
        maxMessage: 'Chỉ được chọn tối đa 2 drivers để thực hiện switch!',
        requiredFields: [
            { key: 'id', value: selectedDrivers?.[0], message: 'Thông tin driver không hợp lệ!' },
            { key: 'orders', value: selectedDrivers?.[0], message: 'Thông tin driver không hợp lệ!' },
            { key: 'id', value: selectedDrivers?.[1], message: 'Thông tin driver không hợp lệ!' },
            { key: 'orders', value: selectedDrivers?.[1], message: 'Thông tin driver không hợp lệ!' }
        ],
        customChecks: [
            { condition: selectedDrivers[0].id !== selectedDrivers[1].id, message: 'Vui lòng chọn 2 drivers khác nhau!' }
        ],
        successMessage: 'Điều kiện hợp lệ để thực hiện switch!'
    });
}

function getSwitchConfirmationInfo(selectedDrivers) {
    if (selectedDrivers.length !== 2) {
        return 'Vui lòng chọn đúng 2 drivers!';
    }

    const driver1 = selectedDrivers[0];
    const driver2 = selectedDrivers[1];

    let info = `Bạn có muốn hoán đổi orders giữa:\n\n`;
    info += `Driver ${driver1.id}:\n`;
    if (driver1.orders && driver1.orders.length > 0) {
        info += `  - Có ${driver1.orders.length} orders\n`;
    } else {
        info += `  - Chưa có orders\n`;
    }

    info += `Driver ${driver2.id}:\n`;
    if (driver2.orders && driver2.orders.length > 0) {
        info += `  - Có ${driver2.orders.length} orders\n`;
    } else {
        info += `  - Chưa có orders\n`;
    }

    info += `\nSau khi switch:\n`;
    info += `- Driver ${driver1.id} sẽ nhận orders từ Driver ${driver2.id}\n`;
    info += `- Driver ${driver2.id} sẽ nhận orders từ Driver ${driver1.id}`;

    return info;
}

// Xử lý sự kiện button Switch
async function switchButtonHandler(driverId1, driverId2) {
    try {
        const customers = await readDataFile('customers_qk7.json');
        const depots = await readDataFile('depots.json');

        const depot1 = depots.find(depot => depot.id === driverId1);
        const depot2 = depots.find(depot => depot.id === driverId2);

        if (!depot1 || !depot2) {
            return {
                success: false,
                message: 'Không tìm thấy thông tin driver/depot!'
            };
        }

        // Tạo dữ liệu driver với orders từ database
        const driver1 = {
            id: driverId1,
            name: depot1.name,
            orders: []
        };

        const driver2 = {
            id: driverId2,
            name: depot2.name,
            orders: []
        };
        const validation = validateSwitchConditions([driver1, driver2]);
        if (!validation.valid) {
            return {
                success: false,
                message: validation.message
            };
        }
        // Thực hiện switch orders
        const switchResult = await switchDrivers([driver1, driver2]);

        if (switchResult.success) {

            const confirmationInfo = getSwitchConfirmationInfo([driver1, driver2]);
            return {
                success: true,
                message: 'Switch thành công!',
                details: switchResult.details,
                confirmationInfo: confirmationInfo,
                depot1: depot1,
                depot2: depot2
            };
        } else {
            return switchResult;
        }

    } catch (error) {
        console.error('Lỗi khi xử lý button switch:', error);
        return {
            success: false,
            message: 'Có lỗi xảy ra: ' + error.message
        };
    }
}

/**
 * Lấy danh sách tất cả drivers (depots) để hiển thị
 * @returns {Promise<Array>} Danh sách drivers
 */
async function getAllDrivers() {
    try {
        const depots = await readDataFile('depots.json');
        return depots.map(depot => ({
            id: depot.id,
            name: depot.name,
            latitude: depot.latitude,
            longitude: depot.longitude
        }));
    } catch (error) {
        console.error('Lỗi khi lấy danh sách drivers:', error);
        throw error;
    }
}
/**
 * Hàm lấy thông tin chi tiết của một driver
 * @param {number} driverId - ID của driver
 * @returns {Promise<Object>} Thông tin chi tiết driver
 */
async function getDriverDetails(driverId) {
    try {
        const customers = await readDataFile('customers_qk7.json');
        const depots = await readDataFile('depots.json');

        const depot = depots.find(d => d.id === driverId);
        if (!depot) {
            throw new Error('Không tìm thấy driver!');
        }

        const orders = [];

        return {
            id: depot.id,
            name: depot.name,
            latitude: depot.latitude,
            longitude: depot.longitude,
            orders: orders
        };
    } catch (error) {
        console.error('Lỗi khi lấy thông tin driver:', error);
        throw error;
    }
}





/**
 * Các chức năng cho Add Customer
 */

/**
 * Lấy ID tiếp theo cho customer từ server
 * @returns {Promise<string>} ID tiếp theo dạng C0802
 */
async function getNextCustomerId() {
    try {
        const response = await fetch('http://127.0.0.1:8000/api/next-customer-id/');
        const result = await response.json();

        if (result.status === 'success') {
            return result.next_id;
        } else {
            throw new Error(result.message || 'Không thể lấy ID tiếp theo');
        }
    } catch (error) {
        console.error('Lỗi khi lấy next customer ID:', error);
        throw error;
    }
}

/**
 * Thêm khách hàng thủ công
 * @param {Object} customerData - Dữ liệu khách hàng
 * @returns {Promise<Object>} Kết quả thêm khách hàng
 */
async function addCustomerManual(customerData) {
    try {
        const response = await fetch('http://127.0.0.1:8000/api/add-customer/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(customerData)
        });

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Lỗi khi thêm khách hàng:', error);
        throw error;
    }
}

/**
 * Upload file Excel để thêm nhiều khách hàng
 * @param {File} excelFile - File Excel cần upload
 * @returns {Promise<Object>} Kết quả upload
 */
async function uploadCustomersExcel(excelFile) {
    try {
        const formData = new FormData();
        formData.append('file', excelFile);

        const response = await fetch('http://127.0.0.1:8000/api/add-customer/', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Lỗi khi upload file Excel:', error);
        throw error;
    }
}

/**
 * Validate dữ liệu khách hàng nhập tay
 * @param {Object} customerData - Dữ liệu khách hàng
 * @returns {Object} Kết quả validation
 */
function validateCustomerData(customerData) {
    const errors = [];

    if (!customerData.name || customerData.name.trim() === '') {
        errors.push('Tên khách hàng không được để trống');
    }

    if (!customerData.address || customerData.address.trim() === '') {
        errors.push('Địa chỉ không được để trống');
    }

    if (!customerData.phone || customerData.phone.trim() === '') {
        errors.push('Số điện thoại không được để trống');
    } else {
        // Validate định dạng số điện thoại Việt Nam
        const phoneRegex = /^(0|\+84)[3|5|7|8|9][0-9]{8}$/;
        if (!phoneRegex.test(customerData.phone.trim())) {
            errors.push('Số điện thoại không hợp lệ (VD: 0987654321 hoặc +84987654321)');
        }
    }

    if (customerData.email && customerData.email.trim() !== '') {
        // Validate định dạng email
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(customerData.email.trim())) {
            errors.push('Định dạng email không hợp lệ');
        }
    }

    // Validate tọa độ nếu có
    if (customerData.latitude && (isNaN(customerData.latitude) || customerData.latitude < -90 || customerData.latitude > 90)) {
        errors.push('Vĩ độ (latitude) phải là số từ -90 đến 90');
    }

    if (customerData.longitude && (isNaN(customerData.longitude) || customerData.longitude < -180 || customerData.longitude > 180)) {
        errors.push('Kinh độ (longitude) phải là số từ -180 đến 180');
    }

    return {
        isValid: errors.length === 0,
        errors: errors
    };
}

/**
 * Validate file Excel trước khi upload
 * @param {File} file - File Excel cần validate
 * @returns {Promise<Object>} Kết quả validation
 */
async function validateExcelFile(file) {
    const errors = [];

    // Kiểm tra định dạng file
    if (!file.name.endsWith('.xlsx') && !file.name.endsWith('.xls')) {
        errors.push('Chỉ chấp nhận file Excel (.xlsx hoặc .xls)');
        return { isValid: false, errors: errors };
    }

    // Kiểm tra kích thước file (tối đa 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        errors.push('Kích thước file không được vượt quá 10MB');
        return { isValid: false, errors: errors };
    }

    try {
        // Đọc file Excel để kiểm tra cấu trúc
        const buffer = await file.arrayBuffer();
        const workbook = XLSX.read(buffer, { type: 'buffer' });

        if (workbook.SheetNames.length === 0) {
            errors.push('File Excel không có sheet nào');
            return { isValid: false, errors: errors };
        }

        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        const jsonData = XLSX.utils.sheet_to_json(worksheet);

        if (jsonData.length === 0) {
            errors.push('Sheet không có dữ liệu');
            return { isValid: false, errors: errors };
        }

        // Kiểm tra các cột bắt buộc
        const requiredColumns = ['name', 'address', 'phone'];
        const headers = Object.keys(jsonData[0]);

        const missingColumns = requiredColumns.filter(col =>
            !headers.some(header => header.toLowerCase() === col.toLowerCase())
        );

        if (missingColumns.length > 0) {
            errors.push(`Thiếu các cột bắt buộc: ${missingColumns.join(', ')}`);
        }

        // Kiểm tra dữ liệu mẫu
        const sampleRow = jsonData[0];
        if (!sampleRow.name || !sampleRow.address || !sampleRow.phone) {
            errors.push('Dữ liệu trong file không đúng định dạng');
        }

        return {
            isValid: errors.length === 0,
            errors: errors,
            data: jsonData,
            headers: headers,
            rowCount: jsonData.length
        };

    } catch (error) {
        errors.push('Không thể đọc file Excel. Vui lòng kiểm tra định dạng file.');
        return { isValid: false, errors: errors };
    }
}

/**
 * Tạo mẫu file Excel để người dùng download
 * @returns {void}
 */
function downloadExcelTemplate() {
    const templateData = [
        {
            name: 'Nguyễn Văn A',
            address: '123 Đường ABC, Quận 1, TP.HCM',
            phone: '0987654321',
            email: 'nguyenvana@example.com',
            latitude: 10.8231,
            longitude: 106.6297
        },
        {
            name: 'Trần Thị B',
            address: '456 Đường XYZ, Quận 2, TP.HCM',
            phone: '0976543210',
            email: 'tranthib@example.com',
            latitude: 10.7769,
            longitude: 106.7009
        }
    ];

    const worksheet = XLSX.utils.json_to_sheet(templateData);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Customers');

    // Tự động điều chỉnh độ rộng cột
    const colWidths = [
        { wch: 20 }, // name
        { wch: 30 }, // address
        { wch: 15 }, // phone
        { wch: 25 }, // email
        { wch: 12 }, // latitude
        { wch: 12 }  // longitude
    ];
    worksheet['!cols'] = colWidths;

    XLSX.writeFile(workbook, 'customer_template.xlsx');
}

/**
 * Format số điện thoại Việt Nam
 * @param {string} phone - Số điện thoại cần format
 * @returns {string} Số điện thoại đã format
 */
function formatPhoneNumber(phone) {
    if (!phone) return '';

    // Loại bỏ tất cả ký tự không phải số
    let cleaned = phone.replace(/\D/g, '');

    // Nếu bắt đầu bằng 84, chuyển thành 0
    if (cleaned.startsWith('84')) {
        cleaned = '0' + cleaned.substring(2);
    }

    // Định dạng số điện thoại Việt Nam
    if (cleaned.length === 10 && cleaned.startsWith('0')) {
        return `${cleaned.substring(0, 4)} ${cleaned.substring(4, 7)} ${cleaned.substring(7)}`;
    }

    return phone; // Trả về nguyên gốc nếu không đúng định dạng
}

/**
 * Lấy thông tin vị trí từ địa chỉ (sử dụng API miễn phí)
 * @param {string} address - Địa chỉ cần tìm tọa độ
 * @returns {Promise<Object>} Tọa độ {latitude, longitude}
 */
async function getCoordinatesFromAddress(address) {
    try {
        // Sử dụng Nominatim API (OpenStreetMap)
        const encodedAddress = encodeURIComponent(address);
        const response = await fetch(
            `https://nominatim.openstreetmap.org/search?format=json&q=${encodedAddress}&limit=1&countrycodes=vn`
        );

        const data = await response.json();

        if (data && data.length > 0) {
            return {
                latitude: parseFloat(data[0].lat),
                longitude: parseFloat(data[0].lon)
            };
        } else {
            throw new Error('Không tìm thấy tọa độ cho địa chỉ này');
        }
    } catch (error) {
        console.error('Lỗi khi lấy tọa độ:', error);
        return { latitude: 0, longitude: 0 };
    }
}

/**
 * Refresh danh sách khách hàng sau khi thêm
 */
async function refreshCustomersList() {
    try {
        const response = await fetch('../data/customers.json');
        const customers = await response.json();

        // Cập nhật biến allCustomers trong frontend
        if (typeof window !== 'undefined' && window.allCustomers) {
            window.allCustomers = customers;
        }

        return customers;
    } catch (error) {
        console.error('Lỗi khi refresh danh sách khách hàng:', error);
        throw error;
    }
}

// ========================================
// CHỨC NĂNG ADD CUSTOMER (FRONTEND)
// ========================================

let currentCustomerId = '';

/**
 * Hiển thị modal thêm khách hàng
 */
async function showAddCustomerModal() {
    try {
        // Lấy ID tiếp theo từ server
        const response = await fetch('http://127.0.0.1:8000/api/next-customer-id/');
        const result = await response.json();

        if (result.status === 'success') {
            currentCustomerId = result.next_id;
        } else {
            throw new Error(result.message || 'Không thể lấy ID tiếp theo');
        }

        // Tạo modal HTML
        const modalHTML = `
            <div class="modal fade" id="addCustomerModal" tabindex="-1" aria-labelledby="addCustomerModalLabel" aria-hidden="true">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="addCustomerModalLabel">
                                <i class="fas fa-user-plus"></i> Thêm Khách Hàng Mới
                            </h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <ul class="nav nav-tabs" id="customerTabs" role="tablist">
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link active" id="manual-tab" data-bs-toggle="tab" data-bs-target="#manual-input" type="button" role="tab">
                                        <i class="fas fa-edit"></i> Nhập Thủ Công
                                    </button>
                                </li>
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link" id="excel-tab" data-bs-toggle="tab" data-bs-target="#excel-upload" type="button" role="tab">
                                        <i class="fas fa-file-excel"></i> Import Excel
                                    </button>
                                </li>
                            </ul>

                            <div class="tab-content mt-3" id="customerTabsContent">
                                <!-- Tab nhập thủ công -->
                                <div class="tab-pane fade show active" id="manual-input" role="tabpanel">
                                    <form id="manualCustomerForm">
                                        <div class="row">
                                            <div class="col-md-6">
                                                <div class="mb-3">
                                                    <label for="customerId" class="form-label">
                                                        <i class="fas fa-hashtag"></i> Customer ID
                                                    </label>
                                                    <input type="text" class="form-control" id="customerId" value="${currentCustomerId}" readonly>
                                                    <div class="form-text">ID được tạo tự động</div>
                                                </div>

                                                <div class="mb-3">
                                                    <label for="customerName" class="form-label">
                                                        <i class="fas fa-user"></i> Tên Khách Hàng <span class="text-danger">*</span>
                                                    </label>
                                                    <input type="text" class="form-control" id="customerName" required>
                                                    <div class="invalid-feedback">Vui lòng nhập tên khách hàng</div>
                                                </div>

                                                <div class="mb-3">
                                                    <label for="customerPhone" class="form-label">
                                                        <i class="fas fa-phone"></i> Số Điện Thoại <span class="text-danger">*</span>
                                                    </label>
                                                    <input type="tel" class="form-control" id="customerPhone" required>
                                                    <div class="form-text">VD: 0987654321 hoặc +84987654321</div>
                                                </div>

                                                <div class="mb-3">
                                                    <label for="customerEmail" class="form-label">
                                                        <i class="fas fa-envelope"></i> Email
                                                    </label>
                                                    <input type="email" class="form-control" id="customerEmail">
                                                    <div class="form-text">Không bắt buộc</div>
                                                </div>
                                            </div>

                                            <div class="col-md-6">
                                                <div class="mb-3">
                                                    <label for="customerAddress" class="form-label">
                                                        <i class="fas fa-map-marker-alt"></i> Địa Chỉ <span class="text-danger">*</span>
                                                    </label>
                                                    <textarea class="form-control" id="customerAddress" rows="3" required></textarea>
                                                    <div class="invalid-feedback">Vui lòng nhập địa chỉ</div>
                                                </div>

                                                <div class="row">
                                                    <div class="col-md-6">
                                                        <div class="mb-3">
                                                            <label for="customerLat" class="form-label">
                                                                <i class="fas fa-globe"></i> Vĩ Độ (Latitude)
                                                            </label>
                                                            <input type="number" class="form-control" id="customerLat" step="any" placeholder="10.8231">
                                                            <div class="form-text">Không bắt buộc</div>
                                                        </div>
                                                    </div>
                                                    <div class="col-md-6">
                                                        <div class="mb-3">
                                                            <label for="customerLng" class="form-label">
                                                                <i class="fas fa-globe"></i> Kinh Độ (Longitude)
                                                            </label>
                                                            <input type="number" class="form-control" id="customerLng" step="any" placeholder="106.6297">
                                                            <div class="form-text">Không bắt buộc</div>
                                                        </div>
                                                    </div>
                                                </div>


                                            </div>
                                        </div>
                                    </form>
                                </div>

                                <!-- Tab import Excel -->
                                <div class="tab-pane fade" id="excel-upload" role="tabpanel">
                                    <div class="mb-3">
                                        <h6><i class="fas fa-info-circle"></i> Hướng Dẫn Import Excel</h6>
                                        <div class="alert alert-info">
                                            <strong>File Excel phải có các cột sau:</strong>
                                            <ul class="mb-0 mt-2">
                                                <li><code>name</code> - Tên khách hàng (bắt buộc)</li>
                                                <li><code>address</code> - Địa chỉ (bắt buộc)</li>
                                                <li><code>phone</code> - Số điện thoại (bắt buộc)</li>
                                                <li><code>email</code> - Email (không bắt buộc)</li>
                                                <li><code>latitude</code> - Vĩ độ (không bắt buộc)</li>
                                                <li><code>longitude</code> - Kinh độ (không bắt buộc)</li>
                                            </ul>
                                        </div>
                                    </div>

                                    <div class="mb-3">
                                        <label for="excelFile" class="form-label">
                                            <i class="fas fa-file-excel"></i> Chọn File Excel <span class="text-danger">*</span>
                                        </label>
                                        <input type="file" class="form-control" id="excelFile" accept=".xlsx,.xls">
                                        <div class="form-text">Chỉ chấp nhận file .xlsx hoặc .xls, tối đa 10MB</div>
                                    </div>

                                    <div class="mb-3">
                                        <button type="button" class="btn btn-outline-success" onclick="downloadExcelTemplate()">
                                            <i class="fas fa-download"></i> Tải Mẫu File Excel
                                        </button>
                                    </div>

                                    <div id="excelPreview" class="d-none">
                                        <h6><i class="fas fa-eye"></i> Preview Dữ Liệu</h6>
                                        <div class="table-responsive">
                                            <table class="table table-sm table-bordered">
                                                <thead id="previewHeader"></thead>
                                                <tbody id="previewBody"></tbody>
                                            </table>
                                        </div>
                                        <div class="alert alert-warning">
                                            <small><i class="fas fa-exclamation-triangle"></i> Dữ liệu sẽ được thêm với ID tiếp theo: <strong id="nextIdPreview"></strong></small>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">
                                <i class="fas fa-times"></i> Đóng
                            </button>
                            <button type="button" class="btn btn-primary" onclick="submitCustomer()">
                                <i class="fas fa-save"></i> Thêm Khách Hàng
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Thêm modal vào body nếu chưa tồn tại
        if (!document.getElementById('addCustomerModal')) {
            document.body.insertAdjacentHTML('beforeend', modalHTML);
        }

        // Hiển thị modal
        const modal = new bootstrap.Modal(document.getElementById('addCustomerModal'));
        modal.show();

        // Reset form khi mở modal
        resetCustomerForm();

    } catch (error) {
        console.error('Lỗi khi hiển thị modal:', error);
        showNotification('Lỗi khi mở form thêm khách hàng!', 'error');
    }
}

/**
 * Reset form khách hàng
 */
function resetCustomerForm() {
    document.getElementById('customerId').value = currentCustomerId;
    document.getElementById('customerName').value = '';
    document.getElementById('customerPhone').value = '';
    document.getElementById('customerEmail').value = '';
    document.getElementById('customerAddress').value = '';
    document.getElementById('customerLat').value = '';
    document.getElementById('customerLng').value = '';
    document.getElementById('excelFile').value = '';

    // Ẩn preview Excel
    document.getElementById('excelPreview').classList.add('d-none');
}

/**
 * Lấy tọa độ từ địa chỉ (Frontend function)
 */
async function getCoordinatesFromAddress() {
    const address = document.getElementById('customerAddress').value.trim();

    if (!address) {
        showNotification('Vui lòng nhập địa chỉ trước!', 'warning');
        return;
    }

    showLoading();

    try {
        // Sử dụng Nominatim API (OpenStreetMap)
        const encodedAddress = encodeURIComponent(address + ', Vietnam');
        const response = await fetch(
            `https://nominatim.openstreetmap.org/search?format=json&q=${encodedAddress}&limit=1&countrycodes=vn`
        );

        const data = await response.json();

        if (data && data.length > 0) {
            const lat = parseFloat(data[0].lat);
            const lng = parseFloat(data[0].lon);

            document.getElementById('customerLat').value = lat;
            document.getElementById('customerLng').value = lng;

            showNotification(`Đã tìm thấy tọa độ: ${lat}, ${lng}`, 'success');
        } else {
            showNotification('Không tìm thấy tọa độ cho địa chỉ này. Vui lòng nhập thủ công.', 'warning');
        }
    } catch (error) {
        console.error('Lỗi khi lấy tọa độ:', error);
        showNotification('Lỗi khi tìm tọa độ. Vui lòng nhập thủ công.', 'error');
    }

    hideLoading();
}

/**
 * Tải mẫu file Excel (Frontend function)
 */
function downloadExcelTemplate() {
    const templateData = [
        {
            name: 'Nguyễn Văn A',
            address: '123 Đường ABC, Quận 1, TP.HCM',
            phone: '0987654321',
            email: 'nguyenvana@example.com',
            latitude: 10.8231,
            longitude: 106.6297
        },
        {
            name: 'Trần Thị B',
            address: '456 Đường XYZ, Quận 2, TP.HCM',
            phone: '0976543210',
            email: 'tranthib@example.com',
            latitude: 10.7769,
            longitude: 106.7009
        }
    ];

    const worksheet = XLSX.utils.json_to_sheet(templateData);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Customers');

    // Tự động điều chỉnh độ rộng cột
    const colWidths = [
        { wch: 20 }, // name
        { wch: 30 }, // address
        { wch: 15 }, // phone
        { wch: 25 }, // email
        { wch: 12 }, // latitude
        { wch: 12 }  // longitude
    ];
    worksheet['!cols'] = colWidths;

    XLSX.writeFile(workbook, 'customer_template.xlsx');
}

/**
 * Submit khách hàng (nhập tay hoặc Excel)
 */
async function submitCustomer() {
    const activeTab = document.querySelector('#customerTabs .nav-link.active');

    if (activeTab.id === 'manual-tab') {
        await submitManualCustomer();
    } else if (activeTab.id === 'excel-tab') {
        await submitExcelCustomers();
    }
}

/**
 * Submit khách hàng nhập tay
 */
async function submitManualCustomer() {
    // Validate form
    const name = document.getElementById('customerName').value.trim();
    const address = document.getElementById('customerAddress').value.trim();
    const phone = document.getElementById('customerPhone').value.trim();
    const email = document.getElementById('customerEmail').value.trim();
    const latitude = document.getElementById('customerLat').value;
    const longitude = document.getElementById('customerLng').value;

    if (!name || !address || !phone) {
        showNotification('Vui lòng điền đầy đủ thông tin bắt buộc!', 'error');
        return;
    }

    // Validate số điện thoại
    const phoneRegex = /^(0|\+84)[3|5|7|8|9][0-9]{8}$/;
    if (!phoneRegex.test(phone)) {
        showNotification('Số điện thoại không hợp lệ!', 'error');
        return;
    }

    // Validate tọa độ nếu có
    if (latitude && (isNaN(latitude) || latitude < -90 || latitude > 90)) {
        showNotification('Vĩ độ không hợp lệ!', 'error');
        return;
    }

    if (longitude && (isNaN(longitude) || longitude < -180 || longitude > 180)) {
        showNotification('Kinh độ không hợp lệ!', 'error');
        return;
    }

    const customerData = {
        name: name,
        address: address,
        phone: phone,
        email: email || '',
        latitude: latitude ? parseFloat(latitude) : 0,
        longitude: longitude ? parseFloat(longitude) : 0
    };

    showLoading();

    try {
        const response = await fetch('http://127.0.0.1:8000/api/add-customer/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(customerData)
        });

        const result = await response.json();
        hideLoading();

        if (result.status === 'success') {
            showNotification(`Đã thêm khách hàng ${result.customer.id} thành công!`, 'success');

            // Đóng modal và refresh dữ liệu
            const modal = bootstrap.Modal.getInstance(document.getElementById('addCustomerModal'));
            modal.hide();

            // Refresh danh sách khách hàng
            await refreshCustomersList();
            populateOrdersTable();

            // Thêm marker mới lên bản đồ
            addCustomerToMap(result.customer);

        } else {
            showNotification(result.message || 'Không thể thêm khách hàng!', 'error');
        }

    } catch (error) {
        hideLoading();
        console.error('Lỗi khi thêm khách hàng:', error);
        showNotification('Lỗi kết nối đến server!', 'error');
    }
}

/**
 * Submit khách hàng từ Excel
 */
async function submitExcelCustomers() {
    const fileInput = document.getElementById('excelFile');

    if (!fileInput.files[0]) {
        showNotification('Vui lòng chọn file Excel!', 'error');
        return;
    }

    showLoading();

    try {
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        const response = await fetch('http://127.0.0.1:8000/api/add-customer/', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        hideLoading();

        if (result.status === 'success') {
            showNotification(`Đã thêm ${result.added_customers.length} khách hàng từ Excel thành công!`, 'success');

            // Đóng modal và refresh dữ liệu
            const modal = bootstrap.Modal.getInstance(document.getElementById('addCustomerModal'));
            modal.hide();

            // Refresh danh sách khách hàng
            await refreshCustomersList();
            populateOrdersTable();

            // Thêm các marker mới lên bản đồ
            result.added_customers.forEach(customer => {
                addCustomerToMap(customer);
            });

        } else {
            showNotification(result.message || 'Không thể thêm khách hàng từ Excel!', 'error');
        }

    } catch (error) {
        hideLoading();
        console.error('Lỗi khi upload Excel:', error);
        showNotification('Lỗi kết nối đến server!', 'error');
    }
}

/**
 * Thêm khách hàng mới lên bản đồ với cơ chế retry thông minh
 */
function addCustomerToMap(customer) {
    try {
        // Kiểm tra điều kiện cơ bản
        if (!customer.latitude || !customer.longitude) {
            console.log('Khách hàng không có tọa độ, bỏ qua việc thêm marker');
            return;
        }

        // Hàm kiểm tra map có sẵn sàng không
        function isMapReady() {
            return typeof window !== 'undefined' &&
                   window.map &&
                   typeof window.map.addLayer === 'function' &&
                   window.map._container; // Kiểm tra container element tồn tại
        }

        // Nếu map chưa sẵn sàng, thử lại với exponential backoff
        if (!isMapReady()) {
            console.log('⏳ Map chưa sẵn sàng, thử lại sau...');

            // Tìm attempt tiếp theo (nếu có)
            const attempt = (addCustomerToMap.attempt || 0) + 1;
            addCustomerToMap.attempt = attempt;

            // Giới hạn số lần thử (tối đa 10 lần)
            if (attempt >= 10) {
                console.error('❌ Không thể thêm marker sau 10 lần thử:', customer.name);
                return;
            }

            // Exponential backoff: 100ms, 200ms, 400ms, 800ms, ...
            const delay = Math.min(100 * Math.pow(2, attempt - 1), 5000);

            setTimeout(() => addCustomerToMap(customer), delay);
            return;
        }

        // Reset attempt counter khi thành công
        addCustomerToMap.attempt = 0;

        // Tạo icon cho customer
        const customerIcon = L.icon({
            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowSize: [41, 41]
        });

        // Tạo marker và thêm vào bản đồ
        const marker = L.marker([customer.latitude, customer.longitude], { icon: customerIcon });
        marker.bindPopup(`
            <b>${customer.name}</b><br>
            ID: ${customer.id}<br>
            Địa chỉ: ${customer.address}<br>
            SĐT: ${customer.phone}<br>
            Email: ${customer.email || 'N/A'}
        `);

        // Thêm marker vào bản đồ một cách an toàn
        try {
            marker.addTo(window.map);
            console.log(`✅ Đã thêm marker cho khách hàng ${customer.name} lên bản đồ`);
        } catch (mapError) {
            console.error('❌ Lỗi khi thêm marker vào bản đồ:', mapError);

            // Thử lại một lần nữa sau 1 giây
            setTimeout(() => {
                try {
                    marker.addTo(window.map);
                    console.log(`✅ Đã thêm marker sau khi thử lại: ${customer.name}`);
                } catch (retryError) {
                    console.error('❌ Không thể thêm marker sau khi thử lại:', retryError);
                }
            }, 1000);
        }

    } catch (error) {
        console.error('❌ Lỗi trong hàm addCustomerToMap:', error);
    }
}

/**
 * Xử lý khi chọn file Excel
 */
async function handleExcelFileSelection(file) {
    if (!file) return;

    const previewDiv = document.getElementById('excelPreview');
    const previewHeader = document.getElementById('previewHeader');
    const previewBody = document.getElementById('previewBody');
    const nextIdPreview = document.getElementById('nextIdPreview');

    showLoading();
    try {
        const buffer = await file.arrayBuffer();
        const workbook = XLSX.read(buffer, { type: 'buffer' });
        if (workbook.SheetNames.length === 0) {
            throw new Error('File Excel không có sheet nào');
        }
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        const jsonData = XLSX.utils.sheet_to_json(worksheet);
        if (jsonData.length === 0) {
            throw new Error('Sheet không có dữ liệu');
        }
        const headers = Object.keys(jsonData[0]);
        const requiredColumns = ['name', 'address', 'phone'];
        const missingColumns = requiredColumns.filter(col =>
            !headers.some(header => header.toLowerCase() === col.toLowerCase())
        );
        if (missingColumns.length > 0) {
            throw new Error(`Thiếu các cột bắt buộc: ${missingColumns.join(', ')}`);
        }
        nextIdPreview.textContent = currentCustomerId;
        let headerHTML = '<tr>';
        headers.forEach(header => {
            headerHTML += `<th>${header}</th>`;
        });
        headerHTML += '</tr>';
        previewHeader.innerHTML = headerHTML;
        let bodyHTML = '';
        const previewRows = jsonData.slice(0, 5);
        previewRows.forEach((row, index) => {
            bodyHTML += '<tr>';
            headers.forEach(header => {
                bodyHTML += `<td>${row[header] || ''}</td>`;
            });
            bodyHTML += '</tr>';
        });
        if (jsonData.length > 5) {
            bodyHTML += `<tr><td colspan="${headers.length}" class="text-center text-muted">... và ${jsonData.length - 5} dòng khác</td></tr>`;
        }
        previewBody.innerHTML = bodyHTML;
        previewDiv.classList.remove('d-none');
        showNotification(`File hợp lệ! Sẽ thêm ${jsonData.length} khách hàng.`, 'success');
    } catch (error) {
        console.error('Lỗi khi đọc file Excel:', error);
        showNotification(`Lỗi: ${error.message}`, 'error');
        previewDiv.classList.add('d-none');
    }

    hideLoading();
}

function initAddCustomerEvents() {
    document.addEventListener('change', function(e) {
        if (e.target.id === 'excelFile') {
            handleExcelFileSelection(e.target.files[0]);
        }
    });
}
if (typeof window !== 'undefined') {
    window.showAddCustomerModal = showAddCustomerModal;
    window.resetCustomerForm = resetCustomerForm;
    window.getCoordinatesFromAddress = getCoordinatesFromAddress;
    window.downloadExcelTemplate = downloadExcelTemplate;
    window.submitCustomer = submitCustomer;
    window.submitManualCustomer = submitManualCustomer;
    window.submitExcelCustomers = submitExcelCustomers;
    window.addCustomerToMap = addCustomerToMap;
    window.handleExcelFileSelection = handleExcelFileSelection;
    window.initAddCustomerEvents = initAddCustomerEvents;
}