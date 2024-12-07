import requests
import json
import urllib3
import csv

# Tắt cảnh báo SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class StudentAPI:
    def __init__(self, login_url, api_url, headers):
        self.login_url = login_url
        self.api_url = api_url
        self.headers = headers
        self.access_token = None

    def get_access_token(self, username, password):
        login_data = {
            'client_id': 'education_client',
            'grant_type': 'password',
            'username': username,
            'password': password,
            'client_secret': 'password',
        }

        # Gửi yêu cầu POST để lấy token
        login_response = requests.post(self.login_url, headers=self.headers, data=login_data, verify=False)

        # Kiểm tra mã trạng thái HTTP khi đăng nhập
        if login_response.status_code == 200:
            try:
                login_data = login_response.json()
                self.access_token = login_data.get('access_token')
                return self.access_token
            except ValueError as e:
                print("Lỗi khi phân tích JSON:", e)
                return None
        else:
            print("Đăng nhập không thành công. Mã lỗi:", login_response.status_code)
            return None

    def get_student_marks(self, subject_codes, diff_subject_codes):
        if not self.access_token:
            print("Không có access token.")
            return []

        api_headers = {
            "Authorization": f"Bearer {self.access_token}"
        }

        # Gửi yêu cầu GET để lấy dữ liệu điểm
        api_response = requests.get(self.api_url, headers=api_headers, verify=False)

        # Kiểm tra kết quả trả về từ API
        if api_response.status_code == 200:
            try:
                result = api_response.json()
                # Lọc và tính toán điểm trung bình ngay trên dữ liệu nhận được
                return self.calculate_average_marks(result, subject_codes, diff_subject_codes)
            except ValueError:
                print("Dữ liệu trả về không phải là JSON hợp lệ.")
                return []
        else:
            print(f"Error {api_response.status_code}: {api_response.text}")
            return []
    
    def calculate_average_marks(self, data, subject_codes, diff_subject_codes):
        subject_marks = []
        processed_subjects = set()  # Tạo một tập hợp để lưu các mã môn học đã được xử lý

        def process_record(record, subject_code):
            subject = record.get("subject", {})
            student = record.get("student", {})
            average_mark = record.get("mark")
            first_name = student.get("firstName")
            last_name = student.get("lastName")

            if average_mark is not None:
                full_name = f"{last_name} {first_name}"
                subject_marks.append({
                    "Full Name": full_name,
                    "Subject Code": subject_code,
                    "Average Mark": average_mark
                })

        # Lặp qua các mã môn học trong subject_codes
        for i, subject_code in enumerate(subject_codes):
            # Kiểm tra trong data với mã môn học trong subject_codes
            found_in_data = False
            for record in data:
                if record.get("subject", {}).get("subjectCode") == subject_code:
                    process_record(record, subject_code)
                    processed_subjects.add(subject_code)  # Đánh dấu môn học đã được xử lý
                    found_in_data = True
                    break  # Nếu tìm thấy, không cần kiểm tra tiếp

            # Nếu không tìm thấy trong data, kiểm tra tương ứng với mã trong diff_subject_codes
            if not found_in_data and i < len(diff_subject_codes):
                diff_subject_code = diff_subject_codes[i]
                for record in data:
                    if record.get("subject", {}).get("subjectCode") == diff_subject_code:
                        process_record(record, diff_subject_code)
                        processed_subjects.add(diff_subject_code)  # Đánh dấu môn học đã được xử lý
                        break  # Nếu tìm thấy, không cần kiểm tra tiếp

        return subject_marks

class JSONHandler:
    def __init__(self):
        pass

    def get_as_json(self, data):
        # Trả về dữ liệu dưới dạng chuỗi JSON
        return json.dumps(data, ensure_ascii=False, indent=4)

class AccountProcessor:
    def __init__(self, student_api, json_handler):
        self.student_api = student_api
        self.json_handler = json_handler
        self.result_json = None  # Khởi tạo biến lưu kết quả JSON

    def process_accounts(self, username, password, subject_codes, diff_subject_codes):
        print(f"Đang xử lý tài khoản: {username}")
        access_token = self.student_api.get_access_token(username, password)
        if access_token:
            # Lấy dữ liệu điểm và lọc luôn kết quả
            average_marks = self.student_api.get_student_marks(subject_codes, diff_subject_codes)

            # Lưu kết quả vào biến nếu có dữ liệu
            if average_marks:
                self.result_json = self.json_handler.get_as_json(average_marks)
                self.save_to_csv(average_marks)
            else:
                print("Không có dữ liệu điểm phù hợp.")
        else:
            print(f"Không thể lấy token cho tài khoản {username}")

    def get_result_json(self):
        return self.result_json  # Trả về kết quả JSON đã lưu
    @staticmethod
    def save_to_csv(data, filename='D:\công việc\H\He thong kinh doanj thong minh\ServerTest\serverf\static\group_3_cleaned.csv'):
        # Mở tệp CSV để ghi dữ liệu
        with open(filename, mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            
            # Nếu file rỗng, ghi tiêu đề cột
            if file.tell() == 0:
                writer.writerow(['Full Name', 'Subject Code', 'Average Mark'])
            
            # Ghi dữ liệu vào file CSV
            for record in data:
                writer.writerow([
                    record['Full Name'],
                    record['Subject Code'],
                    record['Average Mark']
                ])
        print(f"Dữ liệu đã được lưu vào {filename}")