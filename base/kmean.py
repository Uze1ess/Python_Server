import json
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
    
class TeamClustering:
    def __init__(self, data, n_clusters=4, max_group_size=5):
        self.df = data
        self.n_clusters = n_clusters
        self.max_group_size = max_group_size
        self.scaled_features = None
        self.kmeans = None
        self.compatibility_df = None
        self.final_df = None

    def load_data(self):
        """Load dữ liệu từ file CSV."""
        data = pd.read_csv(self.file_path)
        self.df = pd.DataFrame(data)
    
    def shuffle_data(self):
        """Đảo lộn thứ tự các bản ghi và in ra chỉ số sau khi thay đổi."""
        # Lấy danh sách chỉ số ban đầu
        original_index = self.df.index.tolist()

        # Đảo lộn chỉ số
        shuffled_index = np.random.permutation(original_index)

        # Áp dụng lại thứ tự đảo lộn vào DataFrame
        self.df = self.df.iloc[shuffled_index].reset_index(drop=True)

        # In ra chỉ số sau khi đảo lộn
        print("Chỉ số sau khi thay đổi:", shuffled_index.tolist())

    def preprocess_data(self):
        """Mã hóa và chuẩn hóa dữ liệu."""
        self.interest_encoder = LabelEncoder()
        self.teamwork_encoder = LabelEncoder()
        self.work_preference_encoder = LabelEncoder()
        self.information_encoder = LabelEncoder()

        # Mã hóa các cột phân loại
        self.df["Sở thích"] = self.interest_encoder.fit_transform(self.df["Sở thích"])
        self.df["Kĩ năng làm việc"] = self.teamwork_encoder.fit_transform(self.df["Kĩ năng làm việc"])
        self.df["Hoạt động chính"] = self.work_preference_encoder.fit_transform(self.df["Hoạt động chính"])
        self.df["Nguồn thông tin"] = self.information_encoder.fit_transform(self.df["Nguồn thông tin"])

        # Chuyển các điểm tổng kết thành giá trị số nếu cần
        score_mapping = {"Kém": 1, "Yếu": 2, "Trung bình": 3, "Khá": 4, "Giỏi": 5}
        self.df["Điểm tổng kết môn QT HTTT"] = self.df["Điểm tổng kết môn QT HTTT"].map(score_mapping)
        self.df["Khai phá dữ liệu"] = self.df["Khai phá dữ liệu"].map(score_mapping)
        self.df["Học máy"] = self.df["Học máy"].map(score_mapping)

        # Chọn các cột để chuẩn hóa
        features = ["Điểm tổng kết môn QT HTTT", "Khai phá dữ liệu", "Học máy", "Sở thích", "Kĩ năng làm việc", "Hoạt động chính", "Nguồn thông tin"]
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.df[features])

    def cluster_data(self):
        """Phân cụm dữ liệu bằng KMeans."""
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        self.kmeans.fit(self.scaled_features)

    def calculate_compatibility(self):
        """Tính toán độ tương thích và tạo DataFrame kết quả."""
        distances = self.kmeans.transform(self.scaled_features)
        compatibility = 1 / (distances + 1e-10)
        compatibility = compatibility / compatibility.sum(axis=1, keepdims=True) * 100
        compatibility = compatibility.round(2)
        
        compatibility_df = pd.DataFrame(
            compatibility,
            columns=[f"Nhóm {i+1}" for i in range(self.n_clusters)],
            # index=self.shuffled_df["Tên"]
            index=self.df["Tên"]
        )
        compatibility_df.reset_index(inplace=True)
        compatibility_df.rename(columns={"index": "Tên"}, inplace=True)
        
        compatibility_df = compatibility_df.merge(
            self.df[["Tên", "Điểm tổng kết môn QT HTTT", "Khai phá dữ liệu", "Học máy", "Sở thích", "Kĩ năng làm việc", "Hoạt động chính", "Nguồn thông tin"]],
            on="Tên"
        )
        
        self.compatibility_df = compatibility_df

    def assign_groups(self):
        """Gán nhóm cho sinh viên dựa trên độ tương thích."""
        final_groups = {i: [] for i in range(self.n_clusters)}

        for _, student in self.compatibility_df.iterrows():
            ranked_groups = student[1:-3].sort_values(ascending=False).index
            for group in ranked_groups:
                group_index = int(group.split(" ")[1]) - 1
                if len(final_groups[group_index]) < self.max_group_size:
                    final_groups[group_index].append(student["Tên"])
                    break

        final_result = []
        for group, members in final_groups.items():
            for member in members:
                row = self.compatibility_df[self.compatibility_df["Tên"] == member].iloc[0]
                final_result.append({
                    "Tên": row["Tên"],
                    "Nhóm": group + 1,
                    "Độ tương thích": row[1:self.n_clusters+1].to_list()  # Chuyển độ tương thích thành mảng
                    # "Kỹ năng lập trình": row["Kỹ năng lập trình"],
                    # "Kỹ năng làm việc nhóm": row["Kỹ năng làm việc nhóm"],
                    # "Sở thích làm việc": row["Sở thích làm việc"]
                })

        self.final_df = pd.DataFrame(final_result)

    def to_json(self, output_file):
        """Xuất kết quả ra file JSON."""
        self.final_df.to_json(output_file, orient="records", force_ascii=False, indent=4)
    
    def save_model(self, file_name="kmeans_model.pkl"):
        """Lưu model KMeans ra file."""
        if self.kmeans:
            joblib.dump(self.kmeans, file_name)
            print(f"Model đã được lưu vào file {file_name}")
        else:
            raise ValueError("Model chưa được huấn luyện. Vui lòng chạy cluster_data() trước khi lưu.")
        
    def load_model(self, file_name="kmeans_model.pkl"):
        """Tải model KMeans từ file."""
        try:
            self.kmeans = joblib.load(file_name)
            print(f"Model đã được tải từ file {file_name}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file_name} không tồn tại. Vui lòng kiểm tra đường dẫn.")
        
    def run(self, model_file=None, save_model_file=None):
        """Thực hiện toàn bộ quy trình."""
        # self.load_data()
        # Đảo lộn dữ liệu và in ra chỉ số
        # team_clustering.shuffle_data()
        self.preprocess_data()
        # self.shuffle_data()

        # Sử dụng dữ liệu đã đảo
        # data_to_use = self.shuffled_df  # Chọn dữ liệu đã đảo lộn
        
        if model_file:  # Nếu có model file, tải model
            self.load_model(model_file)
        else:
            self.cluster_data()  # Huấn luyện model
        
        self.calculate_compatibility()
        self.assign_groups()
        
        if save_model_file:  # Nếu có yêu cầu lưu model
            self.save_model(save_model_file)
        
        return self.final_df
    

    def add_student_and_predict(self, array):
        if not self.kmeans:
            raise ValueError("Model chưa được tải hoặc huấn luyện. Vui lòng tải model hoặc huấn luyện trước.")
        
        name = array[0]
        qt_score = array[1]  # Điểm tổng kết môn QT HTTT
        data_mining_skill = array[2]  # Khai phá dữ liệu
        machine_learning_skill = array[3]  # Học máy
        work_preference = array[4]  # Sở thích làm việc

        # Chuẩn bị dữ liệu mới
        new_student_df = pd.DataFrame([{
            "Tên": name,
            "Điểm tổng kết môn QT HTTT": qt_score,
            "Khai phá dữ liệu": data_mining_skill,
            "Học máy": machine_learning_skill,
            "Sở thích": work_preference,
            "Kĩ năng làm việc": array[5],  # Giá trị này sẽ được mã hóa sau
            "Hoạt động chính": array[6],  # Giá trị này sẽ được mã hóa sau
            "Nguồn thông tin": array[7]   # Giá trị này sẽ được mã hóa sau
        }])

        # Mã hóa các giá trị phân loại sử dụng bộ mã hóa đã huấn luyện từ preprocess_data
        try:
            new_student_df["Sở thích"] = self.interest_encoder.transform([array[4]])[0]
            new_student_df["Kĩ năng làm việc"] = self.teamwork_encoder.transform([array[5]])[0]
            new_student_df["Hoạt động chính"] = self.work_preference_encoder.transform([array[6]])[0]
            new_student_df["Nguồn thông tin"] = self.information_encoder.transform([array[7]])[0]
        except ValueError:
            print("Lỗi: Giá trị mới không có trong dữ liệu gốc. Sử dụng giá trị mặc định.")
            new_student_df["Sở thích"] = -1
            new_student_df["Kĩ năng làm việc"] = -1
            new_student_df["Hoạt động chính"] = -1
            new_student_df["Nguồn thông tin"] = -1
        # Mã hóa các cột điểm số (Điểm tổng kết môn QT HTTT, Khai phá dữ liệu, Học máy)
        
        # new_student_df["Điểm tổng kết môn QT HTTT"] = new_student_df["Điểm tổng kết môn QT HTTT"].map(score_mapping)
        # new_student_df["Khai phá dữ liệu"] = new_student_df["Khai phá dữ liệu"].map(score_mapping)
        # new_student_df["Học máy"] = new_student_df["Học máy"].map(score_mapping)

        def map_score_to_range(score):
            if 8.5 <= score <= 10:
                return 'Giỏi'
            elif 7 <= score <= 8.4:
                return 'Khá'
            elif 5.5 <= score <= 6.9:
                return 'Trung bình'
            elif 4 <= score <= 5.4:
                return 'Yếu'
            else:
                return 'Kém'
        score_mapping = {"Kém": 1, "Yếu": 2, "Trung bình": 3, "Khá": 4, "Giỏi": 5}
        # Áp dụng hàm cho từng cột
        new_student_df["Điểm tổng kết môn QT HTTT"] = new_student_df["Điểm tổng kết môn QT HTTT"].apply(map_score_to_range).map(score_mapping)
        new_student_df["Khai phá dữ liệu"] = new_student_df["Khai phá dữ liệu"].apply(map_score_to_range).map(score_mapping)
        new_student_df["Học máy"] = new_student_df["Học máy"].apply(map_score_to_range).map(score_mapping)

        # Chỉ chuẩn hóa các cột có giá trị số
        features = ["Điểm tổng kết môn QT HTTT", "Khai phá dữ liệu", "Học máy", "Sở thích", "Kĩ năng làm việc", "Hoạt động chính", "Nguồn thông tin"]

        # Chuẩn hóa dữ liệu với StandardScaler đã được huấn luyện trước đó
        new_student_scaled = self.scaler.transform(new_student_df[features])

        # Dự đoán nhóm
        group = self.kmeans.predict(new_student_scaled)[0] + 1
        distances = self.kmeans.transform(new_student_scaled)[0]
        compatibility = (1 / (distances + 1e-10)) / sum(1 / (distances + 1e-10)) * 100
        
        return {
            "Tên": name,
            "Nhóm được phân": group,
            "Độ tương thích": compatibility.round(2).tolist(),
            "Thông tin gốc": {
                "Tên": name,
                "Điểm tổng kết môn QT HTTT": qt_score,
                "Khai phá dữ liệu": data_mining_skill,
                "Học máy": machine_learning_skill,
                "Sở thích": work_preference,
                "Kĩ năng làm việc": array[5],  # Kỹ năng làm việc
                "Hoạt động chính": array[6],  # Hoạt động chính
                "Nguồn thông tin": array[7]
            }
        }