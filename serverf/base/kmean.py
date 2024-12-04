# import numpy as np
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# class TeamClustering:
#     def __init__(self, data, k):
#         self.data = data
#         self.k = k
#     def get_encoder_number(self):
#         return self.scaler
#     def get_encoder_label(self):
#         return self.label_encoders
#     def get_centroids(self):
#         return self.kmeans.cluster_centers_
#     def get_kmean(self):
#         return self.kmeans
#     def read_data(self):
#         self.df = pd.DataFrame(self.data)
#         self.df = self.df.iloc[:, :-2]
#         return self.df
#     def encoder(self):
#         self.scaler = StandardScaler()
#         self.numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
#         self.df[self.numeric_cols] = self.scaler.fit_transform(self.df[self.numeric_cols])

#         self.label_encoders = {}
#         for col in self.df.select_dtypes(include=['object']).columns:
#             self.le = LabelEncoder()
#             self.df[col] = self.le.fit_transform(self.df[col])
#             self.label_encoders[col] = self.le
#         return self.df
#     def train(self):
#         self.kmeans = KMeans(n_clusters=self.k, init='k-means++', random_state=42)
#         self.kmeans.fit(self.df)
#     def cal_similarity(self,test_data):
#         test_data = pd.DataFrame(test_data, columns=self.df.columns)
#         test_data[self.numeric_cols] = self.scaler.transform(test_data[self.numeric_cols])

#         for col in self.label_encoders.keys():
#             test_data[col] = self.label_encoders[col].transform(test_data[col])

#         distances = np.linalg.norm(self.get_centroids() - test_data.values, axis=1)
        
#         compatibility_scores = 1 / (1 + distances)

#         clusters_with_compatibility = list(enumerate(compatibility_scores))
#         self.sorted_clusters = sorted(clusters_with_compatibility, key=lambda x: x[1], reverse=True)
#         return self.sorted_clusters
#     def get_best_compatibility(self):
#         if hasattr(self, 'sorted_clusters') and self.sorted_clusters:
#             return self.sorted_clusters[0]  # Trả về tuple có độ tương thích lớn nhất
#         else:
#             return None

# import json
# import joblib
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# class TeamClustering:
#     def __init__(self, file_path, n_clusters=4, max_group_size=5):
#         self.file_path = file_path
#         self.n_clusters = n_clusters
#         self.max_group_size = max_group_size
#         self.df = None
#         self.scaled_features = None
#         self.kmeans = None
#         self.compatibility_df = None
#         self.final_df = None

#     def load_data(self):
#         """Load dữ liệu từ file CSV."""
#         data = pd.read_csv(self.file_path)
#         self.df = pd.DataFrame(data)
#     def convert_to_grade(self,score):
#         try:
#             score = float(score)
#         except ValueError:
#             return 'Không hợp lệ'
    
#         if 8.5 <= score <= 10:
#             return 'Giỏi'
#         elif 7 <= score <= 8.4:
#             return 'Khá'
#         elif 5.5 <= score <= 6.9:
#             return 'Trung bình'
#         elif 4 <= score <= 5.4:
#             return 'Yếu'
#         else:
#             return 'Kém'
#     def encode_score(self):
#         pass
#     def get_encoder(self):
#         return self.work_preference_encoder, self.teamwork_encoder, self.scaler
#     def preprocess_data(self):
#         """Mã hóa và chuẩn hóa dữ liệu."""
#         self.teamwork_encoder = LabelEncoder()
#         self.work_preference_encoder = LabelEncoder()
#         # Điểm tổng kết môn QT HTTT,Khai phá dữ liệu,Học máy,Sở thích,Kĩ năng làm việc,Hoạt động chính

#         # Mã hóa các cột phân loại
#         self.df["Sở thích"] = self.work_preference_encoder.fit_transform(self.df["Sở thích"])
#         self.df["Kĩ năng làm việc"] = self.teamwork_encoder.fit_transform(self.df["Kĩ năng làm việc"])
#         self.df["Hoạt động chính"] = self.work_preference_encoder.fit_transform(self.df["Hoạt động chính"])
#         self.df["Nguồn thông tin"] = self.work_preference_encoder.fit_transform(self.df["Nguồn thông tin"])

#         # Chuyển các điểm tổng kết thành giá trị số nếu cần
#         score_mapping = {"Kém": 1, "Yếu": 2, "Trung bình": 3, "Khá": 3, "Giỏi": 5}
#         self.df["Điểm tổng kết môn QT HTTT"] = self.df["Điểm tổng kết môn QT HTTT"].map(score_mapping)
#         self.df["Khai phá dữ liệu"] = self.df["Khai phá dữ liệu"].map(score_mapping)
#         self.df["Học máy"] = self.df["Học máy"].map(score_mapping)

#         # Chọn các cột để chuẩn hóa
#         features = ["Điểm tổng kết môn QT HTTT", "Khai phá dữ liệu", "Học máy", "Sở thích", "Kĩ năng làm việc", "Hoạt động chính", "Nguồn thông tin"]
#         self.scaler = StandardScaler()
#         self.scaled_features = self.scaler.fit_transform(self.df[features])

#     def cluster_data(self):
#         """Phân cụm dữ liệu bằng KMeans."""
#         self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
#         self.kmeans.fit(self.scaled_features)

#     def calculate_compatibility(self):
#         """Tính toán độ tương thích và tạo DataFrame kết quả."""
#         distances = self.kmeans.transform(self.scaled_features)
#         compatibility = 1 / (distances + 1e-10)
#         compatibility = compatibility / compatibility.sum(axis=1, keepdims=True) * 100
#         compatibility = compatibility.round(2)
        
#         compatibility_df = pd.DataFrame(
#             compatibility,
#             columns=[f"Nhóm {i+1}" for i in range(self.n_clusters)],
#             # index=self.shuffled_df["Tên"]
#             index=self.df["Tên"]
#         )
#         compatibility_df.reset_index(inplace=True)
#         compatibility_df.rename(columns={"index": "Tên"}, inplace=True)
        
#         compatibility_df = compatibility_df.merge(
#             self.df[["Tên", "Điểm tổng kết môn QT HTTT", "Khai phá dữ liệu", "Học máy", "Sở thích", "Kĩ năng làm việc", "Hoạt động chính", "Nguồn thông tin"]],
#             on="Tên"
#         )
        
#         self.compatibility_df = compatibility_df

#     def assign_groups(self):
#         """Gán nhóm cho sinh viên dựa trên độ tương thích."""
#         final_groups = {i: [] for i in range(self.n_clusters)}

#         for _, student in self.compatibility_df.iterrows():
#             ranked_groups = student[1:-3].sort_values(ascending=False).index
#             for group in ranked_groups:
#                 group_index = int(group.split(" ")[1]) - 1
#                 if len(final_groups[group_index]) < self.max_group_size:
#                     final_groups[group_index].append(student["Tên"])
#                     break

#         final_result = []
#         for group, members in final_groups.items():
#             for member in members:
#                 row = self.compatibility_df[self.compatibility_df["Tên"] == member].iloc[0]
#                 final_result.append({
#                     "Tên": row["Tên"],
#                     "Nhóm": group + 1,
#                     "Độ tương thích": row[1:self.n_clusters+1].to_list()  # Chuyển độ tương thích thành mảng
#                     # "Kỹ năng lập trình": row["Kỹ năng lập trình"],
#                     # "Kỹ năng làm việc nhóm": row["Kỹ năng làm việc nhóm"],
#                     # "Sở thích làm việc": row["Sở thích làm việc"]
#                 })

#         self.final_df = pd.DataFrame(final_result)

#     def to_json(self, output_file):
#         """Xuất kết quả ra file JSON."""
#         self.final_df.to_json(output_file, orient="records", force_ascii=False, indent=4)
    
#     def save_model(self, file_name="kmeans_model.pkl"):
#         """Lưu model KMeans ra file."""
#         if self.kmeans:
#             joblib.dump(self.kmeans, file_name)
#             print(f"Model đã được lưu vào file {file_name}")
#         else:
#             raise ValueError("Model chưa được huấn luyện. Vui lòng chạy cluster_data() trước khi lưu.")
        
#     def load_model(self, file_name="kmeans_model.pkl"):
#         """Tải model KMeans từ file."""
#         try:
#             self.kmeans = joblib.load(file_name)
#             print(f"Model đã được tải từ file {file_name}")
#         except FileNotFoundError:
#             raise FileNotFoundError(f"File {file_name} không tồn tại. Vui lòng kiểm tra đường dẫn.")
        
#     def run(self, model_file=None, save_model_file=None):
#         """Thực hiện toàn bộ quy trình."""
#         self.load_data()
#         self.preprocess_data()
#         # self.shuffle_data()

#         # Sử dụng dữ liệu đã đảo
#         # data_to_use = self.shuffled_df  # Chọn dữ liệu đã đảo lộn
        
#         if model_file:  # Nếu có model file, tải model
#             self.load_model(model_file)
#         else:
#             self.cluster_data()  # Huấn luyện model
        
#         self.calculate_compatibility()
#         self.assign_groups()
        
#         if save_model_file:  # Nếu có yêu cầu lưu model
#             self.save_model(save_model_file)
        
#         return self.final_df
    

#     def add_student_and_predict(self, array):
#         if not self.kmeans:
#             raise ValueError("Model chưa được tải hoặc huấn luyện. Vui lòng tải model hoặc huấn luyện trước.")
        
#         name = array[0]
#         qt_score = array[1]  # Điểm tổng kết môn QT HTTT
#         data_mining_skill = array[2]  # Khai phá dữ liệu
#         machine_learning_skill = array[3]  # Học máy
#         work_preference = array[4]  # Sở thích làm việc

#         # Chuẩn bị dữ liệu mới
#         new_student_df = pd.DataFrame([{
#             "Tên": name,
#             "Điểm tổng kết môn QT HTTT": qt_score,
#             "Khai phá dữ liệu": data_mining_skill,
#             "Học máy": machine_learning_skill,
#             "Sở thích": work_preference,
#             "Kĩ năng làm việc": "",  # Giá trị này sẽ được mã hóa sau
#             "Hoạt động chính": "",  # Giá trị này sẽ được mã hóa sau
#             "Nguồn thông tin": ""
#         }])
        
#         # Mã hóa các giá trị mới
#         try:
#             # Mã hóa các giá trị phân loại cho các cột như "Kĩ năng làm việc" và "Hoạt động chính"
#             new_student_df["Kĩ năng làm việc"] = self.teamwork_encoder.transform([array[5]])[0]
#             new_student_df["Hoạt động chính"] = self.work_preference_encoder.transform([array[6]])[0]
#             new_student_df["Nguồn thông tin"] = self.work_preference_encoder.transform([array[7]])[0]
#         except ValueError:
#             print("Lỗi: Giá trị mới không có trong dữ liệu gốc. Sử dụng giá trị mặc định.")
#             new_student_df["Kĩ năng làm việc"] = -1  # Giá trị mặc định
#             new_student_df["Hoạt động chính"] = -1  # Giá trị mặc định
#             new_student_df["Nguồn thông tin"] = -1 
        
#         # Chuẩn hóa dữ liệu
#         features = ["Điểm tổng kết môn QT HTTT", "Khai phá dữ liệu", "Học máy", "Sở thích", "Kĩ năng làm việc", "Hoạt động chính", "Nguồn thông tin"]
#         new_student_scaled = self.scaler.transform(new_student_df[features])
        
#         # Dự đoán nhóm
#         group = self.kmeans.predict(new_student_scaled)[0] + 1
#         distances = self.kmeans.transform(new_student_scaled)[0]
#         compatibility = (1 / (distances + 1e-10)) / sum(1 / (distances + 1e-10)) * 100
        
#         return {
#             "Tên": name,
#             "Nhóm được phân": group,
#             "Độ tương thích": compatibility.round(2).tolist(),
#             "Thông tin gốc": {
#                 "Tên": name,
#                 "Điểm tổng kết môn QT HTTT": qt_score,
#                 "Khai phá dữ liệu": data_mining_skill,
#                 "Học máy": machine_learning_skill,
#                 "Sở thích": work_preference,
#                 "Kĩ năng làm việc": array[5],  # Kỹ năng làm việc
#                 "Hoạt động chính": array[6],  # Hoạt động chính
#                 "Nguồn thông tin": array[7]
#             }
#         }

import json
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

class TeamClustering:
    def __init__(self, data, n_clusters=4, max_group_size=5):
        self.df = data
        self.n_clusters = n_clusters
        self.max_group_size = max_group_size
        self.scaled_features = None
        self.kmeans = None
        self.compatibility_df = None
        self.final_df = None
    def convert_to_grade(self,score):
        try:
            score = float(score)
        except ValueError:
            return 'Không hợp lệ'
    
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
    def encode_score(self):
        pass
    def get_encoder(self):
        return self.work_preference_encoder, self.teamwork_encoder, self.scaler
    def preprocess_data(self):
        """Mã hóa và chuẩn hóa dữ liệu."""
        self.teamwork_encoder = LabelEncoder()
        self.work_preference_encoder = LabelEncoder()
        # Điểm tổng kết môn QT HTTT,Khai phá dữ liệu,Học máy,Sở thích,Kĩ năng làm việc,Hoạt động chính

        # Mã hóa các cột phân loại
        self.df["Sở thích"] = self.work_preference_encoder.fit_transform(self.df["Sở thích"])
        self.df["Kĩ năng làm việc"] = self.teamwork_encoder.fit_transform(self.df["Kĩ năng làm việc"])
        self.df["Hoạt động chính"] = self.work_preference_encoder.fit_transform(self.df["Hoạt động chính"])
        self.df["Nguồn thông tin"] = self.work_preference_encoder.fit_transform(self.df["Nguồn thông tin"])

        # Chuyển các điểm tổng kết thành giá trị số nếu cần
        score_mapping = {"Kém": 1, "Yếu": 2, "Trung bình": 3, "Khá": 3, "Giỏi": 5}
        self.df["Điểm tổng kết môn QT HTTT"] = self.df["Điểm tổng kết môn QT HTTT"].map(score_mapping)
        self.df["Khai phá dữ liệu"] = self.df["Khai phá dữ liệu"].map(score_mapping)
        self.df["Học máy"] = self.df["Học máy"].map(score_mapping)

        # Chọn các cột để chuẩn hóa
        features = ["Điểm tổng kết môn QT HTTT", "Khai phá dữ liệu", "Học máy", "Sở thích", "Kĩ năng làm việc", "Hoạt động chính", "Nguồn thông tin"]
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.df[features])

    def cluster_data(self):
        """Phân cụm dữ liệu bằng KMeans."""
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
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
            "Kĩ năng làm việc": "",  # Giá trị này sẽ được mã hóa sau
            "Hoạt động chính": "",  # Giá trị này sẽ được mã hóa sau
            "Nguồn thông tin": ""
        }])
        
        # Mã hóa các giá trị mới
        try:
            # Mã hóa các giá trị phân loại cho các cột như "Kĩ năng làm việc" và "Hoạt động chính"
            new_student_df["Kĩ năng làm việc"] = self.teamwork_encoder.transform([array[5]])[0]
            new_student_df["Hoạt động chính"] = self.work_preference_encoder.transform([array[6]])[0]
            new_student_df["Nguồn thông tin"] = self.work_preference_encoder.transform([array[7]])[0]
        except ValueError:
            print("Lỗi: Giá trị mới không có trong dữ liệu gốc. Sử dụng giá trị mặc định.")
            new_student_df["Kĩ năng làm việc"] = -1  # Giá trị mặc định
            new_student_df["Hoạt động chính"] = -1  # Giá trị mặc định
            new_student_df["Nguồn thông tin"] = -1 
        
        # Chuẩn hóa dữ liệu
        features = ["Điểm tổng kết môn QT HTTT", "Khai phá dữ liệu", "Học máy", "Sở thích", "Kĩ năng làm việc", "Hoạt động chính", "Nguồn thông tin"]
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