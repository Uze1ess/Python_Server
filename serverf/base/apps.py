from django.apps import AppConfig
import numpy as np
import pandas as pd

team_clustering = None
final_df = None

class BaseConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'base'
    def ready(self):
        global team_clustering, final_df

        # data_path = 'D:\công việc\H\He thong kinh doanj thong minh\ServerTest\serverf\static\group_3_cleaned.csv'
        # data = pd.read_csv(data_path)

        # try:
        #     # Đọc dữ liệu và khởi tạo mô hình
        #     kmean = TeamClustering(data, 4)

        #     # Huấn luyện mô hình
        #     kmean.read_data()
        #     kmean.encoder()
        #     kmean.train()
        #     print("Kmean model initialized and trained successfully!")
        
        # except Exception as e:
        #     print(f"Error initializing Kmean model: {str(e)}")