from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
import numpy as np
import pandas as pd
from .apps import team_clustering
from .kmean import TeamClustering
import json
from .getMark import StudentAPI, AccountProcessor, JSONHandler
from pymongo import MongoClient

test_student_tlu = []
students_data = []
data_path = 'D:\công việc\H\He thong kinh doanj thong minh\ServerTest\serverf\static\group_3_cleaned.csv'
login_url = "https://sinhvien1.tlu.edu.vn/education/oauth/token"
api_url = "https://sinhvien1.tlu.edu.vn/education/api/studentsubjectmark/getListStudentMarkBySemesterByLoginUser/0"
headers = {
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Cookie': 'SESSION=7a2b85c3-79cb-4527-bea7-d6d8f67e7300',
    'Origin': 'https://sinhvien1.tlu.edu.vn',
    'Referer': 'https://sinhvien1.tlu.edu.vn/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
}
team_clustering = None

# Create your views here.
def home(request):
    name = request.GET.get('name')

    response_data = {
        'name': name
    }
    
    # Trả về JSON response
    return JsonResponse(response_data)

@csrf_exempt
def getMark(request):
    global login_url, api_url, headers, test_student_tlu

    if request.method == "POST":
        try:
            subject_codes = ['CSE405', 'CSE404', 'CSE445']
            diff_subject_codes = ['CSE481', 'CSE486', 'CSE492']

            body = json.loads(request.body)
            student_api = StudentAPI(login_url, api_url, headers)
            json_handler = JSONHandler()

            account_processor = AccountProcessor(student_api, json_handler)
            account_processor.process_accounts(body['tluuserName'] , body['tlupassWord'], subject_codes, diff_subject_codes)

            result_json = account_processor.get_result_json()
            test_student_tlu.append(result_json)
            return JsonResponse({"message": "Student data get successfully.", "data": result_json}, status=201)
            
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        
        except Exception as e:
            return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)
    elif request.method == "GET":
        if students_data:
            # Trả về phần tử đầu tiên của danh sách (bao gồm cluster và similarity_index)
            return JsonResponse(test_student_tlu[0], status=200)
        else:
            # Trả về thông báo nếu không có dữ liệu nào
            return JsonResponse({"error": "No data available"}, status=404)


def train(request):
    global team_clustering, final_df

    #Tạo data frame từ mongo
    client = MongoClient("mongodb://localhost:27017/")
    db = client['Test'] 
    collection = db['train_set_test']

    fields = {"Tên": 1, "Điểm tổng kết môn QT HTTT": 1, "Khai phá dữ liệu": 1, "Học máy": 1, "Sở thích": 1, "Kĩ năng làm việc": 1,"Hoạt động chính": 1, "Nguồn thông tin": 1 , "_id": 0}
    documents = collection.find({}, fields)

    df = pd.DataFrame(list(documents))

    team_clustering = TeamClustering(df, n_clusters=4, max_group_size=5)

    final_df = team_clustering.run()

    final_array = final_df.to_dict(orient="records")  # Chuyển DataFrame thành danh sách các dict

    return JsonResponse({"infor": "Model trained"})

@csrf_exempt
def post_student_data(request):
    global students_data, final_df, checking_item

    if request.method == "POST":
        checking_item = None
        students_data.clear()

        try:
            checking_item = None
            body = json.loads(request.body)
            required_fields = ['name','score_QT_HTTT','score_data_mining','score_machine_learning','interests','work_skill','prior_activity','data_source']
            if not all(field in body for field in required_fields):
                return JsonResponse({"error": "Missing required fields in request body"}, status=400)
            # Kiểm tra nếu tìm thấy dòng có tên 'Trần Đức Thắng'
            checking_item = final_df[final_df["Tên"] == body['name']]

            name = str(checking_item["Tên"].values[0])
            similarity_scores = list(map(float, checking_item["Độ tương thích"].values[0]))
            suitable_group = int(checking_item["Nhóm"].values[0])
            highest_similarity_score = max(similarity_scores)

            body = {
                "name": name,
                "similarity_scores": {
                    "group_1": {
                        "similarity_score": similarity_scores[0]
                    },
                    "group_2": {
                        "similarity_score": similarity_scores[1]
                    },
                    "group_3": {
                        "similarity_score": similarity_scores[2]
                    },
                    "group_4": {
                        "similarity_score": similarity_scores[3]
                    },
                },
                "suitable_group": suitable_group,
                "highest_similarity_score": highest_similarity_score
            }
            students_data.append(body)

            return JsonResponse({"message": "Data processed successfully.", "data": body}, status=201)
           
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        
        except Exception as e:
            return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)
        
    elif request.method == "GET":
        if students_data:
            # Trả về phần tử đầu tiên của danh sách (bao gồm cluster và similarity_index)
            return JsonResponse(students_data[0], status=200)
        else:
            # Trả về thông báo nếu không có dữ liệu nào
            return JsonResponse({"error": "No data available"}, status=404)

    return JsonResponse({"error": "Method not allowed"}, status=405)

